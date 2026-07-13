#!/usr/bin/env python3
"""
Nexus → plantclef adapter shim
==============================

Translates the Nexus control-plane launch contract into a single-node
``torchrun`` invocation of the real BioCLIP 2.5 multi-task trainer at
``shared/bioclip25_multitask/train.py``.

Why a shim?
-----------
The Nexus pod-agent spawner (``~/Nexus/scripts/run_spawner.py``) launches a
trainer with a fixed set of injected CLI flags and environment variables that
the real trainer does not understand. This shim consumes the Nexus-specific
flags, maps them onto the plantclef telemetry env vars, and forwards
everything else verbatim to ``train.py``.

Contract (must stay in sync with run_spawner.py / run_monitor.py)
-----------------------------------------------------------------
Spawner-injected CLI flags (consumed here, NOT forwarded):
    --name --phase --seed --telemetry-dir [--dataset <path>] [--smoke]
Everything else on the command line passes through to the trainer untouched
(this includes ``spec.extra_args`` such as ``--batch-size``, ``--epochs``,
``--full-finetune``, ``--metadata-csv``, ``--output-dir``, ``--resume`` …).

``--smoke`` (consumed): instead of torchrun + the real trainer, launch the
torch-free fake trainer ``<repo>/src/nexus_smoke.py`` under ``sys.executable``
(the smoke script is stdlib-only, so whatever python runs this shim works)
with the same env mapping and cwd, propagating its exit code.

Spawner-injected env (read here):
    ORACLE_RUN_ID_TEMPLATE (contains literal ``{rank}``), ORACLE_HOST_ID,
    ORACLE_SEED, ORACLE_DATASET_PATH, …

Env mapping applied for the trainer / torchrun workers:
    PLANTCLEF_RUN_ID_TEMPLATE = ORACLE_RUN_ID_TEMPLATE   (verbatim — keeps the
                                 ``oracle_…_r{rank}_…`` stem so telemetry
                                 filenames match the Nexus ``oracle_*.jsonl``
                                 glob and the ``_r<rank>_`` rank-strip)
    PLANTCLEF_TELEMETRY_DIR   = <--telemetry-dir>
    PLANTCLEF_SEED            = <--seed>
    PLANTCLEF_NAME            = <--name>
    PLANTCLEF_PHASE           = <--phase>
    CLUSTER_HOST_ID           = ORACLE_HOST_ID
    PLANTCLEF_TELEMETRY       = file
    PYTHONPATH                = <plantclef repo root>  (so ``src.training.
                                 telemetry`` imports inside torchrun workers)

Argument translation:
    --dataset <path>  →  --train-image-root <path>   (appended only if the
                          trainer didn't already get --train-image-root)
    no --output-dir   →  inject  <repo>/reports/run_outputs/<rank0-run-id>
                          (the spawner normally supplies --output-dir; this is a
                          fallback for manual use so checkpoints are per-run
                          addressable)

Launch:
    torchrun --standalone --nproc_per_node=<N>
             <repo>/shared/bioclip25_multitask/train.py <passthrough args>
where N = $PLANTCLEF_NPROC if set, else CUDA device count, else 1. Worker
stderr is *inherited* (no torchrun --redirects) so a CUDA OOM message
propagates to the parent for Nexus's stderr capture / OOM classification.

torchrun launcher resolution (non-smoke runs), in priority order:
    1. $PLANTCLEF_TORCHRUN (explicit override)
    2. <repo>/.venv/bin/torchrun (if it exists)
    3. shutil.which("torchrun")
If none resolve: one clear line to stderr, exit 3 — before spawning. In
dry-run (``--print-cmd`` / PLANTCLEF_DRYRUN=1) an unresolved launcher prints
the placeholder ``torchrun`` instead, so dry-runs work on torch-less boxes.

Usage
-----
As a Nexus pod trainer command::

    pod_agent --trainer-cmd /home/amaterasu/Research/plantclef/src/nexus_entry.py

Manual / debug (no launch, just print the resolved argv + env mapping)::

    PLANTCLEF_DRYRUN=1 python src/nexus_entry.py --name foo --phase p2a \\
        --seed 42 --telemetry-dir /tmp/tele --batch-size 32
    # or
    python src/nexus_entry.py --print-cmd --name foo --phase p2a ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Repo root is the parent of this file's directory (src/nexus_entry.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "shared" / "bioclip25_multitask" / "train.py"
SMOKE_SCRIPT = REPO_ROOT / "src" / "nexus_smoke.py"


def _parse_args(argv: List[str]):
    p = argparse.ArgumentParser(
        prog="nexus_entry",
        description="Nexus → plantclef trainer adapter shim.",
        add_help=True,
    )
    # Spawner-injected flags consumed by the shim (NOT forwarded to trainer).
    p.add_argument("--name", default=None)
    p.add_argument("--phase", default=None)
    p.add_argument("--seed", default=None)
    p.add_argument("--telemetry-dir", dest="telemetry_dir", default=None)
    p.add_argument("--dataset", default=None)
    # Smoke mode: launch the torch-free fake trainer instead of torchrun.
    p.add_argument("--smoke", action="store_true")
    # Debug: print the resolved command + env mapping and exit without launch.
    p.add_argument("--print-cmd", dest="print_cmd", action="store_true")
    # Everything else passes through verbatim to the trainer.
    return p.parse_known_args(argv)


def _has_flag(passthrough: List[str], flag: str) -> bool:
    """True if ``flag`` (or ``flag=value``) is already present in passthrough."""
    return any(a == flag or a.startswith(flag + "=") for a in passthrough)


def _resolve_run_id_template(phase: str | None) -> str:
    """Return the per-rank run-id template (contains literal ``{rank}``).

    Prefers the spawner's ORACLE_RUN_ID_TEMPLATE verbatim; falls back to a
    locally-generated ``oracle_<phase>_r{rank}_<stamp>_<short>`` template for
    manual use so telemetry filenames still match the Nexus glob.
    """
    template = os.environ.get("ORACLE_RUN_ID_TEMPLATE")
    if template:
        return template
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:6]
    return f"oracle_{phase or 'run'}_r{{rank}}_{stamp}_{short}"


def _resolve_torchrun() -> Optional[str]:
    """Resolve the torchrun launcher robustly (a bare ``"torchrun"`` from
    PATH breaks when the pod-agent's venv isn't the trainer's venv).

    Priority: $PLANTCLEF_TORCHRUN → <repo>/.venv/bin/torchrun → PATH.
    Returns ``None`` if nothing resolves.
    """
    env_launcher = os.environ.get("PLANTCLEF_TORCHRUN")
    if env_launcher:
        return env_launcher
    venv_launcher = REPO_ROOT / ".venv" / "bin" / "torchrun"
    if venv_launcher.exists():
        return str(venv_launcher)
    return shutil.which("torchrun")


def _resolve_nproc() -> int:
    """nproc_per_node: $PLANTCLEF_NPROC, else CUDA device count, else 1."""
    env_n = os.environ.get("PLANTCLEF_NPROC")
    if env_n:
        try:
            return max(1, int(env_n))
        except ValueError:
            pass
    try:
        import torch  # noqa: WPS433 (optional dependency on dev boxes)
        count = torch.cuda.device_count()
        if count and count > 0:
            return int(count)
    except Exception:
        pass
    return 1


def main(argv: List[str] | None = None) -> int:
    args, passthrough = _parse_args(list(sys.argv[1:] if argv is None else argv))
    passthrough = list(passthrough)

    # ── run-id template + env mapping ────────────────────────────────────────
    template = _resolve_run_id_template(args.phase)
    run_id = template.replace("{rank}", "0")

    existing_pp = os.environ.get("PYTHONPATH", "")
    pythonpath = str(REPO_ROOT) + (os.pathsep + existing_pp if existing_pp else "")

    mapping: dict[str, str] = {
        "PLANTCLEF_RUN_ID_TEMPLATE": template,   # verbatim ORACLE_RUN_ID_TEMPLATE
        "PLANTCLEF_TELEMETRY": "file",
        "PYTHONPATH": pythonpath,
    }
    if args.telemetry_dir:
        mapping["PLANTCLEF_TELEMETRY_DIR"] = args.telemetry_dir
    seed = args.seed if args.seed is not None else os.environ.get("ORACLE_SEED")
    if seed is not None:
        mapping["PLANTCLEF_SEED"] = str(seed)
    if args.name:
        mapping["PLANTCLEF_NAME"] = args.name
    if args.phase:
        mapping["PLANTCLEF_PHASE"] = args.phase
    host_id = os.environ.get("ORACLE_HOST_ID")
    if host_id:
        mapping["CLUSTER_HOST_ID"] = host_id

    # ── argument translation ─────────────────────────────────────────────────
    # --dataset <path> → --train-image-root <path> (unless already present).
    dataset = args.dataset or os.environ.get("ORACLE_DATASET_PATH")
    if dataset and not _has_flag(passthrough, "--train-image-root"):
        passthrough += ["--train-image-root", dataset]

    # Fallback --output-dir under reports/run_outputs/<run_id> (spawner normally
    # supplies one explicitly; respect that if present).
    if not _has_flag(passthrough, "--output-dir"):
        out_dir = REPO_ROOT / "reports" / "run_outputs" / run_id
        passthrough += ["--output-dir", str(out_dir)]

    # ── build launch command ─────────────────────────────────────────────────
    dry_run = args.print_cmd or os.environ.get("PLANTCLEF_DRYRUN") == "1"
    if args.smoke:
        # Torch-free smoke trainer: plain python, no torchrun. The smoke
        # script is stdlib-only, so the python running this shim suffices.
        script = SMOKE_SCRIPT
        cmd = [sys.executable, str(SMOKE_SCRIPT), *passthrough]
    else:
        script = TRAIN_SCRIPT
        launcher = _resolve_torchrun()
        if launcher is None:
            if dry_run:
                launcher = "torchrun"  # placeholder so dry-runs work sans torch
            else:
                print(
                    "[nexus_entry] torchrun not found; set PLANTCLEF_TORCHRUN "
                    f"or install torch in {REPO_ROOT}/.venv",
                    file=sys.stderr,
                )
                return 3
        nproc = _resolve_nproc()
        cmd = [
            launcher,
            "--standalone",
            f"--nproc_per_node={nproc}",
            str(TRAIN_SCRIPT),
            *passthrough,
        ]

    # ── dry-run / debug ──────────────────────────────────────────────────────
    if dry_run:
        for token in cmd:
            print(token)
        print(json.dumps(mapping, indent=2, sort_keys=True))
        return 0

    if not script.exists():
        print(f"[nexus_entry] trainer script not found: {script}", file=sys.stderr)
        return 2

    # ── launch ───────────────────────────────────────────────────────────────
    env = os.environ.copy()
    env.update(mapping)
    # Inherited stdio (no capture, no torchrun --redirects) so worker OOM text
    # reaches the parent for Nexus's stderr capture.
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
