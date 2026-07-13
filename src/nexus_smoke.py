#!/usr/bin/env python3
"""
Nexus smoke trainer (torch-free)
================================

A fast (~15-30 s) fake trainer that exercises the full Nexus pipeline —
spawner → telemetry JSONL → run_monitor → dashboard — without torch, GPUs,
or the real dataset. Launched by ``src/nexus_entry.py --smoke``; can also be
run by hand.

It uses the REAL telemetry library (``src.training.telemetry``) so the JSONL
schema, run-id template handling (``PLANTCLEF_RUN_ID_TEMPLATE`` with
``{rank}``), sink selection (``PLANTCLEF_TELEMETRY`` / ``_DIR``) and
heartbeat daemon are identical to a real run.

Flags (``parse_known_args`` — unknown flags are ignored, so spawner-injected
trainer flags like ``--train-image-root`` pass through harmlessly):
    --output-dir <dir>       default ./reports/run_outputs/smoke (under cwd)
    --resume <path>          read ``{"step": N}`` and continue from N+1
                             (clamped to the final step so at least one
                             step event always precedes validation.end)
    --batch-size <int>       logged no-op (lets OOM auto-retry rescaling flow)
    --steps <int>            default 25
    --step-interval <float>  seconds per step, default 0.6
    --oom-at <int>           at step i, print a realistic CUDA OOM traceback
                             to stderr and exit 1 WITHOUT run.end

Behavior per run:
    bind(rank=0, world=1, phase=$PLANTCLEF_PHASE) → run.start (with a small
    ``snapshot`` dict) → start_heartbeat(5) → per step: ``step`` event
    (step, loss, local_acc), sleep, checkpoint ``{"step": i}`` to
    <output-dir>/checkpoints/last.pt every 5 steps → validation.end
    (acc=0.87) → run.end status=ok → exit 0.

Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_TELEMETRY_PATH = REPO_ROOT / "src" / "training" / "telemetry.py"


def _git_capture(cwd: Path) -> dict:
    """Best-effort git state for the run.start snapshot. Silent on failure.
    (Duplicated from shared/bioclip25_multitask/train.py — the smoke trainer
    deliberately imports nothing from the torch-importing training tree.)"""
    def _run(cmd_args):
        try:
            out = subprocess.check_output(
                ["git", "-C", str(cwd), *cmd_args],
                stderr=subprocess.DEVNULL, timeout=2.0,
            )
            return out.decode("utf-8", "replace").strip() or None
        except Exception:
            return None

    git_hash = _run(["rev-parse", "HEAD"])
    if git_hash is None:
        return {"git_hash": None, "dirty_diff_summary": None}
    shortstat = _run(["diff", "--shortstat", "HEAD"])
    return {"git_hash": git_hash, "dirty_diff_summary": shortstat or None}


def _env_hash() -> str:
    """Short stable hash over interpreter + installed package list."""
    try:
        import importlib.metadata as md
        dists = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in md.distributions()
            if d.metadata.get("Name")
        )
    except Exception:
        dists = []
    blob = "\n".join([platform.python_version(), platform.machine(), *dists])
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _load_telemetry():
    """Load the REAL telemetry lib straight from its file. We can't import
    ``src.training.telemetry`` the normal way: the ``src.training`` package
    ``__init__`` imports torch, and the smoke trainer must stay torch-free.
    ``telemetry.py`` itself is stdlib-only, so a direct file load is safe."""
    spec = importlib.util.spec_from_file_location(
        "plantclef_smoke_telemetry", _TELEMETRY_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_telemetry_mod = _load_telemetry()
telemetry = _telemetry_mod.telemetry
start_heartbeat = _telemetry_mod.start_heartbeat

VALIDATION_ACC = 0.87
CHECKPOINT_EVERY = 5
HEARTBEAT_INTERVAL_SEC = 5

_OOM_TRACEBACK = """\
Traceback (most recent call last):
  File "{repo}/shared/bioclip25_multitask/train.py", line 412, in <module>
    main()
  File "{repo}/shared/bioclip25_multitask/train.py", line 287, in main
    loss = model(images, labels)
  File "{repo}/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "{repo}/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB. \
GPU 0 has a total capacity of 23.69 GiB of which 1.06 GiB is free. Including \
non-PyTorch memory, this process has 22.10 GiB memory in use. Of the allocated \
memory 21.32 GiB is allocated by PyTorch, and 310.45 MiB is reserved by \
PyTorch but unallocated. If reserved but unallocated memory is large try \
setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation."""


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="nexus_smoke",
        description="Torch-free fake trainer for Nexus end-to-end smoke runs.",
    )
    p.add_argument("--output-dir", dest="output_dir",
                   default=str(Path.cwd() / "reports" / "run_outputs" / "smoke"))
    p.add_argument("--resume", default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--step-interval", dest="step_interval", type=float, default=0.6)
    p.add_argument("--oom-at", dest="oom_at", type=int, default=None)
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print(f"[nexus_smoke] ignoring unknown args: {unknown}", file=sys.stderr)
    return args


def _read_resume_step(path: str) -> int:
    """Return the step recorded in a ``{"step": N}`` checkpoint, or -1
    (start from 0) if the file is missing/malformed — warn, don't crash."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        step = payload["step"]
        if isinstance(step, int):
            return step
        raise ValueError(f"non-int step: {step!r}")
    except Exception as e:
        print(f"[nexus_smoke] could not read resume checkpoint {path}: {e}; "
              "starting from step 0", file=sys.stderr)
        return -1


def main(argv=None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    out_dir = Path(args.output_dir)
    ckpt_path = out_dir / "checkpoints" / "last.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume:
        start_step = _read_resume_step(args.resume) + 1
        # Guarantee at least one step event per process: run_monitor drops
        # validation.end until it has seen a step, so a checkpoint at/past
        # the final step must not skip the loop entirely.
        clamped = min(start_step, max(0, args.steps - 1))
        if clamped != start_step:
            print(f"[nexus_smoke] resume checkpoint is at/past the final step "
                  f"(start_step={start_step}, steps={args.steps}); clamping "
                  f"to step {clamped} so at least one step event is emitted",
                  file=sys.stderr)
            start_step = clamped

    # Single fake rank; phase comes from the shim's env mapping.
    telemetry.bind(
        rank=0,
        world=1,
        phase=os.environ.get("PLANTCLEF_PHASE", ""),
        host_id=os.environ.get("CLUSTER_HOST_ID"),
    )
    # Snapshot keys follow the canonical shape the Nexus Compare tab reads
    # (git_hash / env_hash / python_version / command / lr / steps / epochs —
    # see Nexus src/lib/runs.ts RunSnapshot); smoke-only extras ride along
    # and are ignored by the frontend projector.
    telemetry.emit("run.start", snapshot={
        **_git_capture(REPO_ROOT),
        "env_hash":       _env_hash(),
        "python_version": sys.version.split()[0],
        "config_yaml":    None,
        "command":        " ".join(shlex.quote(a) for a in sys.argv),
        "lr":             None,
        "steps":          args.steps,
        "epochs":         None,
        "mode":           "smoke",
        "step_interval":  args.step_interval,
        "batch_size":     args.batch_size,
        "resume_from":    args.resume,
    })
    start_heartbeat(HEARTBEAT_INTERVAL_SEC)

    if args.batch_size is not None:
        print(f"[nexus_smoke] batch_size={args.batch_size} (no-op)")

    for i in range(start_step, args.steps):
        telemetry.emit(
            "step",
            step=i,
            loss=2.0 * math.exp(-i / 10),
            local_acc=min(0.95, i / args.steps),
        )
        time.sleep(args.step_interval)
        if i % CHECKPOINT_EVERY == 0:
            ckpt_path.write_text(json.dumps({"step": i}), encoding="utf-8")
        if args.oom_at is not None and args.oom_at == i:
            # Simulate a mid-step CUDA OOM: realistic traceback on stderr
            # (matches the spawner's classifier markers), hard exit, and —
            # crucially — NO run.end so the monitor sees a dead run.
            print(_OOM_TRACEBACK.format(repo=REPO_ROOT), file=sys.stderr)
            sys.exit(1)

    telemetry.emit("validation.end", acc=VALIDATION_ACC)
    telemetry.emit("run.end", status="ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
