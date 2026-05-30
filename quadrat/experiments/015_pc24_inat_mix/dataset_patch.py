"""
015 — Additive patch for 010's dataset.resolve_image_paths.

Apply once on the pod before training:
    python -m src_experiments.015_pc24_inat_mix.dataset_patch \
        /workspace/working/workspace/PlantCLEF2026/src_experiments/010_bioclip25_end_to_end_finetune_multitask/dataset.py

What it does:
    Replaces the body of resolve_image_paths so that if an `image_path` column
    exists and the row's value is a non-empty string, that absolute path is
    used directly. Otherwise the original {root}/{species_id}/{image_name}
    resolution is used. Pure additive — PC24-only runs are unaffected.

Idempotent: detects the patched body and exits without re-patching.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


PATCHED_MARKER = "# 015-PATCH: respect explicit image_path column"

NEW_BODY = '''def resolve_image_paths(df: pd.DataFrame, image_root: str) -> pd.DataFrame:
    # 015-PATCH: respect explicit image_path column
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")
    df = df.copy()
    has_explicit = "image_path" in df.columns
    if has_explicit:
        def _resolve(r):
            v = r.get("image_path")
            if isinstance(v, str) and v.strip():
                return v.strip()
            return str(root / str(r["species_id"]) / str(r["image_name"]))
    else:
        def _resolve(r):
            return str(root / str(r["species_id"]) / str(r["image_name"]))
    df["resolved_path"] = df.apply(_resolve, axis=1)
    return df
'''


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path/to/010/dataset.py>", file=sys.stderr)
        return 2
    target = Path(sys.argv[1])
    if not target.is_file():
        print(f"Not a file: {target}", file=sys.stderr)
        return 1
    src = target.read_text()

    if PATCHED_MARKER in src:
        print(f"Already patched: {target}")
        return 0

    pattern = re.compile(
        r"def resolve_image_paths\(df: pd\.DataFrame, image_root: str\) -> pd\.DataFrame:\n"
        r"(?:    .*\n)+",
        re.MULTILINE,
    )
    match = pattern.search(src)
    if not match:
        print("Could not find resolve_image_paths function — aborting.", file=sys.stderr)
        return 1
    patched = src[: match.start()] + NEW_BODY + src[match.end():]
    backup = target.with_suffix(".py.bak_pre015")
    backup.write_text(src)
    target.write_text(patched)
    print(f"Patched: {target}  (backup at {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
