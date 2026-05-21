"""
Compare inference runs side by side from their summary.json files.

Usage
-----
# Compare specific run directories
python compare_models.py outputs/bioclip_scientific outputs/bioclip-2_all

# Auto-discover all runs under a parent directory
python compare_models.py --parent-dir outputs/

# Output as CSV
python compare_models.py --parent-dir outputs/ --csv comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DISPLAY_KEYS = [
    "run_slug",
    "model_name",
    "prompt_mode",
    "n_species",
    "total_prompts",
    "avg_per_species",
    "n_images_processed",
    "n_errors",
    "top_k",
    "tile_size",
    "stride",
    "text_encoding_secs",
    "inference_total_secs",
    "inference_per_image_secs",
    "device",
    "batch_size",
]


def load_summary(path: Path) -> dict | None:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def discover_runs(parent_dir: Path) -> list[Path]:
    """Find all subdirectories containing a summary.json."""
    return sorted(p.parent for p in parent_dir.glob("*/summary.json"))


def format_table(summaries: list[dict]) -> str:
    if not summaries:
        return "(no runs found)"

    # Collect rows
    rows = []
    for s in summaries:
        row = {k: str(s.get(k, "")) for k in DISPLAY_KEYS}
        rows.append(row)

    # Column widths
    col_widths = {k: max(len(k), max(len(r[k]) for r in rows)) for k in DISPLAY_KEYS}

    def fmt_row(row: dict) -> str:
        return "  ".join(row[k].ljust(col_widths[k]) for k in DISPLAY_KEYS)

    sep = "  ".join("-" * col_widths[k] for k in DISPLAY_KEYS)
    header = "  ".join(k.ljust(col_widths[k]) for k in DISPLAY_KEYS)

    lines = [header, sep] + [fmt_row(r) for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BioCLIP zero-shot run summaries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help="Explicit run directories to compare",
    )
    parser.add_argument(
        "--parent-dir",
        default=None,
        help="Auto-discover all runs under this directory",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="If given, also save the comparison table to this CSV file",
    )
    args = parser.parse_args()

    # Collect paths
    paths: list[Path] = []
    if args.parent_dir:
        paths.extend(discover_runs(Path(args.parent_dir)))
    for d in args.run_dirs:
        p = Path(d)
        if not p.exists():
            print(f"[warn] directory not found, skipping: {p}")
            continue
        paths.append(p)

    if not paths:
        print("No run directories found. Pass run directories as arguments or use --parent-dir.")
        return

    # Load summaries
    summaries: list[dict] = []
    for p in paths:
        s = load_summary(p)
        if s is None:
            print(f"[warn] no summary.json in {p}, skipping")
            continue
        summaries.append(s)

    if not summaries:
        print("No valid summaries found.")
        return

    # Sort by model name then prompt mode for deterministic order
    summaries.sort(key=lambda s: (s.get("model_name", ""), s.get("prompt_mode", "")))

    print(f"\nComparison of {len(summaries)} run(s)\n")
    print(format_table(summaries))
    print()

    # Optional CSV output
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DISPLAY_KEYS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Comparison CSV saved: {out}")


if __name__ == "__main__":
    main()
