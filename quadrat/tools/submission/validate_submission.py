#!/usr/bin/env python3
"""Sanity-check a PlantCLEF submission CSV before upload and compare against a baseline.

Runs six validation passes:
  1. Format: header, separators, parseable rows
  2. Coverage: every test-CSV image has a row, no extras, no duplicates
  3. Predictions: per-image counts (min/max/mean), distribution buckets
  4. Species: every predicted ID exists in species_ids_mapping.csv
  5. Health: detects collapsed-model or random-baseline patterns
  6. Baseline Comparison (Optional): Computes IoU, Exact Matches, and species distribution shifts against an older submission.

Exit code 0 on success, 1 if any critical check fails.

Usage:
    ./scripts/validate_submission.py                          # default ./submission.csv
    ./scripts/validate_submission.py path/to/submission.csv
    ./scripts/validate_submission.py --baseline path/to/008_submission.csv
    ./scripts/validate_submission.py --samples 20             # show 20 random predictions
"""
from __future__ import annotations
import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path


_GREEN = "\033[92m"
_RED   = "\033[91m"
_YEL   = "\033[93m"
_CYAN  = "\033[96m"
_RST   = "\033[0m"


def _ok(msg):    print(f"  {_GREEN}✓{_RST} {msg}")
def _warn(msg):  print(f"  {_YEL}⚠{_RST}  {msg}")
def _err(msg):   print(f"  {_RED}✗{_RST} {msg}")
def _info(msg):  print(f"  {_CYAN}ℹ{_RST} {msg}")


def parse_predictions(pred_str: str) -> set[str]:
    return set(pred_str.replace('[', '').replace(']', '').replace(',', ' ').split())


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a PlantCLEF submission CSV and compare with a baseline.")
    ap.add_argument("submission", nargs="?", default="submission.csv",
                    help="Path to submission CSV (default: ./submission.csv)")
    ap.add_argument("--baseline", type=str, default=None,
                    help="Path to a baseline submission CSV to compare against (e.g., 008_submission.csv)")
    ap.add_argument("--test-csv", default="/workspace/plantclef/kaggle_uploads/test/PlantCLEF2025_test.csv")
    ap.add_argument("--species-mapping", default="/workspace/plantclef/processed/species_ids_mapping.csv")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    if args.no_color:
        global _GREEN, _RED, _YEL, _CYAN, _RST
        _GREEN = _RED = _YEL = _CYAN = _RST = ""

    sub_path = Path(args.submission)
    if not sub_path.is_file():
        print(f"{_RED}FATAL: submission not found: {sub_path}{_RST}", file=sys.stderr)
        return 1

    print(f"\n{'═' * 64}")
    print(f"  Validating: {sub_path.resolve()}")
    print(f"  Size: {sub_path.stat().st_size:,} bytes")
    print(f"{'═' * 64}")

    issues: list[str] = []

    # ── 1. Format & parsing ──────────────────────────────────────────────────
    print("\n[1/6] Format & parsing")
    rows: dict[str, str] = {}
    header: list[str] | None = None
    try:
        with open(sub_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for line_no, row in enumerate(reader, start=2):
                if len(row) < 2:
                    issues.append(f"Line {line_no}: only {len(row)} columns")
                    continue
                img_id = row[0].strip()
                if img_id in rows:
                    issues.append(f"Duplicate observation_id: {img_id}")
                rows[img_id] = row[1].strip()
    except Exception as e:
        _err(f"CSV parse error: {e}")
        return 1

    _ok(f"Header: {header}")
    _ok(f"Data rows: {len(rows):,}")
    if not header or len(header) < 2:
        _err("Missing or malformed header")
        issues.append("Bad header")
    if not rows:
        _err("Zero data rows")
        issues.append("Empty submission")
        return 1

    empty_pred = sum(1 for p in rows.values() if not p)
    if empty_pred:
        _err(f"{empty_pred} rows have empty species_ids")
        issues.append(f"{empty_pred} empty predictions")
    else:
        _ok("No empty prediction rows")

    # ── 2. Coverage vs test CSV ──────────────────────────────────────────────
    print("\n[2/6] Coverage vs test CSV")
    if not Path(args.test_csv).is_file():
        _warn(f"Test CSV not found at {args.test_csv} — skipping")
    else:
        with open(args.test_csv, newline="") as f:
            tr = csv.DictReader(f, delimiter=";")
            id_col = next((c for c in ("quadrat_id", "plot_id", "observation_id", "image_id")
                           if tr.fieldnames and c in tr.fieldnames), None)
            if not id_col:
                _err(f"Test CSV columns {tr.fieldnames} have no recognized id column")
                issues.append("Test CSV id column not found")
                expected: set[str] = set()
            else:
                expected = {row[id_col].strip() for row in tr if row.get(id_col)}

        if expected:
            submitted = set(rows.keys())
            missing  = expected - submitted
            extra    = submitted - expected
            _ok(f"Expected: {len(expected):,}    Submitted: {len(submitted):,}")
            if missing:
                _err(f"Missing from submission: {len(missing)} (e.g. {sorted(missing)[:3]})")
                issues.append(f"{len(missing)} test images have no prediction")
            else:
                _ok("All expected IDs covered")
            if extra:
                _warn(f"Extra IDs not in test CSV: {len(extra)} (e.g. {sorted(extra)[:3]})")
                issues.append(f"{len(extra)} extra IDs in submission")

    # ── 3. Prediction count distribution ─────────────────────────────────────
    print("\n[3/6] Prediction counts per image")
    parsed_preds = {img_id: parse_predictions(p) for img_id, p in rows.items()}
    counts = [len(preds) for preds in parsed_preds.values()]
    
    if counts:
        srt = sorted(counts)
        median = srt[len(srt) // 2]
        mean = sum(counts) / len(counts)
        bucket = Counter()
        for c in counts:
            if c == 0:    bucket["0"]    += 1
            elif c == 1:  bucket["1"]    += 1
            elif c <= 5:  bucket["2-5"]  += 1
            elif c <= 10: bucket["6-10"] += 1
            else:         bucket[">10"]  += 1
        _ok(f"min={min(counts)}  max={max(counts)}  mean={mean:.2f}  median={median}")
        _ok(f"distribution: {dict(bucket)}")

        if min(counts) == 0:
            issues.append("At least one row has 0 predictions")
        if max(counts) > 50:
            _warn(f"Some rows have >50 predictions — possible threshold too low or NMS failure")
        if mean < 0.5:
            issues.append(f"Mean predictions ({mean:.2f}) suspiciously low")

    # ── 4. Species ID validity ───────────────────────────────────────────────
    print("\n[4/6] Species ID validity")
    all_preds: list[str] = []
    for preds in parsed_preds.values():
        all_preds.extend(preds)
    unique_preds = set(all_preds)
    _ok(f"Total predictions made: {len(all_preds):,}")
    _ok(f"Unique species predicted globally: {len(unique_preds):,}")

    if Path(args.species_mapping).is_file():
        valid_ids: set[str] = set()
        with open(args.species_mapping, newline="") as f:
            for row in csv.reader(f):
                if row:
                    valid_ids.add(row[0].strip())
        invalid = unique_preds - valid_ids
        if invalid:
            _err(f"Invalid species IDs: {len(invalid)} (e.g. {sorted(invalid)[:5]})")
            issues.append(f"{len(invalid)} predicted species not in mapping")
        else:
            _ok(f"All {len(unique_preds):,} unique species are valid")
    else:
        _warn(f"Species mapping not found at {args.species_mapping} — skipping")

    # ── 5. Health: collapse / randomness signals ─────────────────────────────
    print("\n[5/6] Model-health heuristics")
    species_freq = Counter(all_preds)
    if species_freq:
        n_test = len(rows) or 1
        top_5_share = sum(c for _, c in species_freq.most_common(5)) / max(len(all_preds), 1)
        if len(unique_preds) < 100:
            _warn(f"Only {len(unique_preds)} distinct species predicted — model might be severely collapsed")
        elif top_5_share > 0.5:
            _warn(f"Top-5 species account for {top_5_share*100:.1f}% of predictions — distribution heavily skewed")
        else:
            _ok(f"Top-5 species share: {top_5_share*100:.1f}% (healthy spread)")

        # Per-image dominance: is every row predicting the same one species?
        # Get the first prediction for each row (assuming it's the highest confidence)
        first_preds = [list(preds)[0] for preds in parsed_preds.values() if preds]
        first_pred_freq = Counter(first_preds)
        if first_pred_freq:
            top_first, top_first_count = first_pred_freq.most_common(1)[0]
            ratio = top_first_count / n_test
            if ratio > 0.30:
                _warn(f"Top-1 species '{top_first}' is the lead prediction "
                      f"on {ratio*100:.1f}% of images")
            else:
                _ok(f"Top-1 lead prediction frequency: {ratio*100:.1f}% (looks distributed)")

    # ── 6. Baseline Comparison ───────────────────────────────────────────────
    print("\n[6/6] Baseline Comparison")
    if args.baseline:
        base_path = Path(args.baseline)
        if not base_path.is_file():
            _warn(f"Baseline file not found at {base_path} — skipping")
        else:
            baseline_rows: dict[str, str] = {}
            try:
                with open(base_path, newline="") as f:
                    reader = csv.reader(f)
                    next(reader, None) # skip header
                    for row in reader:
                        if len(row) >= 2:
                            baseline_rows[row[0].strip()] = row[1].strip()
            except Exception as e:
                _err(f"Baseline CSV parse error: {e}")
            
            if baseline_rows:
                base_parsed = {img_id: parse_predictions(p) for img_id, p in baseline_rows.items()}
                
                common_ids = set(rows.keys()).intersection(set(baseline_rows.keys()))
                _info(f"Comparing {len(common_ids):,} common images against baseline '{base_path.name}'")
                
                exact_matches = 0
                total_iou = 0.0
                
                base_all_preds = []
                for img_id in common_ids:
                    new_p = parsed_preds[img_id]
                    base_p = base_parsed[img_id]
                    base_all_preds.extend(base_p)
                    
                    if new_p == base_p:
                        exact_matches += 1
                        
                    intersection = len(new_p.intersection(base_p))
                    union = len(new_p.union(base_p))
                    if union > 0:
                        total_iou += (intersection / union)
                        
                mean_iou = total_iou / len(common_ids) if common_ids else 0.0
                exact_match_ratio = exact_matches / len(common_ids) if common_ids else 0.0
                
                _ok(f"Jaccard Similarity (Mean IoU): {mean_iou:.4f}")
                _ok(f"Exact Matches: {exact_matches:,} ({exact_match_ratio*100:.1f}%)")
                
                base_unique = len(set(base_all_preds))
                new_unique = len(unique_preds)
                _info(f"Unique Species: Baseline had {base_unique:,}, New has {new_unique:,} ({'+' if new_unique > base_unique else ''}{new_unique - base_unique})")
                
                # Distribution Shifts
                base_freq = Counter(base_all_preds)
                new_freq = species_freq
                
                gained = {s: new_freq.get(s, 0) - base_freq.get(s, 0) for s in set(new_freq) | set(base_freq)}
                top_gains = sorted(gained.items(), key=lambda x: x[1], reverse=True)[:5]
                top_drops = sorted(gained.items(), key=lambda x: x[1])[:5]
                
                print(f"  {_CYAN}Top Species Volume Gained:{_RST} " + ", ".join([f"{s} (+{v})" for s, v in top_gains if v > 0]))
                print(f"  {_CYAN}Top Species Volume Dropped:{_RST} " + ", ".join([f"{s} ({v})" for s, v in top_drops if v < 0]))

    else:
        _info("No --baseline provided. Skipping comparison.")

    # ── Random sample for eyeballing ─────────────────────────────────────────
    print(f"\n  Random sample ({args.samples} rows):")
    random.seed(42)
    sample = random.sample(list(rows.items()), min(args.samples, len(rows)))
    for img_id, preds_str in sample:
        preds = list(parse_predictions(preds_str))
        shown = " ".join(preds[:5])
        suffix = f"  …(+{len(preds)-5} more)" if len(preds) > 5 else ""
        print(f"    {img_id:35s} → [{shown}]{suffix}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 64}")
    if issues:
        print(f"  {_RED}{len(issues)} ISSUE(S) — DO NOT SUBMIT YET{_RST}")
        for i, msg in enumerate(issues, 1):
            print(f"    {i}. {msg}")
        print(f"{'═' * 64}\n")
        return 1
    else:
        print(f"  {_GREEN}CLEAN — submission looks ready{_RST}")
        print(f"{'═' * 64}\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
