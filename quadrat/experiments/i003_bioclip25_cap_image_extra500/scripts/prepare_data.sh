#!/usr/bin/env bash
# Build the combined+capped training manifest for i003.
# Reads the old i002 manifest and the new extra_under100 manifest,
# deduplicates by image_path, caps at 500 images per species (seed=42),
# and writes:
#   data/combined_old_extra_max500_train_manifest.csv
#   data/combined_old_extra_max500_summary.json
#   data/species_counts_before_after.csv
#
# Usage:
#   bash scripts/prepare_data.sh                          # use defaults
#   bash scripts/prepare_data.sh --max-per-species 1000   # override cap
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "  i003 — prepare combined manifest"
echo "=========================================="

python prepare_combined_manifest.py "$@"

echo ""
echo "Combined manifest preparation complete."
echo "Outputs in ./data/"
