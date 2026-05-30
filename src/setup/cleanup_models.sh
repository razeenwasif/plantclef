#!/bin/bash
# ===========================================================================
# plantclef: Safe Models Directory Cleanup
# ===========================================================================
# Removes expendable intermediate artifacts while hard-protecting:
#   - models/cuda_deep_sat/phase1_feature_cache.pt
#   - models/warmup_hardened.pth
#   - models/cuda_deep_sat/*/swa_model_final.pth
#   - models/cuda_deep_sat/plantclef_s42/swa_model_final.pth
#   - models/zero_shot_anchors*.pt
#   - models/optimized_thresholds.json
#
# What gets removed (with --confirm):
#   - p1_shards/        (per-rank merge shards, already merged into cache)
#   - DeepSpeed ep* optimizer dirs  (large, only needed mid-training)
#   - Old phase2_checkpoint_* dirs   (pre-modularization DS checkpoints)
#   - epoch_ep*_final.pth             (SWA inputs, safe to remove after SWA)
#   - phase2_checkpoint_ep*_step_final.pth  (old-format epoch snapshots)
#
# Usage:
#   ./scripts/cleanup_models.sh          # dry-run (shows what would be deleted)
#   ./scripts/cleanup_models.sh --confirm  # actually delete
# ===========================================================================

set -e

MODELS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
DRY_RUN=true
if [[ "${1}" == "--confirm" ]]; then
    DRY_RUN=false
fi

_log()  { echo "  [keep]   $1"; }
_warn() { echo "  [DELETE] $1"; }

_remove() {
    local target="$1"
    if [[ -e "$target" ]]; then
        _warn "$target"
        if [[ "$DRY_RUN" == "false" ]]; then
            rm -rf "$target"
        fi
    fi
}

echo "==========================================================="
echo "PLANTCLEF Model Cleanup  (dry_run=$DRY_RUN)"
echo "Root: $MODELS_DIR"
echo "==========================================================="

# ── Hard-protected paths ────────────────────────────────────────────────────
echo ""
echo "PROTECTED (will never be touched):"
_log "$MODELS_DIR/cuda_deep_sat/phase1_feature_cache.pt"
_log "$MODELS_DIR/warmup_hardened.pth"
_log "$MODELS_DIR/zero_shot_anchors.pt"
_log "$MODELS_DIR/zero_shot_anchors_v2.pt"
_log "$MODELS_DIR/optimized_thresholds.json"
_log "$MODELS_DIR/cuda_deep_sat/teacher_logit_cache.npy"
for f in "$MODELS_DIR"/cuda_deep_sat/*/swa_model_final.pth \
          "$MODELS_DIR"/cuda_deep_sat/plantclef_s*/swa_model_final.pth; do
    [[ -f "$f" ]] && _log "$f"
done

# ── Phase 1 per-rank shards (already merged into cache) ─────────────────────
echo ""
echo "REMOVABLE — phase 1 merge shards:"
_remove "$MODELS_DIR/cuda_deep_sat/p1_shards"

# ── Old-format epoch snapshots (phase2_checkpoint_ep*_step_final.pth) ──────
echo ""
echo "REMOVABLE — old-format epoch snapshots:"
find "$MODELS_DIR" -name "phase2_checkpoint_ep*_step_final.pth" | sort | while read -r f; do
    _remove "$f"
done

# ── New-format epoch snapshots (epoch_ep*_final.pth) ───────────────────────
# Only safe to remove if swa_model_final.pth exists in the same directory.
echo ""
echo "REMOVABLE — new-format epoch snapshots (where SWA already exists):"
find "$MODELS_DIR" -name "epoch_ep*_final.pth" | sort | while read -r f; do
    dir=$(dirname "$f")
    swa="$dir/swa_model_final.pth"
    # SWA must exist AND be substantial (>10 MB) — guards against half-written files.
    if [[ -f "$swa" ]] && [[ $(stat -c %s "$swa" 2>/dev/null || echo 0) -gt 10485760 ]]; then
        _remove "$f"
    else
        echo "  [keep]   $f  (swa_model_final.pth missing or too small in $dir)"
    fi
done

# ── DeepSpeed ep* optimizer directories ─────────────────────────────────────
# These are the ep0/, ep1/, ... subdirs written by model_engine.save_checkpoint().
echo ""
echo "REMOVABLE — DeepSpeed per-epoch optimizer shards:"
find "$MODELS_DIR" -maxdepth 4 -type d -name "ep[0-9]*" | sort | while read -r d; do
    _remove "$d"
done

# ── Old pre-modularization DeepSpeed checkpoint dirs ────────────────────────
# phase2_checkpoint, phase2_checkpoint_A, phase2_checkpoint_B, expert_launcher_A/B
echo ""
echo "REMOVABLE — pre-modularization DeepSpeed checkpoint dirs (no epoch_ep*.pth siblings):"
for pattern in "phase2_checkpoint" "phase2_checkpoint_A" "phase2_checkpoint_B" \
               "expert_launcher_A" "expert_launcher_B"; do
    find "$MODELS_DIR" -maxdepth 4 -type d -name "$pattern" | sort | while read -r d; do
        _remove "$d"
    done
done

# ── latest / latest.bak symlinks / dirs ─────────────────────────────────────
echo ""
echo "REMOVABLE — stale 'latest' symlinks and dirs:"
find "$MODELS_DIR" -maxdepth 4 \( -name "latest" -o -name "latest.bak" \) | sort | while read -r f; do
    _remove "$f"
done

echo ""
echo "==========================================================="
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN complete.  Re-run with --confirm to actually delete."
else
    echo "Cleanup complete."
fi
echo "==========================================================="
