#!/bin/bash
# ===========================================================================
# ORACLE: High-Speed Feature Extraction (Phase 1 Entry Point)
# ===========================================================================
# This script orchestrates the zero-shot feature extraction phase.
# Run this as the first step in the pipeline to build the foundation cache.
# ===========================================================================
set -eo pipefail
trap 'echo "[feature_extract.sh] FAILED at line $LINENO (exit $?)" >&2; exit 1' ERR

ROLE=${1:-"sprint"}
shift 1
# Any remaining args (e.g. --force) are forwarded to the python script via launch_oracle.sh ${@:3}
EXTRA_ARGS=("$@")

# ORACLE: Environment Satiation
source /workspace/pytorch_env/bin/activate
export PYTHONPATH=$PYTHONPATH:/workspace/PlantCLEF2026:/workspace/PlantCLEF2026/src
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "==========================================================="
echo "🔬 ORACLE: Initializing Phase 1 Feature Extraction ($ROLE)"
echo "==========================================================="

# Detect local GPU count
DETECTED_GPUS=$(nvidia-smi -L | wc -l)
NUM_GPUS=${ORACLE_GPUS:-$DETECTED_GPUS}

# Call the launch orchestrator for Phase 1
if ! ./src/setup/launch_oracle.sh p1 $ROLE "${EXTRA_ARGS[@]}"; then
    echo "❌ ERROR: Feature extraction failed."
    exit 1
fi

echo "==========================================================="
echo "✅ PHASE 1 EXTRACTION COMPLETED SUCCESSFULLY."
echo "📍 Cache Location: models/cuda_deep_sat/phase1_feature_cache.pt"
echo "==========================================================="
