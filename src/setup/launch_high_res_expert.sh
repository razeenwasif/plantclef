#!/bin/bash
# ===========================================================================
# ORACLE: High-Resolution Specialist Training (RTX PRO 6000 Blackwell × 3)
# ===========================================================================
# Trains BioCLIP and DINOv3 experts sequentially, then runs SWA on each.
# Produces swa_model_final.pth in models/cuda_deep_sat/<expert>/ for the
# teacher cache builder.
#
# Usage:
#   ./scripts/launch_high_res_expert.sh          # sprint (all 3 local GPUs)
#   ./scripts/launch_high_res_expert.sh bioclip  # BioCLIP expert only
#   ./scripts/launch_high_res_expert.sh dinov3   # DINOv3 expert only
# ===========================================================================

set -e

EXPERT=${1:-"all"}   # all | bioclip | dinov3
FORCE_FLAG=""
EXTRA_TRAIN_ARGS=""

# Parse remaining args; forward --force to the SWA step
shift 1
while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE_FLAG="--force" ; shift ;;
        --res) EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --res $2" ; shift 2 ;;
        --epochs) EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --epochs $2" ; shift 2 ;;
        --seed) EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --seed $2" ; shift 2 ;;
        *) EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS $1" ; shift ;;
    esac
done

source /workspace/pytorch_env/bin/activate

export PYTHONPATH=$PYTHONPATH:/workspace/PlantCLEF2026:/workspace/PlantCLEF2026/src
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export OMP_NUM_THREADS=8

NUM_GPUS=$(nvidia-smi -L | wc -l)
TORCHRUN="torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --standalone"

_train_expert() {
    local CONFIG=$1
    local LABEL=$2
    shift 2
    echo "==========================================================="
    echo " Training: $LABEL"
    echo "==========================================================="
    $TORCHRUN -m phases.p2b_teacher.run --config "$CONFIG" $EXTRA_TRAIN_ARGS "$@"
    echo " $LABEL training done."
}

_swa_expert() {
    local CONFIG=$1
    local LABEL=$2
    echo "--- SWA: $LABEL ---"
    $TORCHRUN -m phases.p2b_teacher.swa --config "$CONFIG" $FORCE_FLAG
    echo " $LABEL SWA done."
}

echo "==========================================================="
echo "ORACLE: High-Resolution Specialist Training  (GPUs: $NUM_GPUS)"
echo "==========================================================="

if [[ "$EXPERT" == "all" || "$EXPERT" == "bioclip" ]]; then
    _train_expert configs/p2b_teacher_bioclip.yaml "BioCLIP Expert"
    _swa_expert   configs/p2b_teacher_bioclip.yaml "BioCLIP Expert"
fi

if [[ "$EXPERT" == "all" || "$EXPERT" == "dinov3" ]]; then
    _train_expert configs/p2b_teacher_dinov3.yaml  "DINOv3 Expert"
    _swa_expert   configs/p2b_teacher_dinov3.yaml  "DINOv3 Expert"
fi

echo "==========================================================="
echo "ALL HIGH-RES SPECIALISTS COMPLETED."
echo "Next: ./oracle.py cache --role sprint --batch 512"
echo "==========================================================="
