#!/bin/bash
# ===========================================================================
# PlantCLEF 2026: Ultra-ORACLE Launch Orchestrator (Unified Source of Truth)
# ===========================================================================
# Usage: 
#   ./launch_oracle.sh <p1|p2a|p2b|swa|opt|cache|inf|pipeline> <master|worker|sprint> [args...]
# ===========================================================================

set -eo pipefail

PHASE=$1
ROLE=$2
MASTER_IP=${ORACLE_MASTER_IP:-"127.0.0.1"}
PORT=${ORACLE_MASTER_PORT:-"29505"}

# ── 1. ENVIRONMENT & HARDWARE DETECT ───────────────────────────────────────

# Auto-detect best network interface
IFACE=$(ip -4 addr show | grep 'inet 10\.' | head -n1 | awk '{print $NF}')
if [[ -z "$IFACE" ]]; then IFACE=$(ip -4 route show default | awk '{print $5}'); fi
if ip addr show podnet1 &>/dev/null; then IFACE="podnet1"; fi

LOCAL_IP=$(ip -4 addr show $IFACE | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n1)
if [[ -z "$LOCAL_IP" ]]; then LOCAL_IP=$(hostname -I | awk '{print $1}'); fi

# ── ACCELERATOR MODE RESOLUTION ──────────────────────────────────────────
# ORACLE_MODE is forwarded by oracle.py (cuda | tpu | auto). Default: auto.
ORACLE_MODE=${ORACLE_MODE:-auto}
if [[ "$ORACLE_MODE" == "auto" ]]; then
    if [[ -n "${TPU_NAME:-}" || "${PJRT_DEVICE:-}" == "TPU" ]]; then
        ORACLE_MODE="tpu"
    elif command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
        ORACLE_MODE="cuda"
    else
        echo "[ORACLE] No CUDA or TPU detected; aborting." >&2
        exit 1
    fi
fi
echo "[ORACLE] Accelerator: $ORACLE_MODE"

# Device count — branch-detected and overrideable by env.
if [[ "$ORACLE_MODE" == "tpu" ]]; then
    # XLA chip count: TPU v3-8 → 8 cores, v4 → 4 chips × 2 cores, etc.
    # `python -c "import torch_xla.core.xla_model as xm; print(xm.xrt_world_size())"`
    # only works inside a spawned process, so we trust ORACLE_TPU_CORES (or 8 as a sane default for v3-8 / v4-8).
    NUM_DEVICES=${ORACLE_TPU_CORES:-8}
    NUM_GPUS=0
else
    DETECTED_GPUS=$(nvidia-smi -L | wc -l)
    NUM_GPUS=${ORACLE_GPUS:-$DETECTED_GPUS}
    NUM_DEVICES=$NUM_GPUS
fi

# ── ACCELERATOR-SPECIFIC ENVIRONMENT ────────────────────────────────────
if [[ "$ORACLE_MODE" == "cuda" ]]; then
    # NCCL / Blackwell Optimizations
    export NCCL_SOCKET_IFNAME=$IFACE
    export GLOO_SOCKET_IFNAME=$IFACE
    export TP_SOCKET_IFNAME=$IFACE
    export NCCL_SOCKET_FAMILY=AF_INET
    export GLOO_SOCKET_FAMILY=AF_INET
    export TP_SOCKET_FAMILY=AF_INET
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_NVLS_ENABLE=0
    export NCCL_LAUNCH_MODE=PARALLEL
    export NCCL_DEBUG=WARN
    export NCCL_BUFFSIZE=16777216
    export NCCL_MAX_NCHANNELS=8

    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_CUDA_MATMUL_TF32=1
    export TORCH_CUDNN_V8_API_ENABLED=1
    export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
elif [[ "$ORACLE_MODE" == "tpu" ]]; then
    # PJRT runtime (preferred over XRT on torch_xla 2.0+)
    export PJRT_DEVICE=${PJRT_DEVICE:-TPU}
    # Mixed precision — emit bfloat16 matmuls automatically
    export XLA_USE_BF16=${XLA_USE_BF16:-1}
    # Quieter logs by default
    export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}
    # Optional: hide XLA's verbose graph dump unless explicitly requested
    : "${XLA_IR_DEBUG:=0}"
    : "${XLA_HLO_DEBUG:=0}"
    export XLA_IR_DEBUG XLA_HLO_DEBUG
fi

export HF_HUB_OFFLINE=1

# Pathing & Python Satiation
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export PYTHONPATH=$PYTHONPATH:/workspace/PlantCLEF2026:/workspace/PlantCLEF2026/src

export ORACLE_MODE

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TORCH_COMPILE_DEBUG=0

# ── 2. PHASE RESOLUTION ────────────────────────────────────────────────────

if [[ -z "$PHASE" ]]; then
    echo "Usage: ./launch_oracle.sh <p1|p2a|p2b|swa|opt|cache|inf|pipeline> <master|worker|sprint>"
    exit 1
fi

case $PHASE in
    p1)          SCRIPT="-m phases.foundation_caching.run"; YAML="configs/p1_extract.yaml" ;;
    p2a)         SCRIPT="-m phases.head_warmup.run";        YAML="configs/p2a_warmup.yaml" ;;
    p2b-expert)  SCRIPT="-m phases.expert_specialization.run"; YAML="configs/p2b_teacher_bioclip.yaml" ;;
    p2b-student) SCRIPT="-m phases.student_distillation.run";   YAML="configs/p2b_student.yaml" ;;
    ad-td)       SCRIPT="-m phases.asymmetric_distillation.run"; YAML="configs/p2b_student.yaml" ;;
    swa)         SCRIPT="-m phases.student_distillation.swa";   YAML="configs/p2b_student.yaml" ;;
    opt)         SCRIPT="tools/modeling/optimize_thresholds.py"; YAML="configs/training.yaml" ;;
    cache)    SCRIPT="tools/data/build_teacher_cache.py"; YAML="configs/training.yaml" ;;
    inf)      SCRIPT="-m phases.inference.run";    YAML="configs/inference.yaml" ;;
    pipeline) SCRIPT="INTERNAL"; YAML="configs/inference.yaml" ;;
    *) echo "Error: Invalid phase '${PHASE}'"; exit 1 ;;
esac

# ── 3. ULTIMATE PIPELINE LOGIC (Unified from ultimate_inf.sh) ──────────────

run_ultimate_pipeline() {
    local ROLE=$1
    local SEEDS=(42 1337 2026 777 9999)
    local SPECIALISTS=("expert_bioclip_512" "expert_dinov3_512")
    local BASE_DIR="/workspace/PlantCLEF2026/models/cuda_deep_sat"

    echo "🛡️  ORACLE: Executing Ultimate Inference Pipeline ($ROLE)"

    # STEP 1: SWA for all seeds
    for SEED in "${SEEDS[@]}"; do
        echo "🔄 Averaging Seed $SEED..."
        ORACLE_SEED=$SEED ORACLE_NAME="oracle_s${SEED}" ./launch_oracle.sh swa $ROLE
    done
    for SPEC in "${SPECIALISTS[@]}"; do
        echo "🔄 Averaging Specialist $SPEC..."
        ORACLE_NAME="$SPEC" ./launch_oracle.sh swa $ROLE
    done

    # STEP 2: Threshold Opt
    echo "📈 Optimizing Thresholds..."
    ./launch_oracle.sh opt $ROLE

    # STEP 3: Conformal Calibration
    if [[ "$ROLE" == "sprint" || "$ROLE" == "master" ]]; then
        echo "📊 Calibrating Conformal Predictor..."
        python3 src/tools/inference/calibrate_conformal.py || true
    fi

    # STEP 4: Mega-Ensemble
    echo "🚀 Launching Final Ensemble..."
    CHECKPOINTS=()
    RESOLUTIONS=()
    for SEED in "${SEEDS[@]}"; do
        CKPT="${BASE_DIR}/oracle_s${SEED}/swa_model_final.pth"
        if [ -f "$CKPT" ]; then 
            CHECKPOINTS+=("--checkpoint" "$CKPT")
            RESOLUTIONS+=("448")
        fi
    done
    for SPEC in "${SPECIALISTS[@]}"; do
        CKPT="${BASE_DIR}/${SPEC}/swa_model_final.pth"
        if [ -f "$CKPT" ]; then 
            CHECKPOINTS+=("--checkpoint" "$CKPT")
            RESOLUTIONS+=("672")
        fi
    done

    RES_ARGS=()
    for RES in "${RESOLUTIONS[@]}"; do RES_ARGS+=("--resolution" "$RES"); done

    ./launch_oracle.sh inf $ROLE --config configs/inference.yaml "${CHECKPOINTS[@]}" "${RES_ARGS[@]}"
}

if [[ "$PHASE" == "pipeline" ]]; then
    run_ultimate_pipeline "$ROLE"
    exit 0
fi

# ── 4. MULTI-SEED SPRINT LOGIC (Unified from multi_seed_train.sh) ──────────

if [[ "$ROLE" == "sprint" && ( "$PHASE" == "p2a" || "$PHASE" == "p2b-student" || "$PHASE" == "p2b-expert" || "$PHASE" == "ad-td" ) ]]; then
    # If --seed is provided in args, we only run that one. Otherwise use standard ensemble seeds.
    SEEDS=(42 1337 2026 777)
    for i in "${!@}"; do
        if [[ "${!i}" == "--seed" ]]; then
            next=$((i+1))
            SEEDS=(${!next})
            break
        fi
    done

    echo "[ORACLE] Multi-Seed Sprint Mode — Seeds: ${SEEDS[*]}"
    for SEED in "${SEEDS[@]}"; do
        export ORACLE_SEED=$SEED
        export ORACLE_NAME="oracle_s${SEED}"
        echo "[ORACLE] Launching seed $SEED..."
        # We call our own script with a temporary role to avoid recursion, 
        # or we just let it fall through if we handle it carefully.
        # To avoid recursion loop, we use a hidden role 'single-sprint'
        ./launch_oracle.sh "$PHASE" "single-sprint" "${@:3}"
        echo "[ORACLE] Seed $SEED done. Waiting 5s..."
        sleep 5
    done
    exit 0
fi

# ── 5. CLUSTER & RANK ORCHESTRATION ────────────────────────────────────────

# Normalize ROLE
[[ "$ROLE" == "single-sprint" ]] && ROLE="sprint"

PORT=${ORACLE_MASTER_PORT:-29505}
MASTER_IP_FILE="/workspace/PlantCLEF2026/src/setup/.oracle_master_ip"

if [[ "$ROLE" == "master" ]]; then
    echo "$LOCAL_IP" > "$MASTER_IP_FILE"
    MASTER_IP=$LOCAL_IP
    NODE_RANK=0
    echo "1" > "/workspace/PlantCLEF2026/src/setup/.oracle_rank_counter"
    echo "👑 [ORACLE] Master (Rank 0) at $LOCAL_IP"
    trap 'rm -f "$MASTER_IP_FILE"' EXIT
elif [[ "$ROLE" == "worker" ]]; then
    if [[ -z "$ORACLE_MASTER_IP" && -f "$MASTER_IP_FILE" ]]; then
        MASTER_IP=$(cat "$MASTER_IP_FILE")
    fi
    NODE_RANK=${ORACLE_NODE_RANK:-""}
    if [[ -z "$NODE_RANK" ]]; then
        # Atomic auto-assignment
        RANK_COUNTER="/workspace/PlantCLEF2026/src/setup/.oracle_rank_counter"
        LOCK_DIR="${RANK_COUNTER}.lockdir"
        while ! mkdir "$LOCK_DIR" 2>/dev/null; do sleep 0.5; done
        NODE_RANK=$(cat "$RANK_COUNTER" 2>/dev/null || echo "1")
        echo $((NODE_RANK + 1)) > "$RANK_COUNTER"
        rmdir "$LOCK_DIR"
    fi
    echo "📡 [ORACLE] Worker (Rank $NODE_RANK) connecting to $MASTER_IP"
else
    # Sprint / standalone
    NODE_RANK=0
    MASTER_IP="127.0.0.1"
fi

# ── 6. EXECUTION ───────────────────────────────────────────────────────────

NNODES=${ORACLE_NNODES:-$(( ${ORACLE_TOTAL_WORKERS:-0} + 1 ))}
[[ "$ROLE" == "sprint" ]] && NNODES=1

RDZV_ENDPOINT="$MASTER_IP:$PORT"
[[ "$ROLE" == "master" ]] && RDZV_ENDPOINT="127.0.0.1:$PORT"

echo "[ORACLE] Executing $PHASE ($ROLE) | Rank $NODE_RANK | Nodes $NNODES | Devices/node $NUM_DEVICES | Mode $ORACLE_MODE"
mkdir -p /workspace/PlantCLEF2026/reports/logs

if [[ "$ORACLE_MODE" == "tpu" ]]; then
    # TPU launch path:
    # PJRT discovers the topology automatically; we just run the entry script
    # once per host. xmp.spawn inside the entry script fans out across the
    # local TPU cores. Multi-host TPU pods rely on `gcloud compute tpus
    # tpu-vm ssh ... --worker=all -- ` to fan-out, which is the user's
    # responsibility — this script handles a single TPU VM.
    python3 $SCRIPT --config $YAML --seed ${ORACLE_SEED:-42} --name ${ORACLE_NAME:-"oracle_expert"} "${@:3}" 2>&1 \
        | tee /workspace/PlantCLEF2026/reports/logs/tpu_${NODE_RANK}.log
else
    # CUDA launch path (unchanged from the historical Blackwell pipeline).
    torchrun \
        --local_addr=$LOCAL_IP \
        --master_addr=$MASTER_IP \
        --master_port=$PORT \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=$NUM_DEVICES \
        --rdzv_backend=static \
        $SCRIPT --config $YAML --seed ${ORACLE_SEED:-42} --name ${ORACLE_NAME:-"oracle_expert"} "${@:3}" 2>&1 \
        | tee /workspace/PlantCLEF2026/reports/logs/rank_${NODE_RANK}.log
fi
