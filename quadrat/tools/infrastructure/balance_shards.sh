#!/bin/bash
# ===========================================================================
# plantclef: Cluster Shard Balancer (Distributed RAM-Disk Edition)
# ===========================================================================
# This script ensures that every pod has exactly the same number of shards,
# preventing distributed hangs caused by unbalanced epoch lengths.
# ===========================================================================

# 1. Configuration
SOURCE_DIR="/workspace/plantclef/shards"
TARGET_DIR="/dev/shm/shards"
TOTAL_PODS=4
MY_RANK=${CLUSTER_NODE_RANK:-$1}

if [[ -z "$MY_RANK" ]]; then
    echo "Usage: ./balance_shards.sh <node_rank>"
    echo "Example: ./balance_shards.sh 0"
    exit 1
fi

echo "[*] Initializing Shard Balancer for Pod $MY_RANK..."

# 2. Cleanup existing (potentially unbalanced/duplicate) shards
mkdir -p "$TARGET_DIR"
rm -rf "$TARGET_DIR"/train_*.tar

# 3. Calculate Slices
ALL_SHARDS=($(ls "$SOURCE_DIR"/train_*.tar | sort))
TOTAL_SHARDS=${#ALL_SHARDS[@]}
SHARDS_PER_POD=$((TOTAL_SHARDS / TOTAL_PODS))
REMAINDER=$((TOTAL_SHARDS % TOTAL_PODS))

# Start/End indices
START_IDX=$((MY_RANK * SHARDS_PER_POD))
END_IDX=$((START_IDX + SHARDS_PER_POD - 1))

# Last pod takes the remainder
if [[ "$MY_RANK" == $((TOTAL_PODS - 1)) ]]; then
    END_IDX=$((TOTAL_SHARDS - 1))
fi

MY_COUNT=$((END_IDX - START_IDX + 1))
echo "[*] Pod $MY_RANK assigned shards $START_IDX to $END_IDX (Total: $MY_COUNT)"

# 4. Copy Slices to RAM
for i in $(seq $START_IDX $END_IDX); do
    SHARD_PATH=${ALL_SHARDS[$i]}
    SHARD_NAME=$(basename "$SHARD_PATH")
    echo "  -> Loading $SHARD_NAME to RAM..."
    cp "$SHARD_PATH" "$TARGET_DIR/"
done

echo "[+] SUCCESS: Pod $MY_RANK is now balanced with $MY_COUNT shards."
echo "[*] Memory Usage: $(du -sh $TARGET_DIR)"
