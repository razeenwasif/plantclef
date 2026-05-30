#!/bin/bash
# ===========================================================================
# plantclef: Multi-Seed Sprint Orchestrator (192px / Extreme Speed)
# ===========================================================================
# This script runs multiple training sessions with different seeds
# and saves them into unique directories for ensembling.
#
# RUN THIS ON EVERY POD SIMULTANEOUSLY.
# ===========================================================================

# Use the first argument as the phase, default to "p2b" if not provided
PHASE=${1:-"p2b"}
SEEDS=(42 1337 2026)

echo "==========================================================="
echo "🛡️  PLANTCLEF: Initializing Multi-Seed Sprint for Phase: $PHASE"
echo "==========================================================="

for SEED in "${SEEDS[@]}"; do
    RUN_NAME="plantclef_s${SEED}"
    
    echo "-----------------------------------------------------------"
    echo "🚀 STARTING SPRINT: Seed $SEED | Name $RUN_NAME | Phase $PHASE"
    echo "-----------------------------------------------------------"
    
    export PLANTCLEF_SEED=$SEED
    export PLANTCLEF_NAME=$RUN_NAME
    
    # Launch the Go Orchestrator (which handles cluster sync)
    ./coord $PHASE
    
    echo "✅ FINISHED SPRINT: Seed $SEED"
    echo "Waiting 30s for cluster cleanup..."
    sleep 30
done

echo "==========================================================="
echo "🏁 ALL SPRINTS COMPLETED FOR PHASE: $PHASE"
echo "==========================================================="
