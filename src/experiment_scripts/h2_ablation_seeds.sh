#!/bin/bash

export PYTHONPATH="."

# 5 seeds for the ablation study
SEEDS=(42 123 2026 777 8765)

# configurations to run
# Format: representation|factorized_policy|suffix
CONFIGS=(
    "simple|false|simple"
    "simple|true|facted"
    "spatial|false|spatial_simple"
    "spatial|true|spatial_fact"
)

for SEED in "${SEEDS[@]}"; do
    echo "========================================================================"
    echo "Starting ablation study runs for Seed: $SEED"
    echo "========================================================================"
    
    # Establish run directory based on the seed
    SAVE_DIR="experiments/h2_ablation_seed_${SEED}"
    mkdir -p "$SAVE_DIR"

    # Loop over each architecture configuration
    for CONFIG in "${CONFIGS[@]}"; do
        # separate with | and unpack config to the variables
        IFS="|" read -r REPR FACTORIZED SUFFIX <<< "$CONFIG"
        
        # Build additional command arguments
        OPT_ARGS=""
        if [ "$FACTORIZED" = "true" ]; then
            OPT_ARGS="--factorized_policy"
        fi
        
        RUN_NAME="h2_abl_seed_${SEED}"
        LOG_FILE="${SAVE_DIR}/trnsf_d4_dk64_depth3_${SUFFIX}.log"
        
        echo "------------------------------------------------------------------------"
        echo "Running: Repr=$REPR, Factored=$FACTORIZED, Seed=$SEED, Suffix=$SUFFIX"
        echo "Saving to: $SAVE_DIR/ (Run name: $RUN_NAME)"
        echo "Logging to: $LOG_FILE"
        echo "------------------------------------------------------------------------"
        
        python3 src/training/train_transformer.py data/gardner_depth4/d4_val.txt 64 \
            --seed "$SEED" \
            --num_blocks 3 \
            --epochs 30 \
            --batch_size 512 \
            --attn_backend math \
            --autocast none \
            --precision high \
            --lr 0.0030 \
            --representation "$REPR" \
            $OPT_ARGS \
            --run_name "$RUN_NAME" \
            --save_dir "$SAVE_DIR" \
            2>&1 \
            | tee "$LOG_FILE"
            
        echo "Finished run for Repr=$REPR, Factored=$FACTORIZED, Seed=$SEED"
        echo ""
    done
done

echo "========================================================================"
echo "All ablation seeds completed!"
echo "========================================================================"
