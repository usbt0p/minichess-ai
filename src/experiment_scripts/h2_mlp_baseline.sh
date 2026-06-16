#!/bin/bash

mkdir -p experiments/h2_ablation_1

# ----------------- MLP Baselines -----------------

echo "Starting MLP Small baseline (hidden_size=512)"
python3 src/models/fnnPromotionMasking.py data/gardner_depth4/d4_val.txt \
    --hidden_size 512 \
    --lr 0.0020 \
    --batch_size 512 \
    --epochs 50 \
    --run_name mlp_small \
    --save_dir experiments/exp1_mlp_transf \
    | tee experiments/exp1_mlp_transf/mlp_small.log

echo "Starting MLP Big baseline (hidden_size=1024)"
python3 src/models/fnnPromotionMasking.py data/gardner_depth4/d4_val.txt \
    --hidden_size 1024 \
    --lr 0.0020 \
    --batch_size 512 \
    --epochs 50 \
    --run_name mlp_big \
    --save_dir experiments/exp1_mlp_transf \
    | tee experiments/exp1_mlp_transf/mlp_big.log