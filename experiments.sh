#!/bin/bash

# Experiment configuration:
# - Embed dim (d_k): 64 and 128
# - Layers (depth): 3 and 6
# - Dataset: data/gardner_depth4/gen_gardner_d4.txt only
# - Epochs: 20
# - Batch size: 512
# - Attention: Math backend (no flash attention)
# - Autocast: Auto mode
# - Representation: Spatial
# - Heads: Factorized auxiliary policy heads
# - Learning rates based on Optuna tuning results:
#   * For d_k=64:
#     - depth=3: lr=0.0030 (based on dk64_depth4 tuning best_lr: ~0.0030)
#     - depth=6: lr=0.0033 (interpolated between depth=4: 0.0030 and depth=8: 0.0035)
#   * For d_k=128 (interpolated between d_k=64: ~0.0030 and d_k=256: ~0.0005):
#     - depth=3: lr=0.0015
#     - depth=6: lr=0.0015

# echo "Starting Experiment 1: d_k=64, depth=3"
# python3 src/training/train_transformer.py data/gardner_depth4/gen_gardner_d4.txt 64 \
#     --num_blocks 3 \
#     --epochs 20 \
#     --batch_size 512 \
#     --attn_backend math \
#     --autocast auto \
#     --lr 0.0030 \
#     --representation spatial \
#     --factorized_policy \
#     --run_name trnsf_d4_dk64_depth3 \
#     | tee logs/exps/trnsf_d4_dk64_depth3.log

# echo "Starting Experiment 2: d_k=64, depth=6"
# python3 src/training/train_transformer.py data/gardner_depth4/gen_gardner_d4.txt 64 \
#     --num_blocks 6 \
#     --epochs 20 \
#     --batch_size 512 \
#     --attn_backend math \
#     --autocast auto \
#     --lr 0.0033 \
#     --representation spatial \
#     --factorized_policy \
#     --run_name trnsf_d4_dk64_depth6 \
#     | tee logs/exps/trnsf_d4_dk64_depth6.log

# echo "Starting Experiment 3: d_k=128, depth=3"
# python3 src/training/train_transformer.py data/gardner_depth4/gen_gardner_d4.txt 128 \
#     --num_blocks 3 \
#     --epochs 20 \
#     --batch_size 512 \
#     --attn_backend math \
#     --autocast auto \
#     --lr 0.0015 \
#     --representation spatial \
#     --factorized_policy \
#     --run_name trnsf_d4_dk128_depth3 \
#     | tee logs/exps/trnsf_d4_dk128_depth3.log

# echo "Starting Experiment 4: d_k=128, depth=6"
# python3 src/training/train_transformer.py data/gardner_depth4/gen_gardner_d4.txt 128 \
#     --num_blocks 6 \
#     --epochs 20 \
#     --batch_size 512 \
#     --attn_backend math \
#     --autocast auto \
#     --lr 0.0015 \
#     --representation spatial \
#     --factorized_policy \
#     --run_name trnsf_d4_dk128_depth6 \
#     | tee logs/exps/trnsf_d4_dk128_depth6.log

echo "Starting Experiment 3: d_k=128, depth=3"
python3 src/training/train_transformer.py data/gardner_depth4/gen_gardner_d4.txt 256 \
    --num_blocks 3 \
    --epochs 20 \
    --batch_size 512 \
    --attn_backend math \
    --autocast auto \
    --lr 0.0005 \
    --representation spatial \
    --factorized_policy \
    --run_name trnsf_d4_dk256_depth3 \
    | tee logs/exps/trnsf_d4_dk256_depth3.log

echo "Starting Experiment 4: d_k=128, depth=6"
python3 src/training/train_transformer.py data/gardner_depth4/gen_gardner_d4.txt 256 \
    --num_blocks 6 \
    --epochs 20 \
    --batch_size 512 \
    --attn_backend math \
    --autocast auto \
    --lr 0.0005 \
    --representation spatial \
    --factorized_policy \
    --run_name trnsf_d4_dk256_depth6 \
    | tee logs/exps/trnsf_d4_dk256_depth6.log

