#!/bin/bash

# Global configuration:
# - Dataset: data/gardner_depth4/d4_val.txt only (97% train/val split of gen_gardner_d4.txt)
# - Epochs: 30 epochs. anything else seems overkill
# - Batch size: 512
# - Attention: Math backend (no flash attention)
# - Autocast: None (no float16/bfloat16)
# - Matmul precision: Highest (automatically set to highest in training script)
# - Save directory: experiments/h2_ablation_1
# - Train/Val split: 97% training (3% validation) (58234050+1795950) = 60030000
# - use dim 64 and 128 for faster training
# - assign respectively 0.003 and 0.0015 for LR. batch stays at 512

mkdir -p experiments/h2_ablation_1

# ----------------- (Flat representation, simple head) -----------------

python3 src/training/train_transformer.py data/gardner_depth4/d4_val.txt 64 \
    --num_blocks 3 \
    --epochs 30 \
    --batch_size 512 \
    --attn_backend math \
    --autocast none \
    --lr 0.0030 \
    --representation simple \
    --run_name h2_abl \
    --save_dir experiments/h2_ablation_1 \
    2>&1 \
    | tee experiments/h2_ablation_1/trnsf_d4_dk64_depth3_simple.log


# ----------------- (Flat representation, factored head) -----------------

python3 src/training/train_transformer.py data/gardner_depth4/d4_val.txt 64 \
    --num_blocks 3 \
    --epochs 30 \
    --batch_size 512 \
    --attn_backend math \
    --autocast none \
    --lr 0.0030 \
    --representation simple \
    --factorized_policy \
    --run_name h2_abl \
    --save_dir experiments/h2_ablation_1 \
    2>&1 \
    | tee experiments/h2_ablation_1/trnsf_d4_dk64_depth3_facted.log

# ----------------- Spatial Transformer + simple Head -----------------

python3 src/training/train_transformer.py data/gardner_depth4/d4_val.txt 64 \
    --num_blocks 3 \
    --epochs 30 \
    --batch_size 512 \
    --attn_backend math \
    --autocast none \
    --lr 0.0030 \
    --representation spatial \
    --run_name h2_abl \
    --save_dir experiments/h2_ablation_1 \
    2>&1 \
    | tee experiments/h2_ablation_1/trnsf_d4_dk64_depth3_spatial_simple.log

# ----------------- Spatial Transformer + Factorized Head -----------------

python3 src/training/train_transformer.py data/gardner_depth4/d4_val.txt 64 \
    --num_blocks 3 \
    --epochs 30 \
    --batch_size 512 \
    --attn_backend math \
    --autocast none \
    --lr 0.0030 \
    --representation spatial \
    --factorized_policy \
    --run_name h2_abl \
    --save_dir experiments/h2_ablation_1 \
    2>&1 \
    | tee experiments/h2_ablation_1/trnsf_d4_dk64_depth3_spatial_fact.log
