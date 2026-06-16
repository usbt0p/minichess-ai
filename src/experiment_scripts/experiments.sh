#!/bin/bash


# - tomarse en serio split train-val-test, hacer uno de test
# - fix some arquitecture recipes (big and small) and commit to them for now
# - fix hiperparams for them (lr, batch size, dtype and sdpa kernel)
# - just go ahead and train them fully, like 50 epochs for example (al entrenar: probar con todo mezclado VS. con primero depth 2 e incrementar)

# - train reproducibly and cleanly:
#     - first DECIDE ON THE DATA SPLIT USED! meaning: d2, d3 or d4 or merged? use this throughout the whole pipeline
#     - then SPLIT IN TRAIN/TEST!!!! hold test out and use for later proofs.
#     - small mlp simple encoding, big mlp simple encoding, small standard encoding, big standard encoding, small 2d + factored head encoding, big 2d + factored head encoding (consult previous results in mlp since its been long ago). pick adequate lr, stick with normal batch size and dtype
#     - compare, draw conclusions. prove statistical significance

# my decisions:
# - depth 4 dataset only, val and train done with 3% split, (58234050+1795950) = 60030000
# - no flash attn, no autocast, matmul precision at highest.  
# - use dim 64 and 128 for faster training
# - assign respectively 0.003 and 0.0015 for LR. batch stays at 512
# - 50 epochs
# for the specific exps:
# - random baseline:
# - mlp small:
# - mlp big:
# - trnsf small:
# - trnsf big:
# - trnsf + spatial input + factored head small:
# - trnsf + spatial input + factored head big:

# Global configuration:
# - Dataset: data/gardner_depth4/d4_val.txt only (97% train/val split of gen_gardner_d4.txt)
# - Epochs: 50
# - Batch size: 512
# - Attention: Math backend (no flash attention)
# - Autocast: None (no float16/bfloat16)
# - Matmul precision: Highest (automatically set to highest in training script)
# - Save directory: experiments/exp1_mlp_transf
# - Train/Val split: 97% training (3% validation)

# note: for some reason results are worse than some previous runs, and learning
# curves are flatter. this might be dues to 
# lack of reproducibility, different dtype (precision highest), the data splits...

# Create save directory
mkdir -p experiments/exp1_mlp_transf

# ----------------- MLP Baselines -----------------

# echo "Starting MLP Small baseline (hidden_size=512)"
# python3 src/models/fnnPromotionMasking.py data/gardner_depth4/d4_val.txt \
#     --hidden_size 512 \
#     --lr 0.0020 \
#     --batch_size 512 \
#     --epochs 50 \
#     --run_name mlp_small \
#     --save_dir experiments/exp1_mlp_transf \
#     | tee experiments/exp1_mlp_transf/mlp_small.log

# echo "Starting MLP Big baseline (hidden_size=1024)"
# python3 src/models/fnnPromotionMasking.py data/gardner_depth4/d4_val.txt \
#     --hidden_size 1024 \
#     --lr 0.0020 \
#     --batch_size 512 \
#     --epochs 50 \
#     --run_name mlp_big \
#     --save_dir experiments/exp1_mlp_transf \
#     | tee experiments/exp1_mlp_transf/mlp_big.log


# ----------------- (Flat representation, simple head) -----------------

python3 src/training/train_transformer.py data/gardner_depth4/d4_val.txt 64 \
    --num_blocks 3 \
    --epochs 30 \
    --batch_size 512 \
    --attn_backend math \
    --autocast none \
    --lr 0.0030 \

    --representation simple \

    --run_name trnsf_d4_dk64_depth3_simple \
    --save_dir experiments/exp1_mlp_transf \
    | tee experiments/exp1_mlp_transf/trnsf_d4_dk64_depth3_simple.log


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

    --run_name trnsf_d4_dk64_depth3_facted \
    --save_dir experiments/exp1_mlp_transf \
    | tee experiments/exp1_mlp_transf/trnsf_d4_dk64_depth3_facted.log

# ----------------- Spatial Transformer + simple Head -----------------

python3 src/training/train_transformer.py data/gardner_depth4/d4_val.txt 64 \
    --num_blocks 3 \
    --epochs 30 \
    --batch_size 512 \
    --attn_backend math \
    --autocast none \
    --lr 0.0030 \

    --representation spatial \

    --run_name trnsf_d4_dk64_depth3_spatial_simple \
    --save_dir experiments/exp1_mlp_transf \
    | tee experiments/exp1_mlp_transf/trnsf_d4_dk64_depth3_spatial_simple.log

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

    --run_name trnsf_d4_dk64_depth3_spatial_fact \
    --save_dir experiments/exp1_mlp_transf \
    | tee experiments/exp1_mlp_transf/trnsf_d4_dk64_depth3_spatial_fact.log


###########################################

## BIg versions

# echo "Starting Standard Transformer Big (d_k=128, depth=6)"
# python3 src/training/train_transformer.py data/gardner_depth4/d4_val.txt 128 \
#     --num_blocks 6 \
#     --epochs 50 \
#     --batch_size 512 \
#     --attn_backend math \
#     --autocast none \
#     --lr 0.0015 \
#     --representation simple \
#     --run_name trnsf_d4_dk128_depth6_simple \
#     --save_dir experiments/exp1_mlp_transf \
#     | tee experiments/exp1_mlp_transf/trnsf_d4_dk128_depth6_simple.log


# echo "Starting Spatial Transformer + Factored Head Big (d_k=128, depth=6)"
# python3 src/training/train_transformer.py data/gardner_depth4/d4_val.txt 128 \
#     --num_blocks 6 \
#     --epochs 50 \
#     --batch_size 512 \
#     --attn_backend math \
#     --autocast none \
#     --lr 0.0015 \
#     --representation spatial \
#     --factorized_policy \
#     --run_name trnsf_d4_dk128_depth6_spatial_fact \
#     --save_dir experiments/exp1_mlp_transf \
#     | tee experiments/exp1_mlp_transf/trnsf_d4_dk128_depth6_spatial_fact.log
