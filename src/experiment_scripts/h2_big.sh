# Create save directory
mkdir -p experiments/h2_ablation_1

python3 src/training/train_transformer.py data/gardner_depth4/d4_val.txt 128 \
    --num_blocks 3 \
    --epochs 30 \
    --batch_size 512 \
    --attn_backend math \
    --autocast none \
    --lr 0.0015 \
    --representation simple \
    --run_name h2_abl_big \
    --save_dir experiments/h2_ablation_1 \
    2>&1 \
    | tee experiments/h2_ablation_1/trnsf_d4_dk128_depth3_simple.log


# ----------------- (Flat representation, factored head) -----------------

python3 src/training/train_transformer.py data/gardner_depth4/d4_val.txt 128 \
    --num_blocks 3 \
    --epochs 30 \
    --batch_size 512 \
    --attn_backend math \
    --autocast none \
    --lr 0.0015 \
    --representation simple \
    --factorized_policy \
    --run_name h2_abl_big \
    --save_dir experiments/h2_ablation_1 \
    2>&1 \
    | tee experiments/h2_ablation_1/trnsf_d4_dk128_depth3_facted.log

# ----------------- Spatial Transformer + simple Head -----------------

python3 src/training/train_transformer.py data/gardner_depth4/d4_val.txt 128 \
    --num_blocks 3 \
    --epochs 30 \
    --batch_size 512 \
    --attn_backend math \
    --autocast none \
    --lr 0.0015 \
    --representation spatial \
    --run_name h2_abl_big \
    --save_dir experiments/h2_ablation_1 \
    2>&1 \
    | tee experiments/h2_ablation_1/trnsf_d4_dk128_depth3_spatial_simple.log

# ----------------- Spatial Transformer + Factorized Head -----------------

python3 src/training/train_transformer.py data/gardner_depth4/d4_val.txt 128 \
    --num_blocks 3 \
    --epochs 30 \
    --batch_size 512 \
    --attn_backend math \
    --autocast none \
    --lr 0.0015 \
    --representation spatial \
    --factorized_policy \
    --run_name h2_abl_big \
    --save_dir experiments/h2_ablation_1 \
    2>&1 \
    | tee experiments/h2_ablation_1/trnsf_d4_dk128_depth3_spatial_fact.log
