#!/bin/bash

export PYTHONPATH="."
source .venv/bin/activate

echo "================================================================="
python3 src/ppo/train_ppo.py \
    --iterations 50 \
    --checkpoint "results/ablations/ablations_dk64_n3/h2_ablation_bigbatch_seed_1/20260620_170835_d4_val_h2_abl_seed_1_spatial_nofact_dk64_depth3_lr2.84e-03_bs16384/best_model_epoch30_move48.90_res79.61.pth" \
    --log_file src/ppo/experiments/ppo_exp2_checkpoint.log \
    --save_dir src/ppo/experiments/checkpoints/checkpoint

echo "================================================================="
python3 src/ppo/train_ppo.py \
    --iterations 50 \
    --log_file src/ppo/experiments/ppo_exp1_scratch.log \
    --save_dir src/ppo/experiments/checkpoints/scratch

echo "Experiment 1 completed. Log saved to src/ppo/ppo_exp1_scratch.log"
echo "-----------------------------------------------------------------"

