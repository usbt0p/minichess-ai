mkdir -p experiments/test_run

# ----------------- (Flat representation, simple head) -----------------

PYTHONPATH=. .venv/bin/python3 src/training/train_transformer.py data/training_data_sample_val.txt 16 \
    --num_blocks 1 \
    --epochs 3 \
    --batch_size 512 \
    --attn_backend math \
    --autocast none \
    --lr 0.0060 \
    --representation simple \
    --factorized_policy \
    --run_name test_run \
    --save_dir experiments/test_run \
    2>&1 \
    | tee experiments/test_run/test_run.log