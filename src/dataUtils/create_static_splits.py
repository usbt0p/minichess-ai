import os
import random
import sys
import numpy as np

def get_dest_paths(txt_path):
    dirname = os.path.dirname(txt_path)
    basename = os.path.basename(txt_path)
    
    # Determine split prefix
    if "depth2" in txt_path or "d2_with_promotions" in basename:
        prefix = "d2"
    elif "depth3" in txt_path or "d3" in basename:
        prefix = "d3"
    elif "depth4" in txt_path or "d4" in basename:
        prefix = "d4"
    elif "merged" in txt_path:
        prefix = "merged"
    else:
        prefix = os.path.splitext(basename)[0]
        
    train_val_out = os.path.join(dirname, f"{prefix}_val.txt")
    test_out = os.path.join("data", "test_splits", f"{prefix}_test.txt")
    
    return train_val_out, test_out

def create_static_splits(txt_path, test_ratio=0.03, seed=42):
    print(f">> Reading ply values from {txt_path} to detect game boundaries...")
    plies = []
    with open(txt_path, "r") as f:
        block_line_idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if block_line_idx == 3:
                # The line is formatted as: 'ply <value>'
                parts = line.split()
                if len(parts) >= 2 and parts[0] == 'ply':
                    ply_val = int(parts[1])
                else:
                    raise ValueError(f"Expected 'ply <val>' line, got: '{line}'")
                plies.append(ply_val)
            block_line_idx = (block_line_idx + 1) % 6

    num_samples = len(plies)
    print(f">> Parsed {num_samples} ply values (blocks).")

    # Detect game starts (where ply does not increase by 1)
    game_starts = []
    prev_ply = -1
    for idx, ply in enumerate(plies):
        if prev_ply == -1 or ply != prev_ply + 1:
            game_starts.append(idx)
        prev_ply = ply
    
    game_starts.append(num_samples) # Sentinel
    num_games = len(game_starts) - 1
    print(f">> Detected {num_games} games in the dataset.")

    # Deterministically split games
    random.seed(seed)
    game_indices = list(range(num_games))
    random.shuffle(game_indices)

    test_game_count = int(num_games * test_ratio)
    test_game_set = set(game_indices[:test_game_count])

    # Map block indices to splits using numpy boolean array for speed and memory efficiency
    is_test_block = np.zeros(num_samples, dtype=bool)
    test_sample_count = 0
    train_val_sample_count = 0
    
    for g in range(num_games):
        start = game_starts[g]
        end = game_starts[g+1]
        if g in test_game_set:
            is_test_block[start:end] = True
            test_sample_count += (end - start)
        else:
            train_val_sample_count += (end - start)

    train_val_out, test_out = get_dest_paths(txt_path)
    
    print(f">> Split summary:")
    print(f"   Train/Val: {num_games - test_game_count} games ({train_val_sample_count} samples, {train_val_sample_count/num_samples*100:.2f}%) -> {train_val_out}")
    print(f"   Test:      {test_game_count} games ({test_sample_count} samples, {test_sample_count/num_samples*100:.2f}%) -> {test_out}")

    print(f">> Streaming blocks to destination files...")
    os.makedirs(os.path.dirname(test_out), exist_ok=True)
    os.makedirs(os.path.dirname(train_val_out), exist_ok=True)
    
    with open(txt_path, "r") as f_in, \
         open(train_val_out, "w") as f_train_val, \
         open(test_out, "w") as f_test:
        
        block_lines = []
        block_idx = 0
        
        for line in f_in:
            if not line.strip():
                continue
            block_lines.append(line)
            if len(block_lines) == 6:
                # Write to the appropriate file
                f_out = f_test if is_test_block[block_idx] else f_train_val
                for bl in block_lines:
                    f_out.write(bl)
                block_lines = []
                block_idx += 1

    print(">> Done successfully!")

if __name__ == "__main__":
    txt_path = "data/gardner_depth4/gen_gardner_d4.txt"
    if len(sys.argv) > 1:
        txt_path = sys.argv[1]
        
    create_static_splits(txt_path, test_ratio=0.03)
