import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.utils import time_this

import os
import numpy as np

class MinichessTextDataset(Dataset):
    """
    Parses a text file with Minichess data blocks.
    Each block is expected to be exactly 6 lines:
    ```
    fen ...
    move ...
    score ...
    ply ...
    result ...
    e
    ```
    - Features: 5x5 board with one-hot encoded pieces (12 types + 1 empty)
    - Moves: 5^2 * (5^2 - 1) = 600 possible moves (from any square to any other square)
    - Result: -1.0 for white win, 0.0 for draw, 1.0 for black win
    """

    @time_this
    def __init__(self, file_path, use_cache=True, result_mode="classification"):
        '''
        The result_mode can be:
        - "classification": the result is treated as categorical, with 3 classes (white win(0), draw(1), black win(2))
        - "regression": the result is treated as a continuous value between -1 and 1
        '''
        super().__init__()
        
        # Check for cached binary version (loads in milliseconds instead of minutes)
        cache_path = file_path + ".pt"
        if use_cache and os.path.exists(cache_path):
            print(f">> Loading cached dataset from {cache_path}...")
            cached_data = torch.load(cache_path, weights_only=True)
            self.features = cached_data['features']
            self.moves = cached_data['moves']
            self.results = cached_data['results']
            self.scores = cached_data['scores']
            return

        print(f">> Parsing dataset from text: {file_path}")
        piece_map = {
            "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
            "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
        }

        # 1. Fast pass to count blocks
        with open(file_path, "r") as f:
            lines_count = sum(1 for line in f if line.strip())
        num_samples = lines_count // 6
        
        # 2. Pre-allocate compact NumPy arrays
        # previously, it was:
        # Casilla 0: 
        # [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (13 floats = 52 bytes)
        # Casilla 1: 
        # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] (13 floats = 52 bytes)
        # ...
        # which ended up being 325 floats per sample, so 325*4 bytes = 1.3kB per sample, 
        # times 1.8M instances => ~2.4 GB

        # now we store the board as 25 bytes (0-12 indices) instead of 325 floats!
        # 1.8M instances * 25 bytes = ~45 MB in RAM
        
        features_arr = np.full((num_samples, 25), 12, dtype=np.uint8) # 12 is 'empty'
        moves_arr = np.zeros(num_samples, dtype=np.int16) # cant use int8, need 600 vals
        if result_mode == "classification":
            results_arr = np.zeros(num_samples, dtype=np.int8)
        elif result_mode == "regression":
            results_arr = np.zeros(num_samples, dtype=np.float16)
        scores_arr = np.zeros(num_samples, dtype=np.float32) # TODO verify this is fine

        with open(file_path, "r") as f:
            lines = []
            idx = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines.append(line)

                if len(lines) == 6:
                    fen_line = lines[0]
                    move_line = lines[1]
                    score_line = lines[2]
                    result_line = lines[4]

                    if fen_line.startswith("fen"):
                        # Parse FEN
                        board_part = fen_line[4:].strip().split(" ")[0]
                        for r_idx, row in enumerate(board_part.split("/")):
                            rank = 4 - r_idx  # FEN gives rank 5 down to 1
                            file_idx = 0
                            for char in row:
                                if char.isdigit():
                                    file_idx += int(char) # Skip empty squares
                                else:
                                    features_arr[idx, rank * 5 + file_idx] = piece_map[char]
                                    file_idx += 1

                        # Parse Move
                        move = move_line[5:].strip()[:4] # TODO change when we i implement promotion
                        file_from = ord(move[0]) - ord("a")
                        rank_from = int(move[1]) - 1
                        file_to = ord(move[2]) - ord("a")
                        rank_to = int(move[3]) - 1

                        # move looks like e3b5 in the original format,
                        # and we encode it as a number from 0 to 599.
                        # from_sq is the index of the starting square (0-24).
                        # to_sq_idx is the index of the ending square relative to the starting square (0-23).
                        from_sq = rank_from * 5 + file_from
                        to_sq = rank_to * 5 + file_to
                        # this results in this board indexing:
                        # | 20 | 21 | 22 | 23 | 24 |
                        # | 15 | 16 | 17 | 18 | 19 |
                        # | 10 | 11 | 12 | 13 | 14 |
                        # | 5  | 6  | 7  | 8  | 9  |
                        # | 0  | 1  | 2  | 3  | 4  |

                        # Policy size is 600 (5^2 * (5^2 - 1)). We adjust the to_sq index.
                        # so this makes a1b1 the smallest move (0), a1e1 is move (3), a1a2 is move (4),
                        # but a2a1 is move 120 (5*24 + 0)
                        # the biggest move has idx 599, which is 24*24 + 23, or e5d5
                        to_sq_idx = to_sq - 1 if to_sq > from_sq else to_sq
                        
                        moves_arr[idx] = from_sq * 24 + to_sq_idx
                        if result_mode == "classification":
                            results_arr[idx] = int(result_line[7:].strip()) + 1
                        elif result_mode == "regression":
                            results_arr[idx] = int(result_line[7:].strip())
                        else:
                            raise ValueError(f"Invalid result_mode: {result_mode}")
                        scores_arr[idx] = float(score_line[6:].strip())
                        
                        idx += 1

                    lines.clear()

        # Slice to actual size just in case there were invalid lines
        features_arr = features_arr[:idx]
        moves_arr = moves_arr[:idx]
        results_arr = results_arr[:idx]
        scores_arr = scores_arr[:idx]

        # TODO check dtypes, for correctness
        self.features = torch.from_numpy(features_arr)
        self.moves = torch.from_numpy(moves_arr).long()
        # in case we switch result_mode, the cached results might be of the wrong type
        if result_mode == "classification":
            self.results = torch.from_numpy(results_arr).float()#.long()
        elif result_mode == "regression":
            self.results = torch.from_numpy(results_arr).float()
        self.scores = torch.from_numpy(scores_arr).float().unsqueeze(1)

        if use_cache:
            print(f">> Saving cached dataset to {cache_path}...")
            torch.save({
                'features': self.features,
                'moves': self.moves,
                'results': self.results,
                'scores': self.scores
            }, cache_path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        '''
        Thanks to this, we keep the one-hot reconstruction on the fly because it is faster
        to just load 25 uint8 values instead of 25*13 floats. Avoid memory bound issues since
        computing this with broadcasting is basically free.
        '''
        compact_board = self.features[idx] # Shape: (25,) of uint8 (0-12 values)
        
        # Reconstruct the one-hot float tensor on the fly!
        one_hot = torch.zeros((25, 13), dtype=torch.float32)
        one_hot.scatter_(1, compact_board.unsqueeze(1).long(), 1.0)
        
        return one_hot.flatten(), self.moves[idx], self.results[idx], self.scores[idx]

@time_this
def get_dataloaders(dataset, batch_size=128, train_ratio=0.96, num_workers=0):
    """
    Returns train and validation dataloaders for the MinichessTextDataset.
    """
    
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    # We use a generator for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
