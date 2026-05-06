import torch
from torch.utils.data import Dataset, DataLoader
import pyffish

from src.utils.utils import time_this, pretty_time

import time
import os
import numpy as np
from functools import lru_cache

import multiprocessing as mp
from functools import partial

@lru_cache(maxsize=1024)
def uci_to_index(move_str: str, promotions: bool) -> int:
    """Helper to decode a UCI string into our custom indexing system."""
    file_from = ord(move_str[0]) - ord("a")
    rank_from = int(move_str[1]) - 1
    file_to = ord(move_str[2]) - ord("a")
    rank_to = int(move_str[3]) - 1
    
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
    
    if promotions and len(move_str) > 4:
        # promotions need to be handled separately
        # on a 5x5 board there are 104 possible promotion moves:
        # moves from a and e files have two options (move or pawn capture), and bcd have 3
        # and we have 4 possible promotions and two players, so (2+3+3+3+2)*4*2 = 13*4*2 = 104
        promo_char = move_str[4].lower()
        promo_types = {'q': 0, 'r': 1, 'b': 2, 'n': 3}
        p_idx = promo_types[promo_char]
        is_black = from_sq < 10
        dx = file_to - file_from
        base_idx = [0, 2, 5, 8, 11][file_from]
        offset = dx if file_from == 0 else dx + 1
        traversal_idx = base_idx + offset
        if is_black:
            traversal_idx += 13
            
        # this is the offset: (13*2) * p, where p in [0,3], plus the black offset +
        # traversal (moving encodes from where to where it moved) + 1 for the move
        # so max would be 600 + 103 (black pawn promoting to knight from e2 to d1)
        # 600 + (3*26) + 11 + 13 + 1 = 703
        return 600 + p_idx * 26 + traversal_idx
    else:
        return from_sq * 24 + to_sq_idx

def parse_fen_to_features(fen_str: str, piece_map: dict, features_out: np.ndarray):
    """Parses a Gardner FEN string and populates the features_out array in place."""
    board_part = fen_str.split(" ")[0]
    for r_idx, row in enumerate(board_part.split("/")):
        rank = 4 - r_idx  # FEN gives rank 5 down to 1
        file_idx = 0
        for char in row:
            if char.isdigit():
                file_idx += int(char) # Skip empty squares
            else:
                features_out[rank * 5 + file_idx] = piece_map[char]
                file_idx += 1

def _parse_chunk(chunk_data, promotions, result_mode, piece_map):

    # now we store the board as 25 bytes (0-12 indices) instead of 325 floats!
    # 1.8M instances * 25 bytes = ~45 MB in RAM
    num_samples = len(chunk_data)
    features_arr = np.full((num_samples, 25), 12, dtype=np.uint8)
    moves_arr = np.zeros(num_samples, dtype=np.int16)
    masks_arr = np.zeros((num_samples, 704 if promotions else 600), dtype=np.bool_)
    
    if result_mode == "classification":
        results_arr = np.zeros(num_samples, dtype=np.int8)
    elif result_mode == "regression":
        results_arr = np.zeros(num_samples, dtype=np.float16)
    else:
        results_arr = np.zeros(num_samples, dtype=np.float32)
    scores_arr = np.zeros(num_samples, dtype=np.float32)

    for i, lines in enumerate(chunk_data):
        fen_str = lines[0][4:].strip()
        parse_fen_to_features(fen_str, piece_map, features_arr[i])
        
        legal_moves = pyffish.legal_moves("gardner", fen_str, [])
        for m in legal_moves:
            m_idx = uci_to_index(m, promotions)
            masks_arr[i, m_idx] = True                           

        move = lines[1][5:].strip()
        if promotions and len(move) == 4:
            file_from = ord(move[0]) - ord("a")
            rank_from = int(move[1]) - 1
            rank_to = int(move[3]) - 1
            from_sq = rank_from * 5 + file_from
            piece = features_arr[i, from_sq]
            if piece in (0, 6) and (rank_to == 4 or rank_to == 0):
                move += 'q'
        
        moves_arr[i] = uci_to_index(move, promotions)
        
        if result_mode == "classification":
            results_arr[i] = int(lines[4][7:].strip()) + 1
        elif result_mode == "regression":
            results_arr[i] = int(lines[4][7:].strip())
            
        scores_arr[i] = float(lines[2][6:].strip())
        
    return features_arr, moves_arr, masks_arr, results_arr, scores_arr

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
    def __init__(self, file_path, promotions=False, use_cache=True, result_mode="classification"):
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
            self.masks = cached_data['masks']
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

        def chunk_generator():
            chunk_size = 10000
            chunk = []
            with open(file_path, "r") as f:
                lines = []
                for line in f:
                    line = line.strip()
                    if not line: continue
                    lines.append(line)
                    if len(lines) == 6:
                        chunk.append(lines)
                        lines = []
                        if len(chunk) == chunk_size:
                            yield chunk
                            chunk = []
                if chunk:
                    yield chunk

        start_time = time.time()
    
        # use partial because _parse_chunk has 5 arguments, and we can only pass one to the pool
        worker = partial(_parse_chunk, promotions=promotions, result_mode=result_mode, piece_map=piece_map)
        
        features_list, moves_list, masks_list, results_list, scores_list = [], [], [], [], []
        processed_samples = 0
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for f_arr, m_arr, mask_arr, r_arr, s_arr in pool.imap(worker, chunk_generator()):
                features_list.append(f_arr)
                moves_list.append(m_arr)
                masks_list.append(mask_arr)
                results_list.append(r_arr)
                scores_list.append(s_arr)
                
                processed_samples += len(f_arr)
                eta = (time.time() - start_time) / (processed_samples / num_samples + 1e-9)
                time_left = eta * (num_samples - processed_samples) / num_samples
                percent = (processed_samples / num_samples) * 100
                print(f">> Processing sample {processed_samples} ({percent:.2f}%) in {time.time() - start_time:.1f}s | Time Left: {pretty_time(int(time_left))} | ETA: {pretty_time(int(eta))}", end="\r", flush=True)

        print() # Salto de línea final al terminar el bucle

        features_arr = np.concatenate(features_list)
        moves_arr = np.concatenate(moves_list)
        masks_arr = np.concatenate(masks_list)
        results_arr = np.concatenate(results_list)
        scores_arr = np.concatenate(scores_list)

        # TODO check dtypes, for correctness
        self.features = torch.from_numpy(features_arr)
        self.moves = torch.from_numpy(moves_arr).long()
        self.masks = torch.from_numpy(masks_arr).bool()
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
                'masks': self.masks,
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
        
        return one_hot.flatten(), self.moves[idx], self.results[idx], self.scores[idx], self.masks[idx]

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
