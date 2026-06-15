import torch
import numpy as np
import time
import os
import multiprocessing as mp
from functools import partial, lru_cache
import pyffish  # pyrefly: ignore [missing-import]
from src.utils.utils import time_this, pretty_time # pyrefly: ignore [missing-import]

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
    to_sq_idx = to_sq - 1 if to_sq > from_sq else to_sq
    
    if promotions and len(move_str) > 4:
        # promotions need to be handled separately
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

def _parse_chunk(chunk_data, promotions, piece_map):
    num_samples = len(chunk_data)
    features_arr = np.full((num_samples, 25), 12, dtype=np.uint8)
    halfmoves_arr = np.zeros(num_samples, dtype=np.uint8)
    active_players_arr = np.zeros(num_samples, dtype=np.uint8)
    moves_arr = np.zeros(num_samples, dtype=np.int16)
    masks_arr = np.zeros((num_samples, 704 if promotions else 600), dtype=np.bool_)
    results_arr = np.zeros(num_samples, dtype=np.float16)
    scores_arr = np.zeros(num_samples, dtype=np.float32)

    for i, lines in enumerate(chunk_data):
        fen_str = lines[0][4:].strip()
        parse_fen_to_features(fen_str, piece_map, features_arr[i])
        
        # Parse halfmove clock and active player
        parts = fen_str.split(" ")
        halfmove = int(parts[4]) if len(parts) > 4 else 0
        halfmoves_arr[i] = halfmove
        active_player = 1 if len(parts) > 1 and parts[1] == 'w' else 0
        active_players_arr[i] = active_player
        
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
        
        results_arr[i] = int(lines[4][7:].strip())
            
        scores_arr[i] = float(lines[2][6:].strip())
        
    return features_arr, halfmoves_arr, active_players_arr, moves_arr, masks_arr, results_arr, scores_arr

@time_this
def parse_minichess_text_file(file_path, promotions=False, return_active_player=False):
    """
    Parses the raw Minichess text file and returns the parsed numpy arrays.

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
    piece_map = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
    }

    # fast pass to count blocks
    with open(file_path, "r") as f:
        lines_count = sum(1 for line in f if line.strip())
    num_samples = lines_count // 6

    # chunks for multiprocessing, generator for leaving memory
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
    # need this in order to pass arguments
    worker = partial(_parse_chunk, promotions=promotions, piece_map=piece_map)
    
    features_list, halfmoves_list, active_players_list, moves_list, masks_list, results_list, scores_list = [], [], [], [], [], [], []
    processed_samples = 0
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for f_arr, h_arr, ap_arr, m_arr, mask_arr, r_arr, s_arr in pool.imap(worker, chunk_generator()):
            features_list.append(f_arr)
            halfmoves_list.append(h_arr)
            active_players_list.append(ap_arr)
            moves_list.append(m_arr)
            masks_list.append(mask_arr)
            results_list.append(r_arr)
            scores_list.append(s_arr)
            
            processed_samples += len(f_arr)
            eta = (time.time() - start_time) / (processed_samples / num_samples + 1e-9)
            time_left = eta * (num_samples - processed_samples) / num_samples
            percent = (processed_samples / num_samples) * 100
            print(f">> Processing sample {processed_samples} ({percent:.2f}%) in {time.time() - start_time:.1f}s | Time Left: {pretty_time(int(time_left))} | ETA: {pretty_time(int(eta))}", end="\r", flush=True)

    print() # Final newline

    features_arr = np.concatenate(features_list)
    halfmoves_arr = np.concatenate(halfmoves_list)
    active_players_arr = np.concatenate(active_players_list)
    moves_arr = np.concatenate(moves_list)
    masks_arr = np.concatenate(masks_list)
    results_arr = np.concatenate(results_list)
    scores_arr = np.concatenate(scores_list)

    if return_active_player:
        return features_arr, halfmoves_arr, active_players_arr, moves_arr, masks_arr, results_arr, scores_arr
    else:
        return features_arr, halfmoves_arr, moves_arr, masks_arr, results_arr, scores_arr
