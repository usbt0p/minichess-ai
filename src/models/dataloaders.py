import torch
from torch.utils.data import Dataset, DataLoader


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

    def __init__(self, file_path):
        super().__init__()
        features_list = []
        moves_list = []
        results_list = []
        scores_list = []
        plys_list = []

        piece_map = {
            "P": 0,
            "N": 1,
            "B": 2,
            "R": 3,
            "Q": 4,
            "K": 5,
            "p": 6,
            "n": 7,
            "b": 8,
            "r": 9,
            "q": 10,
            "k": 11,
        }

        with open(file_path, "r") as f:
            lines = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines.append(line)

                if len(lines) == 6:
                    fen_line = lines[0]
                    move_line = lines[1]
                    score_line = lines[2]
                    ply_line = lines[3]
                    result_line = lines[4]

                    if fen_line.startswith("fen"):
                        # 1. Parse FEN
                        fen = fen_line[4:].strip()
                        board_part = fen.split(" ")[0]
                        rows = board_part.split("/")

                        # One-hot encoded board (5^2 * 13)
                        tensor = torch.zeros((25, 13), dtype=torch.float32)
                        for r_idx, row in enumerate(rows):
                            rank = 4 - r_idx  # FEN gives rank 5 down to 1
                            file_idx = 0
                            for char in row:
                                if char.isdigit():
                                    empty_count = int(char)
                                    for _ in range(empty_count):
                                        tensor[rank * 5 + file_idx, 12] = 1.0
                                        file_idx += 1
                                else:
                                    piece_idx = piece_map[char]
                                    tensor[rank * 5 + file_idx, piece_idx] = 1.0
                                    file_idx += 1

                        features_list.append(tensor.flatten())

                        # 2. Parse Move (e.g., "move d1e2" or "move a4a5q")
                        move = move_line[5:].strip()[
                            :4
                        ]  # handle promotion by taking only first 4 chars
                        # ord('a') is 97, so 'a' becomes 0, and the rest get indexed
                        file_from = ord(move[0]) - ord("a")
                        rank_from = int(move[1]) - 1
                        file_to = ord(move[2]) - ord("a")
                        rank_to = int(move[3]) - 1

                        # move is e3b5 in the original format,
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
                        move_idx = from_sq * 24 + to_sq_idx
                        moves_list.append(move_idx)

                        # 3. Parse Result (target value)
                        result = int(result_line[7:].strip())
                        results_list.append(result)

                        # 4. parse score
                        score_line = score_line[6:].strip()
                        score = int(score_line)
                        scores_list.append(score)

                        # ply is not useful yet

                    lines.clear()

        if features_list:
            self.features = torch.stack(features_list)
            self.moves = torch.tensor(moves_list, dtype=torch.long)
            # Map results from (-1, 0, 1) to class indices (0, 1, 2) for use in cross entropy
            self.results = torch.tensor([r + 1 for r in results_list], dtype=torch.long)
            self.scores = torch.tensor(scores_list, dtype=torch.float32).unsqueeze(1)
        else:
            self.features = torch.empty((0, 325), dtype=torch.float32)
            self.moves = torch.empty((0,), dtype=torch.long)
            self.results = torch.empty((0,), dtype=torch.long)
            self.scores = torch.empty((0, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.moves[idx], self.results[idx], self.scores[idx]


def get_dataloaders(file_path, batch_size=128, train_ratio=0.96, num_workers=0):
    """
    Returns train and validation dataloaders for the MinichessTextDataset.
    """
    dataset = MinichessTextDataset(file_path)

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
