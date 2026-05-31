import torch
from torch.utils.data import Dataset, DataLoader
import os

from src.utils.utils import time_this
from src.models.dataset_parser import parse_minichess_text_file

class MinichessFfnDataset(Dataset):
    """
    Dataset wrapper for Feed-Forward Neural Networks (FFN).
    Yields flattened one-hot encoded board representations.
    """

    @time_this
    def __init__(self, file_path, promotions=False, use_cache=True):
        super().__init__()
        
        # Check for cached binary version
        suffix = ".promo" if promotions else ""
        cache_path = f"{file_path}{suffix}.pt"
        if use_cache and os.path.exists(cache_path):
            print(f">> Loading cached dataset from {cache_path}...")
            cached_data = torch.load(cache_path, weights_only=True)
            self.features = cached_data['features']
            self.moves = cached_data['moves']
            self.results = cached_data['results']
            self.scores = cached_data['scores']
            self.masks = cached_data['masks']
            self.halfmoves = cached_data.get('halfmoves', torch.zeros(len(self.features), dtype=torch.uint8))
            return

        print(f">> Parsing dataset from text: {file_path}")
        features_arr, halfmoves_arr, moves_arr, masks_arr, results_arr, scores_arr = parse_minichess_text_file(
            file_path, promotions=promotions
        )

        self.features = torch.from_numpy(features_arr)
        self.halfmoves = torch.from_numpy(halfmoves_arr)
        self.moves = torch.from_numpy(moves_arr).long()
        self.masks = torch.from_numpy(masks_arr).bool()
        self.results = torch.from_numpy(results_arr).float()
        self.scores = torch.from_numpy(scores_arr).float().unsqueeze(1)

        if use_cache:
            print(f">> Saving cached dataset to {cache_path}...")
            torch.save({
                'features': self.features,
                'halfmoves': self.halfmoves,
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


class MinichessTransformerDataset(Dataset):
    """
    Dataset wrapper for Transformer models.
    Yields 27-element sequences (25 board square tokens, 1 repetition token, 1 halfmove token).
    """

    @time_this 
    def __init__(self, file_path, promotions=False, use_cache=True):
        super().__init__()
        
        # Check for cached binary version
        suffix = ".transformer"
        cache_path = f"{file_path}{suffix}.pt"
        if use_cache and os.path.exists(cache_path):
            print(f">> Loading cached dataset from {cache_path}...")
            cached_data = torch.load(cache_path, weights_only=True)
            self.features = cached_data['features']
            self.moves = cached_data['moves']
            self.results = cached_data['results']
            self.scores = cached_data['scores']
            self.masks = cached_data['masks']
            self.halfmoves = cached_data.get('halfmoves', torch.zeros(len(self.features), dtype=torch.uint8))
            return

        print(f">> Parsing dataset from text: {file_path}")
        features_arr, halfmoves_arr, moves_arr, masks_arr, results_arr, scores_arr = parse_minichess_text_file(
            file_path, promotions=promotions
        )

        self.features = torch.from_numpy(features_arr)
        self.halfmoves = torch.from_numpy(halfmoves_arr)
        self.moves = torch.from_numpy(moves_arr).long()
        self.masks = torch.from_numpy(masks_arr).bool()
        self.results = torch.from_numpy(results_arr).float()
        self.scores = torch.from_numpy(scores_arr).float().unsqueeze(1)

        if use_cache:
            print(f">> Saving cached dataset to {cache_path}...")
            torch.save({
                'features': self.features,
                'halfmoves': self.halfmoves,
                'moves': self.moves,
                'masks': self.masks,
                'results': self.results,
                'scores': self.scores
            }, cache_path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        board = self.features[idx].long()  # Shape: (25,)
        # Default repetition count to 0. We do this because FEN does not have this info,
        # but we'll want it once self-play kicks in
        repetition = torch.tensor([0], dtype=torch.long)  
        halfmove = self.halfmoves[idx].long().unsqueeze(0)  # Shape: (1,)
        
        # Combine into a single sequence of 27 tokens
        flat_state = torch.cat([board, repetition, halfmove], dim=0)
        
        return flat_state, self.moves[idx], self.results[idx], self.scores[idx], self.masks[idx]


@time_this
def get_dataloaders(dataset, batch_size=128, train_ratio=0.96, num_workers=0):
    """
    Returns train and validation dataloaders for the given dataset.
    """
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

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
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )

    return train_loader, val_loader
