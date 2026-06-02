import os
import argparse
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration class for the training process.
    """

    # these must be explicitly set 
    data_path: str
    use_cache: bool

    batch_size: int
    train_ratio: float
    num_workers: int
    num_epochs: int
    patience: int
    lr: float = 2e-3
    weight_decay: float = 2e-5
    
    subsample_ratio: float = 1.0
    
    # these are defaults and should rarely change
    promotions: bool = True
    device: str = "cuda"

    # profiler stuff
    profile_name: str = None
    profile_steps: int = 50
    profile_desc: str = None
    profile_filename: str = None

    def __post_init__(self):
        assert 0.0 < self.train_ratio <= 0.99, "train_ratio must be between 0 and 0.99"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert os.path.exists(self.data_path), "The data file does not exist!"


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniChess Transformer")
    parser.add_argument("data_path", type=str, help="Path to the dataset file")
    parser.add_argument("embed_dim", type=int, help="Embedding dimension (d_k)")
    parser.add_argument("--profile", type=str, default=None, help="Name of the profiling run (enables profiling)")
    parser.add_argument("--profile_steps", type=int, default=50, help="Number of profiling steps (default: 50)")
    parser.add_argument("--profile_desc", type=str, default=None, help="String description of the profiling run")
    parser.add_argument("--profile_filename", type=str, default=None, help="Custom filename for the trace (default: worker name)")
    parser.add_argument("--subsample", type=float, default=1.0, help="Percentage of the dataset to use (e.g. 0.5 for 50%%)")
    
    return parser.parse_args()
