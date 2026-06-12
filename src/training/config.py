import os
import time
import argparse
from dataclasses import dataclass
from src.models.transformerEncoder import EncoderConfig

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
    custom_init: bool = False
    run_name: str = None
    
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
        # Allow running with virtual paths that only have cached .pt versions
        cache_pt = f"{self.data_path}.transformer.pt"
        cache_promo = f"{self.data_path}.promo.pt"
        cache_reg = f"{self.data_path}.pt"
        exists = (os.path.exists(self.data_path) or 
                  os.path.exists(cache_pt) or 
                  os.path.exists(cache_promo) or 
                  os.path.exists(cache_reg))
        assert exists, f"Neither the data file nor any cached version exists for: {self.data_path}"


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniChess Transformer")
    parser.add_argument("data_path", type=str, help="Path to the dataset file")
    parser.add_argument("embed_dim", type=int, help="Embedding dimension (d_k)")
    
    # Optimizer & Training Hyperparams
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate (default: 2e-3)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of transformer blocks (default: 4)")
    parser.add_argument("--mlp_expand", type=int, default=4, help="MLP hidden dimension expand factor (default: 4)")
    parser.add_argument("--custom_init", action="store_true", help="Enable GPT-2 style weight initialization")
    parser.add_argument("--run_name", type=str, default=None, help="Descriptive name of the run to save metadata and logs")

    # Profiler stuff
    parser.add_argument("--profile", type=str, default=None, help="Name of the profiling run (enables profiling)")
    parser.add_argument("--profile_steps", type=int, default=50, help="Number of profiling steps (default: 50)")
    parser.add_argument("--profile_desc", type=str, default=None, help="String description of the profiling run")
    parser.add_argument("--profile_filename", type=str, default=None, help="Custom filename for the trace (default: worker name)")
    
    parser.add_argument("--subsample", type=float, default=1.0, help="Percentage of the dataset to use (e.g. 0.5 for 50%%)")
    
    # careful with these, bad combinations can break things silently! 
    parser.add_argument("--attn_backend", type=str, choices=["math", "efficient", "flash", "auto"], default="auto", help="Scaled Dot-Product Attention backend (default: auto)")
    parser.add_argument("--autocast", type=str, choices=["bfloat16", "float16", "float32", "auto", "none"], default="bfloat16", help="Autocast precision mode (default: bfloat16)")
    
    return parser.parse_args()


def generate_run_name(config: TrainingConfig, encoder_config: EncoderConfig) -> str:
    """Generates a normalized, unique, and descriptive run name for the experiment."""
    # Base dataset name from path
    dataset_name = "data"
    if config.data_path:
        base_name = os.path.basename(config.data_path)
        dataset_name = os.path.splitext(base_name)[0]
        # Remove common extensions/suffixes
        dataset_name = dataset_name.replace(".train_val", "").replace(".test", "").replace(".txt", "")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Base name (default to 'run' if not provided)
    base_run = config.run_name if config.run_name else "run"
    # Clean directory delimiters
    base_run = base_run.replace("/", "_").replace("\\", "_")
    
    # Extract architecture params
    d_k = encoder_config.embed_dim if encoder_config else "unknown"
    depth = encoder_config.num_blocks if encoder_config else "unknown"
    lr = config.lr
    bs = config.batch_size
    
    return f"{timestamp}_{dataset_name}_{base_run}_dk{d_k}_depth{depth}_lr{lr:.2e}_bs{bs}"