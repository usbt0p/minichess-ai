import os
import time
import torch.nn as nn

class TensorBoardLogger:
    """
    Custom TensorBoard logger to handle metric and histogram logging.
    Takes care itself of not logging when Tensorboard is not available, or when
    the run directory is not specified / there is no writer.
    """
    def __init__(self, run_dir: str = None):
        self.writer = None
        if run_dir:
            try:
                import torch.utils.tensorboard as tb # circular import...
                self.writer = tb.SummaryWriter(log_dir=run_dir)
                print(f"[INFO] TensorBoard SummaryWriter initialized at: '{run_dir}'")
            except Exception as e:
                print(f"[WARNING] Could not initialize TensorBoard writer: {e}")

    def log_epoch(self, epoch: int, metrics: dict):
        if self.writer is None:
            return
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, epoch)

    def log_gradient_histograms(self, epoch: int, model: nn.Module):
        if self.writer is None:
            return
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Detach and copy to CPU to avoid memory/device leakage
                self.writer.add_histogram(f"Gradients/{name}", param.grad.detach().cpu(), epoch)

    def close(self):
        if self.writer is not None:
            self.writer.close()

def save_run_metadata(trace_dir, config, encoder_config, description):
    """
    Saves TrainingConfig, EncoderConfig, and user description as a markdown file 
    and also logs them as TensorBoard text summaries.
    """
    os.makedirs(trace_dir, exist_ok=True)
    
    # 1. Write metadata to context.md
    md_path = os.path.join(trace_dir, "context.md")
    
    lines = []
    if description:
        lines.append(f"# Run Description\n{description}\n")
    else:
        lines.append("# Run Description\nNo description provided.\n")
        
    lines.append("# Configurations\n")
    lines.append("## TrainingConfig")
    lines.append("```python")
    lines.append(str(config))
    lines.append("```\n")
    
    if encoder_config is not None:
        lines.append("## EncoderConfig")
        lines.append("```python")
        lines.append(str(encoder_config))
        lines.append("```\n")
        
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Run metadata written to '{md_path}'")
    
    # 2. Write to TensorBoard using SummaryWriter
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=trace_dir)
        writer.add_text("Metadata/Description", description or "No description provided.")
        writer.add_text("Metadata/TrainingConfig", str(config))
        if encoder_config is not None:
            writer.add_text("Metadata/EncoderConfig", str(encoder_config))
        writer.close()
    except Exception as e:
        print(f"[WARNING] Could not write metadata to TensorBoard: {e}")

def generate_run_name(config: 'TrainingConfig', encoder_config: 'EncoderConfig') -> str:
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
    repr_str = config.representation
    fact_str = "fact" if config.use_factorized_policy else "nofact"
    
    return f"{timestamp}_{dataset_name}_{base_run}_{repr_str}_{fact_str}_dk{d_k}_depth{depth}_lr{lr:.2e}_bs{bs}"