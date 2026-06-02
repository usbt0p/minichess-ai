import os
import time
import inspect
from datetime import timedelta
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.utils import time_this

def configure_optimizers(model, weight_decay, learning_rate, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    # Any parameter that is 2D will be weight decayed, otherwise no.
    # All weight tensors in matmuls and embeddings decay, biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
    # Create AdamW optimizer and use the fused version if it is available and device is CUDA
    fused_available = 'fused' in inspect.signature(optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = optim.AdamW(optim_groups, lr=learning_rate, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

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

def configure_profiler(config, prof_name, trace_filename=None, schedule=None):
    if not schedule:
        active_steps = max(1, config.profile_steps - 4)
        sc = dict(wait=2, warmup=2, active=active_steps, repeat=1)
    else:
        sc = schedule

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available() and config.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    assert bool(prof_name)
    trace_dir = f'./log/{prof_name}'
    os.makedirs(trace_dir, exist_ok=True)

    # Use a custom handler if user-defined filename is provided
    if trace_filename:
        def custom_handler(p):
            # Export Chrome trace with the exact user-defined filename
            # TensorBoard plugin looks for *.pt.trace.json files
            filename = f"{trace_filename}.pt.trace.json"
            trace_path = os.path.join(trace_dir, filename)
            p.export_chrome_trace(trace_path)
            print(f"[INFO] Custom trace saved to: '{trace_path}'")
        handler = custom_handler
    else:
        handler = torch.profiler.tensorboard_trace_handler(trace_dir)

    prof = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(**sc),
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )

    return prof

def estimate_training_time(model, train_loader, val_loader, config, num_warmup: int = 5, num_timed: int = 10):
    """
    Estimates training time per epoch and total training time by performing 
    actual forward/backward warmups and timed runs.
    """
    print("\n=== Estimating Training Time ===")
    device = config.device
    model = model.to(device)
    
    # Simple temporal optimizer for backward pass profiling
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    # Get a batch
    train_iter = iter(train_loader)
    try:
        features, moves, results, scores, masks = next(train_iter)
    except StopIteration:
        print("Empty training dataset.")
        return

    features = features.to(device)
    moves = moves.to(device)
    results = results.to(device)
    masks = masks.to(device)

    # Warmup passes for train (compiles kernels, cache allocations)
    model.train()
    print(f"Running {num_warmup} warm-up training steps...")
    for _ in range(num_warmup):
        optimizer.zero_grad()
        policy_logits, value_result = model(features)
        policy_logits = policy_logits.masked_fill(~masks, -1e9)
        loss = policy_criterion(policy_logits, moves) + value_criterion(value_result.squeeze(-1), results.float())
        loss.backward()
        optimizer.step()
    
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed passes for train
    print(f"Running {num_timed} timed training steps...")
    start_train = time.time()

    for i in range(num_timed):
        start = time.time()

        optimizer.zero_grad()
        policy_logits, value_result = model(features)

        policy_logits = policy_logits.masked_fill(~masks, -1e9)
        loss = policy_criterion(policy_logits, moves) + value_criterion(value_result.squeeze(-1), results.float())
        loss.backward()
        optimizer.step()

        # print time for forward
        delta = timedelta(seconds=round(time.time()-start, 4))
        print(f"\tstep {i+1}: Forward + backward + optim took {delta}")

    if device == "cuda":
        torch.cuda.synchronize()
    end_train = time.time()
    avg_step_train = (end_train - start_train) / num_timed

    # Warmup passes for validation
    model.eval()
    val_iter = iter(val_loader)
    try:
        val_features, val_moves, val_results, val_scores, val_masks = next(val_iter)
    except StopIteration:
        val_features, val_moves, val_results, val_scores, val_masks = features, moves, results, scores, masks

    val_features = val_features.to(device)
    val_results = val_results.to(device)
    val_masks = val_masks.to(device)

    print(f"Running {num_warmup} warm-up validation steps...")
    for _ in range(num_warmup):
        with torch.no_grad():
            policy_logits, value_result = model(val_features)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed passes for validation
    print(f"Running {num_timed} timed validation steps...")
    start_val = time.time()
    for _ in range(num_timed):
        with torch.no_grad():
            policy_logits, value_result = model(val_features)

    if device == "cuda":
        torch.cuda.synchronize()
    end_val = time.time()
    avg_step_val = (end_val - start_val) / num_timed

    # Compute epochs and times
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    epoch_train_time = num_train_batches * avg_step_train
    epoch_val_time = num_val_batches * avg_step_val
    total_epoch_time = epoch_train_time + epoch_val_time
    total_training_time = total_epoch_time * config.num_epochs

    print("\n------------------------------------------------")
    print(f"Estimated training stats (Batch size {config.batch_size}):")
    print(f"  Avg single training step:   {avg_step_train*1000:.2f} ms")
    print(f"  Avg single validation step: {avg_step_val*1000:.2f} ms")
    print(f"  Steps per epoch (Train):    {num_train_batches}")
    print(f"  Steps per epoch (Val):      {num_val_batches}")
    print(f"  Estimated epoch train time: {epoch_train_time:.2f} s")
    print(f"  Estimated epoch val time:   {epoch_val_time:.2f} s")
    print(f"  Estimated TOTAL epoch time: {total_epoch_time:.2f} s")
    print(f"  Estimated TOTAL train time ({config.num_epochs} epochs): {total_training_time:.2f} s ({total_training_time/60:.2f} minutes)")
    print("------------------------------------------------\n")

def plot_loss(train_losses, val_losses, val_move_accs, val_res_accs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), [l[0] for l in train_losses], label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
    
    plt.plot(range(len(train_losses)), [l[1] for l in train_losses], label='Train Policy Loss', linestyle='--')
    plt.plot(range(len(train_losses)), [l[2] for l in train_losses], label='Train Value Loss', linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('train_loss.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(val_move_accs)), val_move_accs, label='Val Move Acc')
    plt.plot(range(len(val_res_accs)), val_res_accs, label='Val Result Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig('val_accuracy.png')
    plt.close()
