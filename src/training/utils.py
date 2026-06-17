import os
import time
import inspect
from datetime import timedelta
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.utils import time_this

def configure_optimizers(model, weight_decay, learning_rate, device_type, beta1=0.9, beta2=0.999, eps=1e-8):
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
    optimizer = optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), eps=eps, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

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
    This is a huge ass funcition, but it's worth it for some big runs with high d_k transformer values.
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
        outputs = model(features)
        if len(outputs) == 5:
            policy_logits, value_result, aux_from, aux_to, aux_promo = outputs
            dummy_target = torch.zeros(features.size(0), dtype=torch.long, device=device)
            from_loss = policy_criterion(aux_from, dummy_target)
            to_loss = policy_criterion(aux_to, dummy_target)
            promo_loss = policy_criterion(aux_promo, dummy_target)
            aux_loss = from_loss + to_loss + promo_loss
            loss = policy_criterion(policy_logits, moves) + value_criterion(value_result.squeeze(-1), results.float()) + 0.5 * aux_loss
        else:
            policy_logits, value_result = outputs
            loss = policy_criterion(policy_logits, moves) + value_criterion(value_result.squeeze(-1), results.float())
        
        policy_logits = policy_logits.masked_fill(~masks, -1e9)
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
        outputs = model(features)
        if len(outputs) == 5:
            policy_logits, value_result, aux_from, aux_to, aux_promo = outputs
            dummy_target = torch.zeros(features.size(0), dtype=torch.long, device=device)
            from_loss = policy_criterion(aux_from, dummy_target)
            to_loss = policy_criterion(aux_to, dummy_target)
            promo_loss = policy_criterion(aux_promo, dummy_target)
            aux_loss = from_loss + to_loss + promo_loss
            loss = policy_criterion(policy_logits, moves) + value_criterion(value_result.squeeze(-1), results.float()) + 0.5 * aux_loss
        else:
            policy_logits, value_result = outputs
            loss = policy_criterion(policy_logits, moves) + value_criterion(value_result.squeeze(-1), results.float())

        policy_logits = policy_logits.masked_fill(~masks, -1e9)
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
            outputs = model(val_features)
            if len(outputs) == 5:
                policy_logits, value_result, _, _, _ = outputs
            else:
                policy_logits, value_result = outputs

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed passes for validation
    print(f"Running {num_timed} timed validation steps...")
    start_val = time.time()
    for _ in range(num_timed):
        with torch.no_grad():
            outputs = model(val_features)
            if len(outputs) == 5:
                policy_logits, value_result, _, _, _ = outputs
            else:
                policy_logits, value_result = outputs

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

def plot_loss(train_losses, val_losses, val_move_accs, val_res_accs, save_dir=None):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), [l[0] for l in train_losses], label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
    
    plt.plot(range(len(train_losses)), [l[1] for l in train_losses], label='Train Policy Loss', linestyle='--')
    plt.plot(range(len(train_losses)), [l[2] for l in train_losses], label='Train Value Loss', linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    train_loss_path = os.path.join(save_dir, 'train_loss.png') if save_dir else 'train_loss.png'
    plt.savefig(train_loss_path)
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(val_move_accs)), val_move_accs, label='Val Move Acc')
    plt.plot(range(len(val_res_accs)), val_res_accs, label='Val Result Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    val_accuracy_path = os.path.join(save_dir, 'val_accuracy.png') if save_dir else 'val_accuracy.png'
    plt.savefig(val_accuracy_path)
    plt.close()

def decode_move_indices(move_idx, device):
    """
    Decodes move indices (0-703) into from_square (0-24), to_square (0-24), and promotion class (0-8).
    Used for the factorized policy head, but didn't exactly fit in other modules.
    """
    # Precomputed lookup tables for promotion moves (600+)
    # Mapping of traversal_idx (0-25) to rank/file coordinates
    promo_lookup_from = []
    promo_lookup_to = []
    for traversal_idx in range(26):
        is_black = (traversal_idx >= 13)
        t_idx = traversal_idx - 13 if is_black else traversal_idx
        
        # file_from based on standard Minichess pawn promotion rules
        if t_idx in [0, 1]:
            file_from = 0
            dx = t_idx
        elif t_idx in [2, 3, 4]:
            file_from = 1
            dx = t_idx - 3
        elif t_idx in [5, 6, 7]:
            file_from = 2
            dx = t_idx - 6
        elif t_idx in [8, 9, 10]:
            file_from = 3
            dx = t_idx - 9
        else:
            file_from = 4
            dx = t_idx - 12
            
        rank_from = 1 if is_black else 3
        rank_to = 0 if is_black else 4
        file_to = file_from + dx
        
        promo_lookup_from.append(rank_from * 5 + file_from)
        promo_lookup_to.append(rank_to * 5 + file_to)
        
    lookup_from = torch.tensor(promo_lookup_from, device=device)
    lookup_to = torch.tensor(promo_lookup_to, device=device)
    
    B = move_idx.size(0)
    from_sq = torch.zeros(B, dtype=torch.long, device=device)
    to_sq = torch.zeros(B, dtype=torch.long, device=device)
    promo_class = torch.zeros(B, dtype=torch.long, device=device)
    
    is_promo = (move_idx >= 600)
    is_normal = ~is_promo
    
    # Normal moves
    if is_normal.any():
        m_normal = move_idx[is_normal]
        from_sq[is_normal] = m_normal // 24
        to_sq_idx = m_normal % 24
        # If to_sq_idx >= from_sq, we add 1 to skip the diagonal (since a piece cannot move to its own square)
        to_sq[is_normal] = torch.where(to_sq_idx >= from_sq[is_normal], to_sq_idx + 1, to_sq_idx)
        promo_class[is_normal] = 0
        
    # Promotion moves
    if is_promo.any():
        m_promo = move_idx[is_promo]
        offset_idx = m_promo - 600
        p_idx = offset_idx // 26
        traversal_idx = offset_idx % 26
        
        from_sq[is_promo] = lookup_from[traversal_idx]
        to_sq[is_promo] = lookup_to[traversal_idx]
        
        is_black_promo = (traversal_idx >= 13)
        # 0: no promotion, 1-4: White (Q, R, B, N), 5-8: Black (Q, R, B, N)
        promo_class[is_promo] = torch.where(is_black_promo, 5 + p_idx, 1 + p_idx)
        
    return from_sq, to_sq, promo_class
