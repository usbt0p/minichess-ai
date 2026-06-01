from importlib import metadata
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
import matplotlib.pyplot as plt
import time
import os
import sys
import inspect

from datetime import timedelta

from src.utils.utils import time_this, count_params
from src.models.dataloaders import get_dataloaders, MinichessTransformerDataset
from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig

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
    
    # these are defaults and should rarely change
    promotions: bool = True
    device: str = "cuda"

    def __post_init__(self):
        assert 0.0 < self.train_ratio <= 0.99, "train_ratio must be between 0 and 0.99"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert os.path.exists(self.data_path), "The data file does not exist!"


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


@time_this
def train_model(
    model,
    train_loader, 
    val_loader,
    config: TrainingConfig,
):
    model = model.to(config.device)
    optimizer = configure_optimizers(
        model, 
        weight_decay=config.weight_decay, 
        learning_rate=config.lr, 
        device_type=config.device
    )
    print(f"Using device: {config.device}")

    # Loss criteria
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    train_losses = [] # list of train loss per epoch: (total, policy, value)
    val_losses = []

    val_move_accs = []
    val_res_accs = []

    patience_count = 0
    debug_flag = True

    best_move_acc = float("-inf")
    best_result_acc = float("-inf")
    best_epoch = 1
    prev_validation_loss = float("inf")

    try:
        for epoch in range(config.num_epochs):
            model.train()
            total_loss, total_policy_loss, total_value_loss = 0.0, 0.0, 0.0

            start_time = time.time()

            for features, moves, results, scores, masks in train_loader:
                features = features.to(config.device)
                moves = moves.to(config.device)
                results = results.to(config.device)
                masks = masks.to(config.device)

                optimizer.zero_grad()

                
                policy_logits, value_result = model(features)

                # Apply legal moves masking
                policy_logits = policy_logits.masked_fill(~masks, -1e9)


                policy_loss = policy_criterion(policy_logits, moves)
                value_loss = value_criterion(value_result.squeeze(-1), results.float())

                loss = policy_loss + value_loss

                if debug_flag: # useful for debugging tensor dims
                    print("[DEBUG TENSOR SIZES]:")
                    print("\tfeatures (flat_state): ", features.shape)
                    print("\tresults: ", results.shape)
                    print("\tvalue_result: ", value_result.shape)
                    print("\tmoves: ", moves.shape)
                    print("\tpolicy_logits: ", policy_logits.shape)
                    print("\n")
                    debug_flag = False

                loss.backward()

                # Gradient clipping for Transformer block stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Accumulate sum of losses for correct normalization at end of epoch
                total_loss += loss.item() * features.size(0)
                total_policy_loss += policy_loss.item() * features.size(0)
                total_value_loss += value_loss.item() * features.size(0)

            num_samples = len(train_loader.dataset)
            total_loss /= num_samples
            total_policy_loss /= num_samples
            total_value_loss /= num_samples
            train_losses.append((total_loss, total_policy_loss, total_value_loss))

            epoch_time = time.time() - start_time

            # Validation phase
            model.eval()
            val_loss, correct_moves, correct_results, total_val_samples = 0.0, 0, 0, 0

            with torch.no_grad():
                for features, moves, results, scores, masks in val_loader:
                    features = features.to(config.device)
                    moves = moves.to(config.device)
                    results = results.to(config.device)
                    masks = masks.to(config.device)

                    policy_logits, value_result = model(features)
                    
                    # get value loss and "correct" results (round)
                    value_loss = value_criterion(value_result.squeeze(-1), results.float())
                    predicted_results = torch.round(value_result.squeeze(-1))
                    correct_results += (predicted_results == results).sum().item()

                    # get polocy logits, loss and correct moves for accuracy
                    policy_logits = policy_logits.masked_fill(~masks, -1e9)
                    policy_loss = policy_criterion(policy_logits, moves)
                    _, predicted_moves = torch.max(policy_logits, 1)
                    correct_moves += (predicted_moves == moves).sum().item()

                    # overall validation loss
                    val_loss += (policy_loss + value_loss).item() * features.size(0)
                    total_val_samples += moves.size(0)

            val_move_acc = correct_moves / total_val_samples if total_val_samples > 0 else 0 
            val_res_acc = correct_results / total_val_samples if total_val_samples > 0 else 0 
            val_loss /= total_val_samples

            val_losses.append(val_loss)
            val_move_accs.append(val_move_acc)
            val_res_accs.append(val_res_acc)

            print(f"Epoch {epoch+1}/{config.num_epochs} [{epoch_time:.2f}s]")
            print(f"  Train Loss: {total_loss:.4f} (Policy: {total_policy_loss:.4f}, Value: {total_value_loss:.4f})")
            print(f"  Val Loss:   {val_loss:.4f} | Val Move Acc: {val_move_acc*100:.2f}% | Val Result Acc: {val_res_acc*100:.2f}%")

            # Consider the best model as the one with the best mean acc
            if (val_move_acc + val_res_acc)/2 > (best_move_acc + best_result_acc)/2:
                best_move_acc = val_move_acc
                best_result_acc = val_res_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), "best_model.pth")

            # Early stopping based on validation loss
            if config.patience > 0:
                if epoch == 0:
                    prev_validation_loss = val_loss
                elif val_loss > prev_validation_loss:
                    patience_count += 1
                    prev_validation_loss = val_loss
                    if patience_count >= config.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                else:
                    patience_count = 0
                    prev_validation_loss = val_loss
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user (Ctrl+C). Processing progress up to this point...")

    print(f"Best mean accuracy: {(best_move_acc + best_result_acc)/2*100:.2f}% achieved at epoch {best_epoch}")
    print(f"Best move accuracy: {best_move_acc*100:.2f}%")
    print(f"Best result accuracy: {best_result_acc*100:.2f}%")

    return train_losses, val_losses, val_move_accs, val_res_accs, model


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


def validation_test(model, val_loader, device="cuda"):
    """Tests the model on the validation set and prints the accuracy for moves and results."""
    model = model.to(device)
    model.eval()
    
    correct_moves = 0
    correct_results = 0
    total_val_samples = 0

    value_size = 1
    for layer in model.heads.value:
        if isinstance(layer, nn.Linear):
            value_size = layer.out_features

    with torch.no_grad():
        for features, moves, results, scores, masks in val_loader:
            features = features.to(device)
            moves = moves.to(device)
            results = results.to(device)
            masks = masks.to(device)
            
            policy_logits, value_result = model(features)
            policy_logits = policy_logits.masked_fill(~masks, -1e9)
            
            # policy results
            _, predicted_moves = torch.max(policy_logits, 1)
            correct_moves += (predicted_moves == moves).sum().item()
            total_val_samples += moves.size(0)
            
            # value results
            predicted_results = torch.round(value_result.squeeze(-1))
            correct_results += (predicted_results == results).sum().item()

    print("\n\nValidation test results:\n")
    print("\tTotal samples: ", total_val_samples)
    print("\tMove Accuracy: ", correct_moves / total_val_samples)
    print("\tResult Accuracy: ", correct_results / total_val_samples)


def estimate_training_time(model, train_loader, val_loader, config: TrainingConfig, num_warmup: int = 5, num_timed: int = 10):
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


if __name__ == '__main__':

    # Overwrite data path if provided in the command line
    path = None
    if len(sys.argv) > 1:
       path = sys.argv[1]
       d_k = int(sys.argv[2])

    # Initialize train configurations
    train_config = TrainingConfig(
        data_path=path,
        use_cache=True,
        batch_size=512,
        train_ratio=0.98,
        num_workers=12,
        num_epochs=3,
        patience=4,
        lr=2e-3,
        weight_decay=2e-5,
    )
    print(train_config)

    # Initialize model config
    encoder_config = EncoderConfig(
        embed_dim=d_k, 
        num_heads=4,
        num_blocks=1,
        batch_size=train_config.batch_size,
        policy_size=704,
        mlp_expand_factor=1,
    )
    print(encoder_config)

    # Load dataset using MinichessTransformerDataset
    dataset = MinichessTransformerDataset(
        train_config.data_path, 
        promotions=train_config.promotions, 
        use_cache=train_config.use_cache,
    )

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        dataset, 
        batch_size=train_config.batch_size, 
        train_ratio=train_config.train_ratio, 
        num_workers=train_config.num_workers,
    )

    # Instantiate model
    model = MiniChessTransformerEncoder(encoder_config)
    count_params(model)
    #model = torch.compile(model) # JIT for optimized triton kernels

    # Estimate training time before beginning full training
    estimate_training_time(model, train_loader, val_loader, train_config)

    # Run the training loop
    train_losses, val_losses, val_move_accs, val_res_accs, model = train_model(
        model, train_loader, val_loader, train_config
    )

    # Optional loss plotting
    plot_loss(train_losses, val_losses, val_move_accs, val_res_accs)
    
    # Use best model for final validation test validation
    best_model_path = "best_model.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=train_config.device))
        validation_test(model, val_loader, device=train_config.device)

    '''
>> Loading cached dataset from data/gardner_depth2/d2_with_promotions.txt.transformer.pt...

Total number of trainable parameters: 3,505,897
        In bits: 112,188,704 bits
        In bytes: 14,023,588 bytes
        In kilobytes: 13,694.91015625 KB
        In megabytes: 13.37393569946289 MB


=== Estimating Training Time ===
Running 5 warm-up training steps...
Running 10 timed training steps...
Running 5 warm-up validation steps...
Running 10 timed validation steps...

------------------------------------------------
Estimated training stats (Batch size 512):
  Avg single training step:   50.50 ms
  Avg single validation step: 13.17 ms
  Steps per epoch (Train):    19141
  Steps per epoch (Val):      391
  Estimated epoch train time: 966.63 s
  Estimated epoch val time:   5.15 s
  Estimated TOTAL epoch time: 971.78 s
  Estimated TOTAL train time: 9717.82 s (161.96 minutes)
------------------------------------------------

num decayed parameter tensors: 24, with 3,493,120 parameters
num non-decayed parameter tensors: 30, with 12,777 parameters
using fused AdamW: True
Using device: cuda
features (flat_state):  torch.Size([512, 27])
results:  torch.Size([512])
value_result:  torch.Size([512, 1])
moves:  torch.Size([512])
policy_logits:  torch.Size([512, 704])


Epoch 1/10 [978.16s]
  Train Loss: 1.7822 (Policy: 1.5202, Value: 0.2620)
  Val Loss:   1.5672 | Val Move Acc: 49.02% | Val Result Acc: 74.37%
Epoch 2/10 [969.11s]
  Train Loss: 1.5779 (Policy: 1.3393, Value: 0.2386)
  Val Loss:   1.4790 | Val Move Acc: 51.78% | Val Result Acc: 75.63%
Epoch 3/10 [970.59s]
  Train Loss: 1.5040 (Policy: 1.2728, Value: 0.2312)
  Val Loss:   1.4255 | Val Move Acc: 53.71% | Val Result Acc: 75.96%
Epoch 4/10 [986.89s]
  Train Loss: 1.4596 (Policy: 1.2330, Value: 0.2266)
  Val Loss:   1.3937 | Val Move Acc: 54.72% | Val Result Acc: 76.40%
Epoch 5/10 [984.69s]
  Train Loss: 1.4279 (Policy: 1.2048, Value: 0.2231)
  Val Loss:   1.3717 | Val Move Acc: 55.39% | Val Result Acc: 76.43%
Epoch 6/10 [1010.49s]
  Train Loss: 1.4044 (Policy: 1.1841, Value: 0.2203)
  Val Loss:   1.3531 | Val Move Acc: 55.94% | Val Result Acc: 76.89%

python3 src/models/train_transformer.py data/gardner_depth2/d2_with_promotions.txt 64 | tee trnsf_64.log; python3 src/models/train_transformer.py data/gardner_depth2/d2_with_promotions.txt 128 | tee trnsf_128.log; python3 src/models/train_transformer.py data/gardner_depth2/d2_with_promotions.txt 256 | tee trnsf_256.log; python3 src/models/train_transformer.py data/gardner_depth2/d2_with_promotions.txt 512 | tee trnsf_512.log; python3 src/models/train_transformer.py data/gardner_depth2/d2_with_promotions.txt 1024 | tee trnsf_1024.log;        
    '''