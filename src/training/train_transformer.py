import time
import os
import sys
import torch
import torch.nn as nn
from src.utils.utils import time_this, count_params
from src.models.dataloaders import get_dataloaders, MinichessTransformerDataset
from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig
from src.training.config import TrainingConfig, parse_args
from src.training.utils import (
    configure_optimizers,
    save_run_metadata,
    configure_profiler,
    estimate_training_time,
    plot_loss
)

@time_this
def train_model(
    model,
    train_loader, 
    val_loader,
    config: TrainingConfig,
    encoder_config: EncoderConfig = None,
):
    # Setup run directory and TensorBoard writer
    run_dir = None
    writer = None
    if config.run_name:
        run_dir = f"logs/exps/{config.run_name}"
        save_run_metadata(run_dir, config, encoder_config, config.profile_desc or f"Experiment run: {config.run_name}")
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=run_dir)
        except Exception as e:
            print(f"[WARNING] Could not initialize TensorBoard writer: {e}")
    elif config.profile_name is not None:
        run_dir = f"./log/{config.profile_name}"
        save_run_metadata(run_dir, config, encoder_config, config.profile_desc)

    profile_training = config.profile_name is not None
    prof = None
    if profile_training:
        prof = configure_profiler(config, config.profile_name, trace_filename=config.profile_filename)
        prof.start()

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

            step_idx = 0
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

                if prof is not None:
                    prof.step()
                    step_idx += 1
                    if step_idx >= config.profile_steps:
                        prof.stop()
                        print(f"\n[INFO] Profiling complete. Trace saved to './log/{config.profile_name}'. Exiting early.")
                        sys.exit(0)

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

                    # get policy logits, loss and correct moves for accuracy
                    policy_logits = policy_logits.masked_fill(~masks, -1e9)
                    policy_loss = policy_criterion(policy_logits, moves)
                    _, predicted_moves = torch.max(policy_logits, 1)
                    correct_moves += (predicted_moves == moves).sum().item()

                    # overall validation loss
                    val_loss += (policy_loss + value_loss).item() * features.size(0)
                    total_val_samples += moves.size(0)

            val_move_acc = correct_moves / total_val_samples if total_val_samples > 0 else 0 
            val_res_acc = correct_results / total_val_samples if total_val_samples > 0 else 0 
            val_mean_acc = ((val_move_acc + val_res_acc) / 2) if total_val_samples > 0 else 0
            val_loss /= total_val_samples

            val_losses.append(val_loss)
            val_move_accs.append(val_move_acc)
            val_res_accs.append(val_res_acc)

            # TensorBoard logging of scalar metrics
            if writer is not None:
                writer.add_scalar("Loss/Train", total_loss, epoch + 1)
                writer.add_scalar("Loss/Train_Policy", total_policy_loss, epoch + 1)
                writer.add_scalar("Loss/Train_Value", total_value_loss, epoch + 1)
                writer.add_scalar("Loss/Val", val_loss, epoch + 1)
                writer.add_scalar("Accuracy/Val_Move", val_move_acc * 100, epoch + 1)
                writer.add_scalar("Accuracy/Val_Result", val_res_acc * 100, epoch + 1)
                writer.add_scalar("Accuracy/Val_Mean", val_mean_acc * 100, epoch + 1)

            print(f"Epoch {epoch+1}/{config.num_epochs} [{epoch_time:.2f}s]")
            print(f"  Train Loss: {total_loss:.4f} (Policy: {total_policy_loss:.4f}, Value: {total_value_loss:.4f})")
            print(f"  Val Loss:   {val_loss:.4f} | Val Move Acc: {val_move_acc*100:.2f}% | Val Result Acc: {val_res_acc*100:.2f}%")

            # Consider the best model as the one with the best mean acc
            if  val_mean_acc > (best_move_acc + best_result_acc)/2:
                best_move_acc = val_move_acc
                best_result_acc = val_res_acc
                best_epoch = epoch + 1
                model_save_path = os.path.join(run_dir, "best_model.pth") if run_dir else "best_model.pth"
                torch.save(model.state_dict(), model_save_path)

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

    if writer is not None:
        writer.close()

    return train_losses, val_losses, val_move_accs, val_res_accs, model

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


if __name__ == '__main__':
    args = parse_args()
    d_k = args.embed_dim

    # Initialize train configurations
    train_config = TrainingConfig(
        data_path=args.data_path,
        use_cache=True,
        batch_size=args.batch_size,
        train_ratio=0.98,
        num_workers=12,
        num_epochs=args.epochs,
        patience=4,
        lr=args.lr,
        weight_decay=2e-5,
        custom_init=args.custom_init,
        run_name=args.run_name,
        profile_name=args.profile,
        profile_steps=args.profile_steps,
        profile_desc=args.profile_desc,
        profile_filename=args.profile_filename,
        subsample_ratio=args.subsample,
    )
    print(train_config)

    # Initialize model config
    encoder_config = EncoderConfig(
        embed_dim=d_k, 
        num_heads=8,
        num_blocks=args.num_blocks,
        batch_size=train_config.batch_size,
        policy_size=704,
        mlp_expand_factor=args.mlp_expand,
        custom_init=train_config.custom_init,
    )
    print(encoder_config)

    # Load dataset using MinichessTransformerDataset
    dataset = MinichessTransformerDataset(
        train_config.data_path, 
        promotions=train_config.promotions, 
        use_cache=train_config.use_cache,
        subsample_ratio=train_config.subsample_ratio,
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
    model = torch.compile(model) # JIT for optimized triton kernels

    # Estimate training time before beginning full training
    # estimate_training_time(model, train_loader, val_loader, train_config)

    # Run the training loop
    train_losses, val_losses, val_move_accs, val_res_accs, model = train_model(
        model, train_loader, val_loader, train_config, encoder_config
    )

    # Optional loss plotting
    run_dir = f"logs/exps/{train_config.run_name}" if train_config.run_name else None
    plot_loss(train_losses, val_losses, val_move_accs, val_res_accs, save_dir=run_dir)
    
    # Use best model for final validation test validation
    # best_model_path = os.path.join(run_dir, "best_model.pth") if run_dir else "best_model.pth"
    # if os.path.exists(best_model_path):
    #     model.load_state_dict(torch.load(best_model_path, map_location=train_config.device))
    #     validation_test(model, val_loader, device=train_config.device)

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