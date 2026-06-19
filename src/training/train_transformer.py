import time
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.utils import time_this, count_params, set_seed
from src.models.dataloaders import get_dataloaders, MinichessTransformerDataset
from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig
from src.training.config import TrainingConfig, parse_args
from src.training.utils import (
    configure_optimizers,
    configure_profiler,
    estimate_training_time,
    plot_loss,
    decode_move_indices
)
from src.training.logger import TensorBoardLogger, generate_run_name, save_run_metadata

@time_this
def train_model(
    model,
    train_loader, 
    val_loader,
    config: TrainingConfig,
    encoder_config: EncoderConfig = None,
):
    # Always normalize the run name to ensure uniqueness and descriptive folder names
    if encoder_config is not None:
        config.run_name = generate_run_name(config, encoder_config)

    # Setup run directory and TensorBoard writer
    run_dir = None
    if config.run_name:
        run_dir = os.path.join(config.save_dir, config.run_name) if config.save_dir else config.run_name
        save_run_metadata(run_dir, config, encoder_config, config.profile_desc or f"Experiment run: {config.run_name}")
    elif config.profile_name is not None:
        run_dir = f"./profiles/{config.profile_name}"
        save_run_metadata(run_dir, config, encoder_config, config.profile_desc)

    # only log if a run name has been set or
    tb_logger = TensorBoardLogger(run_dir if config.run_name else None)

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
        device_type=config.device,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps
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
            
            ###########################################
            ##            Training phase             ##
            ###########################################
            model.train()
            total_loss, total_policy_loss, total_value_loss = 0.0, 0.0, 0.0
            total_aux_loss = 0.0

            start_time = time.time()
            epoch_grad_norms = [] # to plot gradients norm (tensorboard debugging mostly)

            step_idx = 0
            for features, moves, results, scores, masks in train_loader:
                features = features.to(config.device)
                moves = moves.to(config.device)
                results = results.to(config.device)
                masks = masks.to(config.device)

                optimizer.zero_grad()

                outputs = model(features)
                # different outputs depending on the model heads
                if len(outputs) == 5:
                    policy_logits, value_result, aux_from, aux_to, aux_promo = outputs
                else:
                    policy_logits, value_result = outputs

                # Apply legal moves masking
                policy_logits = policy_logits.masked_fill(~masks, -1e9)

                policy_loss = policy_criterion(policy_logits, moves)
                value_loss = value_criterion(value_result.squeeze(-1), results.float())

                # the loss for the factored policy is less straightforwad, but basically
                # we just train each of the three factors (from, to, promo) independently
                if len(outputs) == 5:
                    target_from, target_to, target_promo = decode_move_indices(moves, features.device)
                    
                    from_loss = policy_criterion(aux_from, target_from)
                    to_loss = policy_criterion(aux_to, target_to)
                    promo_loss = policy_criterion(aux_promo, target_promo)
                    aux_loss = from_loss + to_loss + promo_loss
                    
                    loss = policy_loss + value_loss + 0.5 * aux_loss
                    total_aux_loss += aux_loss.item() * features.size(0)
                
                else:
                    loss = policy_loss + value_loss

                if debug_flag: # useful for debugging tensor dims
                    print("[DEBUG TENSOR SIZES]:")
                    print("\tfeatures (flat_state): ", features.shape); print("\tresults: ", results.shape)
                    print("\tvalue_result: ", value_result.shape); print("\tmoves: ", moves.shape)
                    print("\tpolicy_logits: ", policy_logits.shape)
                    if len(outputs) == 5:
                        print("\taux_from: ", aux_from.shape); print("\taux_to: ", aux_to.shape); 
                        print("\taux_promo: ", aux_promo.shape)
                    print("\n")
                    debug_flag = False

                loss.backward()

                # Gradient clipping for Transformer block stability
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # TODO hmmmmm maybe increase
                epoch_grad_norms.append(
                    grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                    )

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
            total_aux_loss /= num_samples
            avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0
            train_losses.append((total_loss, total_policy_loss, total_value_loss))

            epoch_time = time.time() - start_time

            ###########################################
            ##           Validation phase            ##
            ###########################################
            model.eval() # for dropout and batchnorm
            val_loss, correct_moves, correct_results, total_val_samples = 0.0, 0, 0, 0
            val_policy_activations, val_value_activations = [], []

            with torch.no_grad():
                for features, moves, results, scores, masks in val_loader:
                    features = features.to(config.device)
                    moves = moves.to(config.device)
                    results = results.to(config.device)
                    masks = masks.to(config.device)

                    outputs = model(features)
                    if len(outputs) == 5:
                        policy_logits, value_result, _, _, _ = outputs
                    else:
                        policy_logits, value_result = outputs
                    
                    # Apply move masking before logging or computing loss
                    policy_logits = policy_logits.masked_fill(~masks, -1e9)
                    
                    # store activations on CPU to avoid GPU memory growth
                    val_policy_activations.append(policy_logits.detach().cpu())
                    val_value_activations.append(value_result.detach().cpu())
                    
                    # get value loss and "correct" results (round)
                    value_loss = value_criterion(value_result.squeeze(-1), results.float())
                    predicted_results = torch.round(value_result.squeeze(-1))
                    correct_results += (predicted_results == results).sum().item()

                    # get policy logits, loss and correct moves for accuracy
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

            # TensorBoard logging of metrics and histograms
            tb_logger.log_epoch(
                epoch=epoch + 1,
                metrics={
                    "Loss/Train": total_loss,
                    "Loss/Train_Policy": total_policy_loss,
                    "Loss/Train_Value": total_value_loss,
                    "Loss/Val": val_loss,
                    "Loss/Train_Aux": total_aux_loss,
                    "Loss/Grad_Norm": avg_grad_norm,
                    "Accuracy/Val_Move": val_move_acc * 100,
                    "Accuracy/Val_Result": val_res_acc * 100,
                    "Accuracy/Val_Mean": val_mean_acc * 100,
                },
                val_policy_activations=val_policy_activations,
                val_value_activations=val_value_activations,
                model=model
            )

            print(f"Epoch {epoch+1}/{config.num_epochs} [{epoch_time:.2f}s]")
            if total_aux_loss > 0:
                print(f"  Train Loss: {total_loss:.4f} (Policy: {total_policy_loss:.4f}, Value: {total_value_loss:.4f}, Aux: {total_aux_loss:.4f})")
            else:
                print(f"  Train Loss: {total_loss:.4f} (Policy: {total_policy_loss:.4f}, Value: {total_value_loss:.4f})")
            print(f"  Val Loss:   {val_loss:.4f} | Val Move Acc: {val_move_acc*100:.2f}% | Val Result Acc: {val_res_acc*100:.2f}%")

            # Consider the best model as the one with the best mean acc
            if  val_mean_acc > (best_move_acc + best_result_acc)/2:
                best_move_acc = val_move_acc
                best_result_acc = val_res_acc
                best_epoch = epoch + 1
                
                # Delete previous metrics checkpoint in this run directory to keep it clean
                if hasattr(train_model, 'last_saved_checkpoint') and train_model.last_saved_checkpoint:
                    try:
                        if os.path.exists(train_model.last_saved_checkpoint):
                            os.remove(train_model.last_saved_checkpoint)
                    except Exception:
                        pass
                
                # Save standard best model path
                model_save_path = os.path.join(run_dir, "best_model.pth") if run_dir else "best_model.pth"
                torch.save(model.state_dict(), model_save_path)
                
                # Save a copy with metrics in the filename for easy reference
                metrics_name = f"best_model_epoch{epoch+1}_move{val_move_acc*100:.2f}_res{val_res_acc*100:.2f}.pth"
                metrics_save_path = os.path.join(run_dir, metrics_name) if run_dir else metrics_name
                torch.save(model.state_dict(), metrics_save_path)
                train_model.last_saved_checkpoint = metrics_save_path

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

    tb_logger.close()

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
            
            outputs = model(features)
            if len(outputs) == 5:
                policy_logits, value_result, _, _, _ = outputs
            else:
                policy_logits, value_result = outputs
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


def test_model_holdout(model, train_config):
    """Loads the holdout test set (if it exists) and evaluates the model on it."""
    basename = os.path.basename(train_config.data_path)
    prefix = basename.replace("_val.txt", "").replace(".txt", "")
    test_path = os.path.join("data", "test_splits", f"{prefix}_test.txt")
    
    if not os.path.exists(test_path):
        print(f"\n>> Holdout test file not found at: {test_path}")
        return
        
    print(f"\n>> Loading dedicated holdout test dataset from: {test_path}...")
    from torch.utils.data import DataLoader
    test_dataset = MinichessTransformerDataset(
        test_path, 
        promotions=train_config.promotions, 
        use_cache=True,
        representation=train_config.representation,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    print(">> Evaluating final model on Holdout Test Set...")
    print("\nHoldout Test Set Results:")
    validation_test(model, test_loader, device=train_config.device)


if __name__ == '__main__':
    args = parse_args()

    # global configs that affect whole training
    set_seed(args.seed) # defaults to 42
    torch.set_float32_matmul_precision(args.precision)

    # Initialize train configurations
    train_config = TrainingConfig(
        data_path=args.data_path,
        use_cache=args.use_cache,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        num_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay, # fuck, i was running ablations with this hardcoded
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        custom_init=args.custom_init,
        run_name=args.run_name,
        save_dir=args.save_dir,
        profile_name=args.profile,
        profile_steps=args.profile_steps,
        profile_desc=args.profile_desc,
        profile_filename=args.profile_filename,
        subsample_ratio=args.subsample,
        representation=args.representation,
        use_factorized_policy=args.factorized_policy,
    )
    print(train_config)

    # Initialize model config
    encoder_config = EncoderConfig(
        embed_dim=args.embed_dim, 
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        batch_size=train_config.batch_size,
        policy_size=704, # this stays fixed. rare would be the case in which size is different
        mlp_expand_factor=args.mlp_expand,
        custom_init=train_config.custom_init,
        
        # careful with these, bad combinations can break things silently! 
        attn_backend=args.attn_backend,
        autocast_mode=args.autocast,
        representation=train_config.representation,
        use_factorized_policy=train_config.use_factorized_policy,
    )
    
    print(encoder_config)

    # Load dataset using MinichessTransformerDataset
    dataset = MinichessTransformerDataset(
        train_config.data_path, 
        promotions=train_config.promotions, 
        use_cache=train_config.use_cache,
        subsample_ratio=train_config.subsample_ratio,
        representation=train_config.representation,
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
    estimate_training_time(model, train_loader, val_loader, train_config)

    # Run the training loop
    train_losses, val_losses, val_move_accs, val_res_accs, model = train_model(
        model, train_loader, val_loader, train_config, encoder_config, time=True
    )

    # Optional loss plotting
    #run_dir = f"{train_config.save_dir}/{train_config.run_name}" if train_config.run_name else None
    # plot_loss(train_losses, val_losses, val_move_accs, val_res_accs, save_dir=run_dir)
    
    # Use best model for final validation/test validation
    # best_model_path = os.path.join(run_dir, "best_model.pth") if run_dir else "best_model.pth"
    # if os.path.exists(best_model_path):
    #     model.load_state_dict(torch.load(best_model_path, map_location=train_config.device))
    #     validation_test(model, val_loader, device=train_config.device)
    #     test_model_holdout(model, train_config)

    print("\n"*3, "/\\"*40, "\n"*3) # this is just for pretty printing