import os
import sys
import json
import argparse
import re
import torch
from torch.utils.data import DataLoader

# Ensure PYTHONPATH includes the current directory
sys.path.append(os.getcwd())

from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig
from src.models.dataloaders import MinichessTransformerDataset

# Config
ABLATION_ROOT = "results/ablations/ablations_dk64_n3"
TEST_FILE = "data/test_splits/d4_test.txt"
BATCH_SIZE = 16384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_agent_dir(path):
    dirname = os.path.basename(path)
    config = {
        "embed_dim": 64,
        "num_blocks": 3,
        "representation": "simple",
        "factorized_policy": False
    }
    
    if "simple" in dirname:
        config["representation"] = "simple"
    elif "spatial" in dirname:
        config["representation"] = "spatial"
        
    if "nofact" in dirname:
        config["factorized_policy"] = False
    elif "fact" in dirname:
        config["factorized_policy"] = True
        
    # Parse embed_dim from folder name (e.g. dk64, dk128)
    dk_match = re.search(r"dk(\d+)", dirname)
    if dk_match:
        config["embed_dim"] = int(dk_match.group(1))
        
    # Parse num_blocks from folder name (e.g. depth3, depth4, depth5)
    depth_match = re.search(r"depth(\d+)", dirname)
    if depth_match:
        config["num_blocks"] = int(depth_match.group(1))
        
    return config

def evaluate_checkpoint(model_path, config, dataset_simple, dataset_spatial):
    # Select dataset depending on representation
    dataset = dataset_spatial if config["representation"] == "spatial" else dataset_simple
    
    # Instantiate DataLoader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    # Configure model
    encoder_config = EncoderConfig(
        embed_dim=config["embed_dim"],
        num_heads=8,
        num_blocks=config["num_blocks"],
        batch_size=BATCH_SIZE,
        policy_size=704,
        mlp_expand_factor=4,
        representation=config["representation"],
        use_factorized_policy=config["factorized_policy"],
        attn_backend="math",
        autocast_mode="none"
    )
    
    model = MiniChessTransformerEncoder(encoder_config).to(DEVICE)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=DEVICE)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[10:]] = v
        else:
            clean_state_dict[k] = v
    model.load_state_dict(clean_state_dict)
    model.eval()
    
    total_samples = 0
    correct_moves_top1 = 0
    correct_moves_top3 = 0
    correct_moves_top5 = 0
    mrr_sum = 0.0
    
    correct_results = 0
    val_mse_sum = 0.0
    val_mae_sum = 0.0
    
    with torch.no_grad():
        for features, moves, results, scores, masks in loader:
            features = features.to(DEVICE)
            moves = moves.to(DEVICE)
            results = results.to(DEVICE).float()
            masks = masks.to(DEVICE)
            
            outputs = model(features)
            if len(outputs) == 5:
                policy_logits, value_result, _, _, _ = outputs
            else:
                policy_logits, value_result = outputs
                
            policy_logits = policy_logits.masked_fill(~masks, -1e9)
            
            # Policy Metrics
            # Sort logits descending
            sorted_logits, sorted_indices = torch.sort(policy_logits, dim=1, descending=True)
            
            # Precision@1 (Top-1 Move Accuracy)
            pred_top1 = sorted_indices[:, 0]
            correct_moves_top1 += (pred_top1 == moves).sum().item()
            
            # Precision@3
            pred_top3 = sorted_indices[:, :3]
            correct_moves_top3 += (pred_top3 == moves.unsqueeze(-1)).any(dim=1).sum().item()
            
            # Precision@5
            pred_top5 = sorted_indices[:, :5]
            correct_moves_top5 += (pred_top5 == moves.unsqueeze(-1)).any(dim=1).sum().item()
            
            # MRR (Mean Reciprocal Rank)
            # Find rank of target moves (1-based index in sorted_indices)
            ranks = (sorted_indices == moves.unsqueeze(-1)).nonzero()[:, 1] + 1
            mrr_sum += (1.0 / ranks.float()).sum().item()
            
            # Value Metrics
            value_result = value_result.squeeze(-1)
            predicted_results = torch.round(value_result)
            correct_results += (predicted_results == results).sum().item()
            
            val_mse_sum += ((value_result - results) ** 2).sum().item()
            val_mae_sum += (torch.abs(value_result - results)).sum().item()
            
            total_samples += moves.size(0)
            
    return {
        "move_accuracy": correct_moves_top1 / total_samples,
        "precision_at_3": correct_moves_top3 / total_samples,
        "precision_at_5": correct_moves_top5 / total_samples,
        "mrr": mrr_sum / total_samples,
        "result_accuracy": correct_results / total_samples,
        "value_mse": val_mse_sum / total_samples,
        "value_mae": val_mae_sum / total_samples,
        "total_samples": total_samples
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in an ablation root")
    parser.add_argument(
        "--ablation_root",
        type=str,
        default="results/ablations/ablations_dk64_n3",
        help="Ablation root directory to evaluate (default: results/ablations/ablations_dk64_n3)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ablations",
        help="Output directory to save evaluation results JSON (default: results/ablations)"
    )
    args = parser.parse_args()

    print(f"[*] Loading test dataset simple representation from {TEST_FILE}...")
    dataset_simple = MinichessTransformerDataset(TEST_FILE, promotions=True, use_cache=True, representation="simple")
    
    print(f"[*] Loading test dataset spatial representation from {TEST_FILE}...")
    dataset_spatial = MinichessTransformerDataset(TEST_FILE, promotions=True, use_cache=True, representation="spatial")
    
    seeds = sorted(os.listdir(args.ablation_root))
    
    results = {}
    
    for seed in seeds:
        seed_path = os.path.join(args.ablation_root, seed)
        if not os.path.isdir(seed_path):
            continue
            
        print(f"\n>>> Evaluating Seed: {seed}")
        results[seed] = {}
        
        runs = sorted(os.listdir(seed_path))
        for run in runs:
            run_path = os.path.join(seed_path, run)
            if not os.path.isdir(run_path):
                continue
                
            model_path = os.path.join(run_path, "best_model.pth")
            if not os.path.exists(model_path):
                continue
                
            config = parse_agent_dir(run_path)
            
            # Determine standard name
            parts = run.split("_")
            repr_str = "simple" if "simple" in parts else "spatial"
            fact_str = "fact" if "fact" in parts else "nofact"
            name = f"{repr_str}_{fact_str}"
            
            print(f"  Evaluating Model: {name} (folder: {run})")
            metrics = evaluate_checkpoint(model_path, config, dataset_simple, dataset_spatial)
            results[seed][name] = metrics
            
            print(f"    - Move Acc (P@1): {metrics['move_accuracy']*100:.2f}% | P@3: {metrics['precision_at_3']*100:.2f}% | P@5: {metrics['precision_at_5']*100:.2f}%")
            print(f"    - MRR: {metrics['mrr']:.4f}")
            print(f"    - Result Acc: {metrics['result_accuracy']*100:.2f}% | MSE: {metrics['value_mse']:.4f} | MAE: {metrics['value_mae']:.4f}")
            
    # Save results to json
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "test_evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] All test evaluation results saved to {out_path}")

if __name__ == "__main__":
    main()
