import os
import sys
import json
import time
import optuna
import torch
import torch.nn as nn
from datetime import datetime

from src.training.config import TrainingConfig
from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig
from src.training.train_transformer import train_model, get_dataloaders
from src.models.dataloaders import MinichessTransformerDataset

# Suppress excessive optuna logs to keep output clean, but show info
optuna.logging.set_verbosity(optuna.logging.INFO)

# Global configurations
DATASET_PATH = "data/d4/d4_val.txt"
SUBSAMPLE_RATIO = 0.8 # leave 20% of data out for speed
EPOCHS = 15
BATCH_SIZE = 8192
NUM_WORKERS = 12
TRIALS_PER_CONFIG = 35
TUNING_DIR = "logs/tuning"

# depends on what speed we want vs precision...
torch.set_float32_matmul_precision('high')

# Configurations to optimize
CONFIGS_TO_TUNE = [
    {"d_k": 64, "depth": 3},
    {"d_k": 128, "depth": 5},
    # {"d_k": 256, "depth": 4},
    # {"d_k": 256, "depth": 8}, 
]

class TuningObjective:
    """
    Objetivo de optimización de Optuna modularizado como una clase ejecutable.
    Permite encapsular parámetros dinámicos sin recurrir a variables globales.
    """
    def __init__(self, config_dict, study_name, train_loader, val_loader):
        self.config_dict = config_dict
        self.study_name = study_name
        self.train_loader = train_loader
        self.val_loader = val_loader

    def __call__(self, trial):
        # Extraemos los parámetros de configuración. Si no existen, fallará inmediatamente.
        d_k = self.config_dict["d_k"]
        depth = self.config_dict["depth"]

        # IMPORTANT!!
        lr = trial.suggest_float("lr", 4e-5, 1e-2, log=True)
        beta1 = trial.suggest_float("beta1", 0.85, 0.95)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1)
        eps = trial.suggest_float("eps", 1e-9, 1e-7)
            
        # log folder for TensorBoard tracing
        run_name = f"tuning/{self.study_name}_trial{trial.number}_lr{lr:.6f}_beta1{beta1:.4f}_wd{weight_decay:.2e}"
        
        train_config = TrainingConfig(
            data_path=DATASET_PATH,
            use_cache=True,
            batch_size=BATCH_SIZE,
            train_ratio=0.97,
            num_workers=NUM_WORKERS,
            num_epochs=EPOCHS,
            patience=4,

            weight_decay=weight_decay,
            lr=lr,
            beta1=beta1,
            #eps=eps,
            
            custom_init=False,  # Enable custom init for training stability at batch size 4096
            run_name=run_name,
            subsample_ratio=SUBSAMPLE_RATIO
        )
        
        encoder_config = EncoderConfig(
            embed_dim=d_k,
            num_heads=8,
            num_blocks=depth,
            batch_size=BATCH_SIZE,
            policy_size=704,
            mlp_expand_factor=4,
            custom_init=True,
            attn_backend="math",
            autocast_mode="none",
        )
        
        model = MiniChessTransformerEncoder(encoder_config)
        model = torch.compile(model)
 
        print(f"\n  --> Trial {trial.number}: Testing LR = {lr:.6e}, Beta1 = {beta1:.4f}, Weight Decay = {weight_decay:.2e}")
        start_time = time.time()
        
        try:
            train_losses, val_losses, val_move_accs, val_res_accs, _ = train_model(
                model, self.train_loader, self.val_loader, train_config, encoder_config
            )
            duration = time.time() - start_time
            
            # We optimize the best move accuracy achieved during the 5 epochs
            best_move_acc = max(val_move_accs) if val_move_accs else 0.0
            best_val_acc = max(val_res_accs) if val_res_accs else 0.0
            best_mean_acc = (best_move_acc + best_val_acc) / 2
            print(f"  --> Trial {trial.number} Finished. Best Mean Acc: {best_mean_acc:.4f} in {duration:.1f}s")
            return best_mean_acc
            
        except Exception as e:
            import traceback
            print(f"  ⚠️[TRIAL FAILED]")
            print(f"  ⚠️[TRIAL FAILED] Exception raised: {e}")
            print(traceback.format_exc())
            print(f"  ⚠️[TRIAL FAILED]")
            return 0.0 # Return 0.0 on failure to penalize the hyperparameter choice

def summary_table(summary_results):
    print("\n" + "=" * 70)
    print("ALL OPTUNA STUDIES COMPLETE. SUMMARY:")
    print("=" * 70)
    for res in summary_results:
        print(f"  {res['study']:<30} | Best Acc: {res['best_acc']:.4f} | Best LR: {res['best_lr']:.6e} | Best Beta1: {res['best_beta1']:.4f} | Best Eps: {res['best_eps']:.2e}")
    print("=" * 70)

def main():
    os.makedirs(TUNING_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print(f"MINICHESS HYPERPARAMETER TUNING (OPTUNA) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {DATASET_PATH} (Subsample: {SUBSAMPLE_RATIO})")
    print(f"Fixed Batch Size: {BATCH_SIZE} | Epochs per Trial: {EPOCHS}")
    print("=" * 70)

    print(">> Loading dataset and caching features...")
    dataset = MinichessTransformerDataset(
        DATASET_PATH, 
        promotions=True, 
        use_cache=True,
        subsample_ratio=SUBSAMPLE_RATIO
    )
    
    print(">> Preparing DataLoader...")
    train_loader, val_loader = get_dataloaders(
        dataset, 
        batch_size=BATCH_SIZE, 
        train_ratio=0.98, 
        num_workers=NUM_WORKERS
    )
    
    summary_results = []
    for i, cfg in enumerate(CONFIGS_TO_TUNE):
        # these have to be extracted from our custom optim config dict
        d_k = cfg["d_k"]
        depth = cfg["depth"]
        
        study_name = f"tuning_dk{d_k}_depth{depth}"
        db_path = os.path.join(TUNING_DIR, f"{study_name}.db")
        
        print(f"\n[{i+1}/{len(CONFIGS_TO_TUNE)}] Starting Study: {study_name}")
        print(f"  Storage DB: sqlite:///{db_path}")
        
        # SQLite storage allows checkpointing and resuming interrupted studies
        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}",
            direction="maximize",
            load_if_exists=True
        )
        
        objective = TuningObjective(cfg, study_name, train_loader, val_loader)
        study.optimize(objective, n_trials=TRIALS_PER_CONFIG, show_progress_bar=True)
        
        best_trial = study.best_trial
        print(f"\n=== STUDY {study_name} COMPLETE ===")
        print(f"  Best Val Move Accuracy: {best_trial.value:.5f}")
        print(f"  Best Learning Rate: {best_trial.params['lr']:.5e}")
        print(f"  Best Beta1: {best_trial.params['beta1']:.5f}")
        print(f"  Best Weight Decay: {best_trial.params['weight_decay']:.5e}")
        
        # Save a summary file for this specific configuration study
        summary_path = os.path.join(TUNING_DIR, f"{study_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "d_k": d_k,
                "depth": depth,
                "best_move_accuracy": best_trial.value,
                "best_lr": best_trial.params["lr"],
                "best_beta1": best_trial.params["beta1"],
                "best_weight_decay": best_trial.params["weight_decay"],
                "best_trial_number": best_trial.number
            }, f, indent=4)
            
        summary_results.append({
            "study": study_name,
            "best_acc": best_trial.value,
            "best_lr": best_trial.params["lr"],
            "best_beta1": best_trial.params["beta1"],
            "best_weight_decay": best_trial.params["weight_decay"]
        })
    summary_table(summary_results)

if __name__ == "__main__":
    main()
