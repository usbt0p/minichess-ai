import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import re
import shutil
from tensorboard.backend.event_processing import event_accumulator

# Set up matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

RESULTS_JSON = "results/ablations/ablations_dk64_n3/test_evaluation_results.json"
ABLATION_ROOT = "results/ablations/ablations_dk64_n3"
OUTPUT_DIR = "results/ablations"

def parse_log_file(log_path):
    epochs = []
    train_losses = []
    train_policy_losses = []
    train_value_losses = []
    train_aux_losses = []
    val_losses = []
    val_move_accs = []
    val_res_accs = []
    
    if not os.path.exists(log_path):
        return None
        
    with open(log_path, 'r') as f:
        content = f.read()
        
    lines = content.split('\n')
    current_epoch = None
    for line in lines:
        epoch_match = re.search(r"Epoch (\d+)/(\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            
        train_match = re.search(r"Train Loss:\s+([0-9.]+)\s+\(Policy:\s+([0-9.]+),\s+Value:\s+([0-9.]+)(?:,\s+Aux:\s+([0-9.]+))?\)", line)
        if train_match and current_epoch is not None:
            train_loss = float(train_match.group(1))
            train_policy_loss = float(train_match.group(2))
            train_value_loss = float(train_match.group(3))
            train_aux_loss = float(train_match.group(4)) if train_match.group(4) is not None else 0.0
            
        val_match = re.search(r"Val Loss:\s+([0-9.]+)\s+\|\s+Val Move Acc:\s+([0-9.]+)%\s+\|\s+Val Result Acc:\s+([0-9.]+)%", line)
        if val_match and current_epoch is not None:
            val_loss = float(val_match.group(1))
            val_move_acc = float(val_match.group(2)) / 100.0
            val_res_acc = float(val_match.group(3)) / 100.0
            
            epochs.append(current_epoch)
            train_losses.append(train_loss)
            train_policy_losses.append(train_policy_loss)
            train_value_losses.append(train_value_loss)
            train_aux_losses.append(train_aux_loss)
            val_losses.append(val_loss)
            val_move_accs.append(val_move_acc)
            val_res_accs.append(val_res_acc)
            
    return {
        "epochs": epochs,
        "train_loss": train_losses,
        "train_policy_loss": train_policy_losses,
        "train_value_loss": train_value_losses,
        "train_aux_loss": train_aux_losses,
        "val_loss": val_losses,
        "val_move_acc": val_move_accs,
        "val_res_acc": val_res_accs
    }

def process_learning_curves(seeds, model_names, log_mapping):
    # Max epochs is 30
    num_epochs = 30
    history = {model: {
        "train_loss": [], "train_policy_loss": [], "train_value_loss": [], "train_aux_loss": [],
        "val_loss": [], "val_move_acc": [], "val_res_acc": []
    } for model in model_names}
    
    for seed in seeds:
        seed_dir = os.path.join(ABLATION_ROOT, seed)
        for model in model_names:
            log_file = os.path.join(seed_dir, log_mapping[model])
            data = parse_log_file(log_file)
            
            if data is None or len(data["epochs"]) == 0:
                print(f"[WARNING] Log file not found or empty: {log_file}")
                continue
                
            train_loss = data["train_loss"]
            train_policy_loss = data["train_policy_loss"]
            train_value_loss = data["train_value_loss"]
            train_aux_loss = data["train_aux_loss"]
            val_loss = data["val_loss"]
            val_move_acc = data["val_move_acc"]
            val_res_acc = data["val_res_acc"]
            
            while len(train_loss) < num_epochs:
                train_loss.append(train_loss[-1])
                train_policy_loss.append(train_policy_loss[-1])
                train_value_loss.append(train_value_loss[-1])
                train_aux_loss.append(train_aux_loss[-1])
                val_loss.append(val_loss[-1])
                val_move_acc.append(val_move_acc[-1])
                val_res_acc.append(val_res_acc[-1])
                
            train_loss = train_loss[:num_epochs]
            train_policy_loss = train_policy_loss[:num_epochs]
            train_value_loss = train_value_loss[:num_epochs]
            train_aux_loss = train_aux_loss[:num_epochs]
            val_loss = val_loss[:num_epochs]
            val_move_acc = val_move_acc[:num_epochs]
            val_res_acc = val_res_acc[:num_epochs]
            
            history[model]["train_loss"].append(train_loss)
            history[model]["train_policy_loss"].append(train_policy_loss)
            history[model]["train_value_loss"].append(train_value_loss)
            history[model]["train_aux_loss"].append(train_aux_loss)
            history[model]["val_loss"].append(val_loss)
            history[model]["val_move_acc"].append(val_move_acc)
            history[model]["val_res_acc"].append(val_res_acc)
            
    return history

def plot_curves(history, metric_name, ylabel, title, save_path):
    plt.figure(figsize=(10, 6), dpi=150)
    epochs = np.arange(1, 31)
    
    colors = {
        "simple_nofact": "#1f77b4", # Blue
        "simple_fact": "#2ca02c", # Green
        "spatial_nofact": "#ff7f0e", # Orange
        "spatial_fact": "#d62728", # Red
        "mlp_big": "#9467bd"       # Purple
    }
    
    labels = {
        "simple_nofact": "Simple Flat (Baseline)",
        "simple_fact": "Simple Factored",
        "spatial_nofact": "Spatial 2D",
        "spatial_fact": "Spatial 2D Factored",
        "mlp_big": "MLP Baseline"
    }
    
    for model, metrics in history.items():
        # For auxiliary loss, only plot factorized models
        if metric_name == "train_aux_loss" and model not in ["simple_fact", "spatial_fact"]:
            continue
            
        data = np.array(metrics[metric_name])
        if len(data) == 0:
            continue
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        plt.plot(epochs, mean, label=labels[model], color=colors[model], linewidth=2.0)
        plt.fill_between(epochs, mean - std, mean + std, color=colors[model], alpha=0.15)
        
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.legend(fontsize=10, loc="lower right" if "acc" in metric_name else "upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Shaded line plot saved to {save_path}")

def run_stats_on_metric(results_data, metric_key, metric_display_name):
    print(f"\n=========================================")
    print(f" STATISTICAL TEST FOR {metric_display_name.upper()} ")
    print(f"=========================================")
    
    models = ["simple_nofact", "simple_fact", "spatial_nofact", "spatial_fact"]
    
    model_data = {model: [] for model in models}
    
    for seed, seed_results in results_data.items():
        for model in models:
            model_data[model].append(seed_results[model][metric_key])
            
    # Print mean and std for each model
    print("Descriptive Statistics (Mean ± Std):")
    for model in models:
        mean_val = np.mean(model_data[model])
        std_val = np.std(model_data[model])
        print(f"  {model:15s}: {mean_val*100:.4f}% ± {std_val*100:.4f}%")
        
    # 1. Shapiro-Wilk Test for Normality (grouped)
    all_values = []
    for model in models:
        all_values.extend(model_data[model])
    _, p_norm = stats.shapiro(all_values)
    print(f"\nShapiro-Wilk normality test p-value: {p_norm:.6f}")
    if p_norm > 0.05:
        print("  -> Distribution looks normal (p > 0.05).")
    else:
        print("  -> Distribution does NOT look normal (p <= 0.05).")
        
    # 2. One-way ANOVA
    f_stat, p_val_anova = stats.f_oneway(
        model_data["simple_nofact"],
        model_data["simple_fact"],
        model_data["spatial_nofact"],
        model_data["spatial_fact"]
    )
    print(f"\nOne-way ANOVA: F-statistic = {f_stat:.4f}, p-value = {p_val_anova:.6g}")
    
    # 3. Post-Hoc Tukey HSD
    if p_val_anova < 0.05:
        print("\nANOVA is significant (p < 0.05). Running post-hoc Tukey HSD...")
        flat_data = []
        labels = []
        for model in models:
            flat_data.extend(model_data[model])
            labels.extend([model] * len(model_data[model]))
            
        tukey = pairwise_tukeyhsd(endog=flat_data, groups=labels, alpha=0.05)
        print(tukey)
        
        # Save Tukey table output to a string
        tukey_str = str(tukey)
    else:
        print("\nANOVA is NOT significant (p >= 0.05). No Tukey HSD needed.")
        tukey_str = "ANOVA not significant."
        
    # Generate Boxplot
    plt.figure(figsize=(8, 6), dpi=150)
    box_data = [model_data[model] for model in models]
    labels_display = ["Simple Flat\n(Baseline)", "Simple\nFactored", "Spatial 2D", "Spatial 2D\nFactored"]
    
    # Customize boxplot aesthetics
    box = plt.boxplot(box_data, labels=labels_display, patch_artist=True, widths=0.5)
    
    colors_box = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]
    for patch, color in zip(box['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    for median in box['medians']:
        median.set(color='black', linewidth=2)
        
    plt.ylabel(metric_display_name, fontsize=12)
    plt.title(f"Test Set {metric_display_name} across 10 Seeds", fontsize=14, pad=15)
    plt.tight_layout()
    boxplot_path = os.path.join(OUTPUT_DIR, f"test_{metric_key}_boxplot.png")
    plt.savefig(boxplot_path)
    plt.close()
    print(f"[*] Boxplot saved to {boxplot_path}")
    
    # Generate Violin Plot
    plt.figure(figsize=(8, 6), dpi=150)
    violin = plt.violinplot(box_data, showmedians=True, showextrema=True)
    
    for i, pc in enumerate(violin['bodies']):
        pc.set_facecolor(colors_box[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
        violin[partname].set_edgecolor('black')
        violin[partname].set_linewidth(1.5)
        
    plt.xticks(np.arange(1, len(labels_display) + 1), labels_display)
    plt.ylabel(metric_display_name, fontsize=12)
    plt.title(f"Test Set {metric_display_name} Distribution across 10 Seeds", fontsize=14, pad=15)
    plt.tight_layout()
    violin_path = os.path.join(OUTPUT_DIR, f"test_{metric_key}_violin.png")
    plt.savefig(violin_path)
    plt.close()
    print(f"[*] Violin plot saved to {violin_path}")
    
    return {
        "descriptive": {model: {"mean": float(np.mean(model_data[model])), "std": float(np.std(model_data[model]))} for model in models},
        "shapiro_p": float(p_norm),
        "anova_f": float(f_stat),
        "anova_p": float(p_val_anova),
        "tukey_str": tukey_str
    }

def load_mlp_history(tfevents_path, num_epochs=30):
    ea = event_accumulator.EventAccumulator(tfevents_path)
    ea.Reload()
    
    train_loss = [e.value for e in ea.Scalars('Loss/Train')[:num_epochs]]
    train_policy_loss = [e.value for e in ea.Scalars('Loss/Train_Policy')[:num_epochs]]
    train_value_loss = [e.value for e in ea.Scalars('Loss/Train_Value')[:num_epochs]]
    val_loss = [e.value for e in ea.Scalars('Loss/Val')[:num_epochs]]
    val_move_acc = [e.value / 100.0 for e in ea.Scalars('Accuracy/Val_Move')[:num_epochs]]
    val_res_acc = [e.value / 100.0 for e in ea.Scalars('Accuracy/Val_Result')[:num_epochs]]
    
    while len(train_loss) < num_epochs:
        train_loss.append(train_loss[-1])
        train_policy_loss.append(train_policy_loss[-1])
        train_value_loss.append(train_value_loss[-1])
        val_loss.append(val_loss[-1])
        val_move_acc.append(val_move_acc[-1])
        val_res_acc.append(val_res_acc[-1])
        
    return {
        "train_loss": [train_loss],
        "train_policy_loss": [train_policy_loss],
        "train_value_loss": [train_value_loss],
        "train_aux_loss": [[0.0] * num_epochs],
        "val_loss": [val_loss],
        "val_move_acc": [val_move_acc],
        "val_res_acc": [val_res_acc]
    }

def main():
    global ABLATION_ROOT, RESULTS_JSON, OUTPUT_DIR
    
    parser = argparse.ArgumentParser(description="Run statistical tests on supervised ablations")
    parser.add_argument("--ablation_root", type=str, default="results/ablations/ablations_dk64_n3",
        help="Ablations root directory (default: results/ablations/ablations_dk64_n3)"
    )
    parser.add_argument("--results_json", type=str, default=None,
        help="Path to evaluation results JSON (default: <ablation_root>/test_evaluation_results.json)"
    )
    parser.add_argument("--output_dir", type=str, default=None,
        help="Output directory (default: <ablation_root>)"
    )
    args = parser.parse_args()
    
    ABLATION_ROOT = args.ablation_root
    RESULTS_JSON = args.results_json if args.results_json is not None else os.path.join(ABLATION_ROOT, "test_evaluation_results.json")
    OUTPUT_DIR = args.output_dir if args.output_dir is not None else ABLATION_ROOT

    if not os.path.exists(RESULTS_JSON):
        print(f"[ERROR] Evaluation results file {RESULTS_JSON} not found. Run evaluation first.")
        sys.exit(1)
        
    with open(RESULTS_JSON, 'r') as f:
        results_data = json.load(f)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Run statistical tests for Move Accuracy (Precision@1) and Result Accuracy
    move_stats = run_stats_on_metric(results_data, "move_accuracy", "Move Accuracy (Precision@1)")
    res_stats = run_stats_on_metric(results_data, "result_accuracy", "Result Accuracy")
    
    # 2. Extract and Plot Learning Curves
    seeds = sorted([s for s in os.listdir(ABLATION_ROOT) if os.path.isdir(os.path.join(ABLATION_ROOT, s))])
    model_names = ["simple_nofact", "simple_fact", "spatial_nofact", "spatial_fact"]
    
    if "dk128" in ABLATION_ROOT:
        log_mapping = {
            "simple_nofact": "trnsf_dk128_d5_simple.log",
            "simple_fact": "trnsf_dk128_d5_facted.log",
            "spatial_nofact": "trnsf_dk128_d5_spatial_simple.log",
            "spatial_fact": "trnsf_dk128_d5_spatial_fact.log"
        }
    else:
        log_mapping = {
            "simple_nofact": "trnsf_d4_dk64_depth3_simple.log",
            "simple_fact": "trnsf_d4_dk64_depth3_facted.log",
            "spatial_nofact": "trnsf_d4_dk64_depth3_spatial_simple.log",
            "spatial_fact": "trnsf_d4_dk64_depth3_spatial_fact.log"
        }
    
    print("\n[*] Processing learning curves from logs...")
    history = process_learning_curves(seeds, model_names, log_mapping)
    
    # Load MLP history and inject it
    print("[*] Loading MLP agent history from TensorBoard events...")
    mlp_history = load_mlp_history("experiments/exp1_mlp_transf/mlp_big/events.out.tfevents.1781575278.usbt0p-machine.2982652.0", num_epochs=30)
    history["mlp_big"] = mlp_history
    
    plot_curves(history, "train_loss", "Training Loss", "Training Loss Evolution", os.path.join(OUTPUT_DIR, "learning_curves_train_loss.png"))
    plot_curves(history, "train_policy_loss", "Training Policy Loss", "Policy Head Training Loss Evolution", os.path.join(OUTPUT_DIR, "learning_curves_train_policy_loss.png"))
    plot_curves(history, "train_value_loss", "Training Value Loss", "Value Head Training Loss Evolution", os.path.join(OUTPUT_DIR, "learning_curves_train_value_loss.png"))
    plot_curves(history, "train_aux_loss", "Training Auxiliary Loss", "Auxiliary Policy Head Training Loss Evolution", os.path.join(OUTPUT_DIR, "learning_curves_train_aux_loss.png"))
    plot_curves(history, "val_loss", "Validation Loss", "Validation Loss Evolution", os.path.join(OUTPUT_DIR, "learning_curves_val_loss.png"))
    plot_curves(history, "val_move_acc", "Validation Move Accuracy", "Validation Move Accuracy Evolution", os.path.join(OUTPUT_DIR, "learning_curves_val_move_acc.png"))
    plot_curves(history, "val_res_acc", "Validation Result Accuracy", "Validation Result Accuracy Evolution", os.path.join(OUTPUT_DIR, "learning_curves_val_res_acc.png"))
    
    # Save statistics details to text file
    stat_text_path = os.path.join(OUTPUT_DIR, "statistical_analysis_summary.txt")
    with open(stat_text_path, "w") as f:
        f.write("=== STATISTICAL ANALYSIS OF SUPERVISED ABLATIONS ===\n\n")
        f.write("--- MOVE ACCURACY (PRECISION@1) ---\n")
        f.write(f"Normality SW test p-value: {move_stats['shapiro_p']:.6f}\n")
        f.write(f"One-way ANOVA p-value: {move_stats['anova_p']:.6g}\n")
        f.write("Tukey HSD Post-Hoc Pairwise Comparisons:\n")
        f.write(move_stats['tukey_str'])
        f.write("\n\n")
        f.write("--- RESULT ACCURACY ---\n")
        f.write(f"Normality SW test p-value: {res_stats['shapiro_p']:.6f}\n")
        f.write(f"One-way ANOVA p-value: {res_stats['anova_p']:.6g}\n")
        f.write("Tukey HSD Post-Hoc Pairwise Comparisons:\n")
        f.write(res_stats['tukey_str'])
        
    print(f"\n[*] Statistical analysis text summary saved to {stat_text_path}")
    
    # Copy files to ABLATION_ROOT and doc/figures
    filenames = [
        "learning_curves_train_loss.png",
        "learning_curves_train_policy_loss.png",
        "learning_curves_train_value_loss.png",
        "learning_curves_train_aux_loss.png",
        "learning_curves_val_loss.png",
        "learning_curves_val_move_acc.png",
        "learning_curves_val_res_acc.png",
        "test_move_accuracy_boxplot.png",
        "test_move_accuracy_violin.png",
        "test_result_accuracy_boxplot.png",
        "test_result_accuracy_violin.png",
        "statistical_analysis_summary.txt"
    ]
    
    os.makedirs(ABLATION_ROOT, exist_ok=True)
    
    for fname in filenames:
        src_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(src_path):
            dest_ablation = os.path.join(ABLATION_ROOT, fname)
            if os.path.abspath(src_path) != os.path.abspath(dest_ablation):
                shutil.copy(src_path, dest_ablation)
            print(f"[*] Copied {fname} to {ABLATION_ROOT} if not already there")

if __name__ == "__main__":
    main()
