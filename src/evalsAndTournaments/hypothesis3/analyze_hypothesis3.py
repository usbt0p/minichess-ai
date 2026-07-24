import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import stats

def parse_tb_log(tb_dir):
    """Parse TensorBoard events to extract training metrics."""
    metrics = {
        "iteration": [],
        "reward": [],
        "reward_variance": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "total_loss": []
    }
    if not os.path.exists(tb_dir):
        return metrics
        
    tfevents_files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if "tfevents" in f]
    if not tfevents_files:
        return metrics
        
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(tfevents_files[0])
        ea.Reload()
        
        tags = ea.Tags().get("scalars", [])
        mapping = {
            "reward": "PPO/avg_reward",
            "reward_variance": "PPO/reward_variance",
            "policy_loss": "PPO/policy_loss",
            "value_loss": "PPO/value_loss",
            "entropy": "PPO/entropy",
            "total_loss": "PPO/total_loss"
        }
        
        first_key = "PPO/avg_reward"
        if first_key in tags:
            events = ea.Scalars(first_key)
            iters = [e.step for e in events]
            metrics["iteration"] = iters
            for metric_key, tb_tag in mapping.items():
                if tb_tag in tags:
                    metrics[metric_key] = [e.value for e in ea.Scalars(tb_tag)]
                else:
                    metrics[metric_key] = [0.0] * len(iters)
    except Exception as e:
        print(f"[WARNING] Failed to parse TensorBoard logs in {tb_dir}: {e}")
        
    return metrics

def parse_train_log(log_path):
    """Parse train.log using regex to extract training metrics per iteration as a fallback."""
    metrics = {
        "iteration": [],
        "reward": [],
        "reward_variance": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "total_loss": []
    }
    if not os.path.exists(log_path):
        return metrics
        
    with open(log_path, "r") as f:
        content = f.read()
        
    lines = []
    for line in content.splitlines():
        clean_line = re.sub(r"^\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+\[\w+\]\s+", "", line)
        clean_line = re.sub(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+\[\w+\]\s+", "", clean_line)
        lines.append(clean_line)
    clean_content = "\n".join(lines)
        
    blocks = re.findall(
        r"Iteration (\d+)/\d+.*?\n\s+Avg Episode Reward .*?: ([+-]?\d+\.\d+) \(variance: (\d+\.\d+)\)\n\s+Policy Loss: ([+-]?\d+\.\d+) \| Value Loss: (\d+\.\d+) \| Entropy: (\d+\.\d+)\n\s+Total Loss:\s+([+-]?\d+\.\d+)",
        clean_content,
        re.DOTALL
    )
    
    for b in blocks:
        metrics["iteration"].append(int(b[0]))
        metrics["reward"].append(float(b[1]))
        metrics["reward_variance"].append(float(b[2]))
        metrics["policy_loss"].append(float(b[3]))
        metrics["value_loss"].append(float(b[4]))
        metrics["entropy"].append(float(b[5]))
        metrics["total_loss"].append(float(b[6]))
        
    return metrics

def parse_evaluations(save_dir, baseline):
    """Find and parse all evaluation tournament JSON files for a baseline."""
    evals = {}
    if not os.path.exists(save_dir):
        return evals
        
    # Search for files like: eval_random_iter_005.json
    for filename in os.listdir(save_dir):
        match = re.match(rf"eval_{baseline}_iter_(\d+)\.json", filename)
        if match:
            iter_num = int(match.group(1))
            filepath = os.path.join(save_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                summary = data.get("summary", {})
                evals[iter_num] = {
                    "winrate": (summary.get("agent1_wins", 0) + 0.5 * summary.get("draws", 0)) / summary.get("total_games", 60.0),
                    "elo": summary.get("elo_diff", 0.0),
                    "avg_game_length": summary.get("avg_game_length", 0.0),
                    "reasons": summary.get("reasons", {})
                }
            except Exception:
                pass
    return evals

def gather_run_data(run_dir):
    """Gathers all logs and JSON files for a specific seed/run."""
    train_log = os.path.join(run_dir, "train.log")
    save_dir = os.path.join(run_dir, "save")
    tb_dir = os.path.join(run_dir, "tb")
    
    train_metrics = parse_tb_log(tb_dir)
    if not train_metrics["iteration"]:
        train_metrics = parse_train_log(train_log)
        
    eval_random = parse_evaluations(save_dir, "random")
    eval_heuristic = parse_evaluations(save_dir, "heuristic")
    
    return {
        "train": train_metrics,
        "eval_random": eval_random,
        "eval_heuristic": eval_heuristic
    }

def aggregate_seeds(results_base, variant):
    """Aggregate metrics across all seeds of a given variant."""
    aggregated = {
        "train": {},
        "eval_random": {},
        "eval_heuristic": {}
    }
    
    # List all subfolders matching the variant pattern
    run_folders = []
    if os.path.exists(results_base):
        for name in os.listdir(results_base):
            if name.startswith(f"{variant}_seed_"):
                run_folders.append(os.path.join(results_base, name))
                
    if not run_folders:
        return aggregated
        
    all_runs_data = [gather_run_data(folder) for folder in run_folders]
    
    all_runs_data = [r for r in all_runs_data if r["train"]["iteration"]]
    if not all_runs_data:
        return aggregated
        
    # 1. Aggregate Training Metrics
    for key in ["reward", "reward_variance", "policy_loss", "value_loss", "entropy", "total_loss"]:
        # Find iterations present in all runs
        common_iters = None
        for data in all_runs_data:
            iters = set(data["train"]["iteration"])
            if common_iters is None:
                common_iters = iters
            else:
                common_iters = common_iters.intersection(iters)
                
        if not common_iters:
            continue
            
        sorted_iters = sorted(list(common_iters))
        aggregated["train"]["iteration"] = sorted_iters
        aggregated["train"][f"{key}_mean"] = []
        aggregated["train"][f"{key}_std"] = []
        aggregated["train"][f"{key}_raw"] = []
        
        # Build raw array for statistics
        raw_vals = []
        for data in all_runs_data:
            run_vals = []
            for it in sorted_iters:
                idx = data["train"]["iteration"].index(it)
                run_vals.append(data["train"][key][idx])
            raw_vals.append(run_vals)
            
        raw_vals = np.array(raw_vals)
        aggregated["train"][f"{key}_raw"] = raw_vals.tolist()
        aggregated["train"][f"{key}_mean"] = np.mean(raw_vals, axis=0).tolist()
        aggregated["train"][f"{key}_std"] = np.std(raw_vals, axis=0).tolist()

    # 2. Aggregate Evaluations (Random & Heuristic)
    for baseline in ["random", "heuristic"]:
        eval_key = f"eval_{baseline}"
        
        # Find common iterations across runs
        common_iters = None
        for data in all_runs_data:
            iters = set(data[eval_key].keys())
            if common_iters is None:
                common_iters = iters
            else:
                common_iters = common_iters.intersection(iters)
                
        if not common_iters:
            continue
            
        sorted_iters = sorted(list(common_iters))
        aggregated[eval_key]["iteration"] = sorted_iters
        
        for metric in ["winrate", "elo", "avg_game_length"]:
            raw_vals = []
            for data in all_runs_data:
                run_vals = []
                for it in sorted_iters:
                    run_vals.append(data[eval_key][it][metric])
                raw_vals.append(run_vals)
                
            raw_vals = np.array(raw_vals)
            aggregated[eval_key][f"{metric}_raw"] = raw_vals.tolist()
            aggregated[eval_key][f"{metric}_mean"] = np.mean(raw_vals, axis=0).tolist()
            aggregated[eval_key][f"{metric}_std"] = np.std(raw_vals, axis=0).tolist()
            
        # Aggregate reasons (count checkmate, stalemate, etc. at final iteration)
        final_it = sorted_iters[-1]
        aggregated[eval_key]["final_reasons"] = {}
        for data in all_runs_data:
            reasons = data[eval_key].get(final_it, {}).get("reasons", {})
            for r, count in reasons.items():
                aggregated[eval_key]["final_reasons"][r] = aggregated[eval_key]["final_reasons"].get(r, 0) + count

        categories = ["checkmate", "stalemate", "insufficient_material", "50_move_rule", "3_repetition_rule", "max_moves"]
        aggregated[eval_key]["reasons_over_time"] = {cat: [] for cat in categories}
        for it in sorted_iters:
            temp_counts = {cat: 0.0 for cat in categories}
            for data in all_runs_data:
                reasons = data[eval_key].get(it, {}).get("reasons", {})
                for cat in categories:
                    temp_counts[cat] += reasons.get(cat, 0)
            for cat in categories:
                aggregated[eval_key]["reasons_over_time"][cat].append(temp_counts[cat] / len(all_runs_data))
                
    return aggregated

def plot_mean_std_curve(x, mean, std, label, color, ax, linestyle='-'):
    mean = np.array(mean)
    std = np.array(std)
    ax.plot(x, mean, color=color, label=label, linewidth=2, linestyle=linestyle)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

def plot_stacked_termination_reasons(data, baseline, variant_name, output_dir):
    """Plot a stacked area chart of game termination reasons over iterations."""
    eval_key = f"eval_{baseline}"
    if eval_key not in data or "iteration" not in data[eval_key]:
        return
        
    iters = data[eval_key]["iteration"]
    reasons_over_time = data[eval_key]["reasons_over_time"]
    
    categories = ["checkmate", "stalemate", "insufficient_material", "50_move_rule", "3_repetition_rule", "max_moves"]
    category_labels = {
        "checkmate": "Checkmate",
        "stalemate": "Stalemate",
        "insufficient_material": "Insufficient Material",
        "50_move_rule": "50-Move Rule",
        "3_repetition_rule": "3-Repetition Rule",
        "max_moves": "Max Moves"
    }
    
    y = np.array([reasons_over_time[cat] for cat in categories])
    
    is_pretrained = "pre" in variant_name.lower()
    if is_pretrained:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    else:
        # Tabula Rasa: Slightly wider to place the legend cleanly outside to the right
        fig, ax = plt.subplots(figsize=(9.5, 5), dpi=150)
        
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#ccb974"]
    
    ax.stackplot(iters, y, labels=[category_labels[cat] for cat in categories], colors=colors, alpha=0.85)
    
    ax.set_xlabel("PPO Training Iteration", fontsize=11)
    ax.set_ylabel("Average Games Played (Stacked)", fontsize=11)
    ax.set_title(f"Game Termination Reasons vs {baseline.capitalize()} ({variant_name})", fontsize=12, pad=12)
    
    if is_pretrained:
        # Pre-trained: Bottom right
        ax.legend(loc="lower right", fontsize=9)
    else:
        # Tabula Rasa: Outside to the right (Fix to prevent overlap in the flat stacked area)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
        
    ax.grid(True, linestyle="--", alpha=0.3)
    
    variant_folder = "pretrained" if is_pretrained else "tabula_rasa"
    save_folder = os.path.join(os.path.dirname(output_dir), "imgs", variant_folder)
    os.makedirs(save_folder, exist_ok=True)
    
    plt.tight_layout()
    save_path = os.path.join(save_folder, f"reasons_vs_{baseline}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Stacked reasons chart saved to {save_path}")

def plot_final_elo_distribution_joint(tabula_data, pretrained_data, output_dir):
    """Plot joint boxplots and violin plots comparing final Elo ratings across seeds."""
    baselines = ["random", "heuristic"]
    
    for baseline in baselines:
        eval_key = f"eval_{baseline}"
        if eval_key not in tabula_data or eval_key not in pretrained_data:
            continue
            
        final_idx = -1
        # Extract the last evaluated Elo difference for all runs (seeds)
        tab_elos = [run[final_idx] for run in tabula_data[eval_key]["elo_raw"]]
        pre_elos = [run[final_idx] for run in pretrained_data[eval_key]["elo_raw"]]
        
        # Unified Boxplot
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        box = ax.boxplot([tab_elos, pre_elos], patch_artist=True, widths=0.4, labels=["Tabula Rasa", "Pre-trained"])
        colors = ["#ff9999", "#99ccff"]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in box['medians']:
            median.set_color("black")
            median.set_linewidth(2)
        ax.set_ylabel("Final Elo Rating Difference", fontsize=11)
        ax.set_title(f"Final Elo Difference Distribution vs {baseline.capitalize()} (10 seeds)", fontsize=12, pad=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        save_path_box = os.path.join(output_dir, f"final_elo_vs_{baseline}_boxplot.png")
        plt.savefig(save_path_box)
        plt.close()
        print(f"[*] Boxplot vs {baseline} saved to {save_path_box}")
        
        # Unified Violin Plot
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        violins = ax.violinplot([tab_elos, pre_elos], showmedians=True, showextrema=True)
        colors_violin = ["#cc0000", "#0066cc"]
        for i, body in enumerate(violins['bodies']):
            body.set_facecolor(colors_violin[i])
            body.set_alpha(0.4)
        for part in ['cmaxes', 'cmins', 'cbars', 'cmedians']:
            violins[part].set_edgecolor('black')
            violins[part].set_linewidth(1.5)
            
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Tabula Rasa", "Pre-trained"], fontsize=10)
        ax.set_ylabel("Final Elo Rating Difference", fontsize=11)
        ax.set_title(f"Final Elo Difference Violin vs {baseline.capitalize()} (10 seeds)", fontsize=12, pad=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        save_path_violin = os.path.join(output_dir, f"final_elo_vs_{baseline}_violin.png")
        plt.savefig(save_path_violin)
        plt.close()
        print(f"[*] Violin plot vs {baseline} saved to {save_path_violin}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Gardner Minichess PPO results (Hypothesis 3).")
    parser.add_argument("--results_dir", type=str, default="results/hypothesis3", help="Base directory containing training run seed folders")
    parser.add_argument("--output_dir", type=str, default="results/hypothesis3/plots", help="Directory to save final plots and table reports")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("[*] Aggregating seed results for Tabula Rasa...")
    tabula_data = aggregate_seeds(args.results_dir, "tabula_rasa")
    
    print("[*] Aggregating seed results for Pre-trained...")
    pretrained_data = aggregate_seeds(args.results_dir, "pretrained")
    
    has_tabula = bool(tabula_data.get("train"))
    has_pretrained = bool(pretrained_data.get("train"))
    
    if not has_tabula or not has_pretrained:
        print("[WARNING] Missing training data for either Tabula Rasa or Pre-trained. Both are required.")
        return
        
    print("[*] Generating comparison plots...")
    
    # 1. Joined Elo Evolution Plot (Relative Elo Difference)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Tabula Rasa vs Random
    if "eval_random" in tabula_data and "iteration" in tabula_data["eval_random"]:
        x_tab = tabula_data["eval_random"]["iteration"]
        tab_rand_mean = np.array(tabula_data["eval_random"]["elo_mean"])
        plot_mean_std_curve(x_tab, tab_rand_mean, tabula_data["eval_random"]["elo_std"], "Tabula Rasa (vs Random)", "#ff9999", ax)
        
    # Tabula Rasa vs Heuristic
    if "eval_heuristic" in tabula_data and "iteration" in tabula_data["eval_heuristic"]:
        x_tab = tabula_data["eval_heuristic"]["iteration"]
        tab_heur_mean = np.array(tabula_data["eval_heuristic"]["elo_mean"])
        plot_mean_std_curve(x_tab, tab_heur_mean, tabula_data["eval_heuristic"]["elo_std"], "Tabula Rasa (vs Heuristic)", "#cc0000", ax)
        
    # Pre-trained vs Random
    if "eval_random" in pretrained_data and "iteration" in pretrained_data["eval_random"]:
        x_pre = pretrained_data["eval_random"]["iteration"]
        pre_rand_mean = np.array(pretrained_data["eval_random"]["elo_mean"])
        plot_mean_std_curve(x_pre, pre_rand_mean, pretrained_data["eval_random"]["elo_std"], "Pre-trained (vs Random)", "#99ccff", ax)
        
    # Pre-trained vs Heuristic
    if "eval_heuristic" in pretrained_data and "iteration" in pretrained_data["eval_heuristic"]:
        x_pre = pretrained_data["eval_heuristic"]["iteration"]
        pre_heur_mean = np.array(pretrained_data["eval_heuristic"]["elo_mean"])
        plot_mean_std_curve(x_pre, pre_heur_mean, pretrained_data["eval_heuristic"]["elo_std"], "Pre-trained (vs Heuristic)", "#0066cc", ax)
        
    # Baseline line (since this is Elo difference, baseline is always 0.0)
    ax.axhline(y=0.0, color="#4D4D4D", linestyle="--", alpha=0.6, label="Baseline Strength (Diff = 0)")
    
    ax.set_xlabel("PPO Training Iteration", fontsize=11)
    ax.set_ylabel("Elo Rating Difference (vs. Baseline)", fontsize=11)
    ax.set_title("Playing Strength Evolution (Relative Elo Differences)", fontsize=13, pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(-150, 700)
    ax.set_yticks(np.arange(-100, 800, 100))
    ax.legend(loc="upper left", fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "elo_evolution_joint.png"))
    plt.close()
    print(f"[*] Joint Elo evolution plot saved to {os.path.join(args.output_dir, 'elo_evolution_joint.png')}")
            
    # 3. Average Episode Reward Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    if has_tabula:
        x_tab = tabula_data["train"]["iteration"]
        plot_mean_std_curve(x_tab, tabula_data["train"]["reward_mean"], tabula_data["train"]["reward_std"], "Tabula Rasa", "#cc0000", ax)
    if has_pretrained:
        x_pre = pretrained_data["train"]["iteration"]
        plot_mean_std_curve(x_pre, pretrained_data["train"]["reward_mean"], pretrained_data["train"]["reward_std"], "Pre-trained (Supervised Warm-Start)", "#0066cc", ax)
    
    ax.set_xlabel("PPO Training Iteration", fontsize=11)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title("Self-Play Reward Convergence", fontsize=13, pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "mean_episode_reward.png"))
    plt.close()
    
    # 4. Policy Entropy Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    if has_tabula:
        x_tab = tabula_data["train"]["iteration"]
        plot_mean_std_curve(x_tab, tabula_data["train"]["entropy_mean"], tabula_data["train"]["entropy_std"], "Tabula Rasa", "#cc0000", ax)
    if has_pretrained:
        x_pre = pretrained_data["train"]["iteration"]
        plot_mean_std_curve(x_pre, pretrained_data["train"]["entropy_mean"], pretrained_data["train"]["entropy_std"], "Pre-trained (Supervised Warm-Start)", "#0066cc", ax)
    ax.set_xlabel("PPO Training Iteration", fontsize=11)
    ax.set_ylabel("Policy Entropy", fontsize=11)
    ax.set_title("Exploration Policy Entropy Decay", fontsize=13, pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "policy_entropy.png"))
    plt.close()

    # 5. Mean Total Loss Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    if has_tabula:
        x_tab = tabula_data["train"]["iteration"]
        plot_mean_std_curve(x_tab, tabula_data["train"]["total_loss_mean"], tabula_data["train"]["total_loss_std"], "Tabula Rasa", "#cc0000", ax)
    if has_pretrained:
        x_pre = pretrained_data["train"]["iteration"]
        plot_mean_std_curve(x_pre, pretrained_data["train"]["total_loss_mean"], pretrained_data["train"]["total_loss_std"], "Pre-trained (Supervised Warm-Start)", "#0066cc", ax)
    ax.set_xlabel("PPO Training Iteration", fontsize=11)
    ax.set_ylabel("Mean Total Loss", fontsize=11)
    ax.set_title("PPO Total Loss Convergence", fontsize=13, pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "mean_total_loss.png"))
    plt.close()
    print(f"[*] Mean total loss plot saved to {os.path.join(args.output_dir, 'mean_total_loss.png')}")

    # 5. Stacked termination reasons over time
    if has_tabula:
        for baseline in ["random", "heuristic"]:
            plot_stacked_termination_reasons(tabula_data, baseline, "Tabula Rasa", args.output_dir)
    if has_pretrained:
        for baseline in ["random", "heuristic"]:
            plot_stacked_termination_reasons(pretrained_data, baseline, "Pre-trained", args.output_dir)

    # 6. Boxplot & Violin plots of Elo distribution (Unified / Joint)
    plot_final_elo_distribution_joint(tabula_data, pretrained_data, args.output_dir)

    # 6. Statistical report
    report_lines = []
    report_lines.append("=========================================================================================================")
    report_lines.append(" HYPOTHESIS 3 STATISTICAL ANALYSIS REPORT ")
    report_lines.append("=========================================================================================================\n")
    
    table_headers = f"  {'Metric':25s} | {'Tabula Rasa':15s} | {'Pre-trained':15s} | {'t-stat':10s} | {'t-pval':10s} | {'W-stat':10s} | {'W-pval':10s}"
    report_lines.append(table_headers)
    report_lines.append("-" * 75)

    def get_stabilization_iter(iters, elos_mean):
        # Stabilization iteration (first iteration where Elo >= 90% of final mean Elo)
        target = 0.9 * elos_mean[-1]
        for i, elo in enumerate(elos_mean):
            if elo >= target:
                return iters[i]
        return iters[-1]
        
    report_lines.append("-" * 75)
    summary_data = {}

    for baseline in ["random", "heuristic"]:
        eval_key = f"eval_{baseline}"
        if eval_key in tabula_data and "elo_raw" in tabula_data[eval_key]:
            # Extracted final Elo ratings (the last evaluation)
            final_idx = -1
            tab_final_elos = [run[final_idx] for run in tabula_data[eval_key]["elo_raw"]]
            pre_final_elos = [run[final_idx] for run in pretrained_data[eval_key]["elo_raw"]]
            
            t_stat, p_val_t = stats.ttest_ind(pre_final_elos, tab_final_elos, equal_var=False)
            # Wilcoxon rank-sum test
            w_stat, p_val_w = stats.ranksums(pre_final_elos, tab_final_elos)
            
            stab_tab = get_stabilization_iter(tabula_data[eval_key]["iteration"], tabula_data[eval_key]["elo_mean"])
            stab_pre = get_stabilization_iter(pretrained_data[eval_key]["iteration"], pretrained_data[eval_key]["elo_mean"])
            
            summary_data[baseline] = {
                "tab_final_mean": np.mean(tab_final_elos),
                "tab_final_std": np.std(tab_final_elos),
                "pre_final_mean": np.mean(pre_final_elos),
                "pre_final_std": np.std(pre_final_elos),
                "t_stat": t_stat,
                "p_val_t": p_val_t,
                "w_stat": w_stat,
                "p_val_w": p_val_w,
                "stab_tab": stab_tab,
                "stab_pre": stab_pre
            }
            
            report_lines.append(f"  Final Elo Diff vs {baseline.capitalize():5s} | {summary_data[baseline]['tab_final_mean']:6.1f} ± {summary_data[baseline]['tab_final_std']:4.1f} | {summary_data[baseline]['pre_final_mean']:6.1f} ± {summary_data[baseline]['pre_final_std']:4.1f} | {summary_data[baseline]['t_stat']:7.2f} | {summary_data[baseline]['p_val_t']:.6f} | {summary_data[baseline]['w_stat']:7.2f} | {summary_data[baseline]['p_val_w']:.6f}")
            report_lines.append(f"  Stabilization Iteration   | {stab_tab:15d} | {stab_pre:15d} |    N/A    |    N/A     |    N/A     |    N/A")
            report_lines.append("-" * 105)

    # Formal statistical conclusions using both tests
    is_t_sig = all(summary_data[b]["p_val_t"] < 0.05 for b in summary_data)
    is_w_sig = all(summary_data[b]["p_val_w"] < 0.05 for b in summary_data)
    
    report_lines.append("\n=== STATISTICAL INFERENCE SUMMARY ===")
    
    t_concl = (
        f"1. Welch's t-test (parametric): Welch's t-test confirms a highly significant difference in playing strength.\n"
        f"   - Vs. Random: t = {summary_data['random']['t_stat']:.2f}, p = {summary_data['random']['p_val_t']:.6f}\n"
        f"   - Vs. Heuristic: t = {summary_data['heuristic']['t_stat']:.2f}, p = {summary_data['heuristic']['p_val_t']:.6f}\n"
        f"   We reject the null hypothesis of equality of means under Welch's t-test (p < 0.05)."
        if is_t_sig else
        "1. Welch's t-test (parametric): We fail to reject the null hypothesis of equality of means (p >= 0.05)."
    )
    report_lines.append(t_concl)
    
    w_concl = (
        f"2. Wilcoxon Rank-Sum test (non-parametric): The Wilcoxon test also confirms a significant difference in rankings.\n"
        f"   - Vs. Random: W = {summary_data['random']['w_stat']:.2f}, p = {summary_data['random']['p_val_w']:.6f}\n"
        f"   - Vs. Heuristic: W = {summary_data['heuristic']['w_stat']:.2f}, p = {summary_data['heuristic']['p_val_w']:.6f}\n"
        f"   We reject the null hypothesis of equal rank distributions under Wilcoxon rank-sum (p < 0.05)."
        if is_w_sig else
        "2. Wilcoxon Rank-Sum test (non-parametric): We fail to reject the null hypothesis of equal rank distributions (p >= 0.05)."
    )
    report_lines.append(w_concl)
    
    report_lines.append("\nFinal Conclusion:")
    if is_t_sig and is_w_sig:
        report_lines.append("  Based on the final Elo ratings, we reject the null hypothesis of equality of means.\n"
                            "  The warm-start supervised pre-training provides a STATISTICALLY SIGNIFICANT advantage\n"
                            "  in playing strength convergence (p < 0.05).")
    else:
        report_lines.append("  We fail to reject the null hypothesis of equal playing strength (p >= 0.05).")
        
    report_content = "\n".join(report_lines)
    print(report_content)
    
    report_path = os.path.join(args.output_dir, "statistical_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"[*] Statistical analysis report saved to {report_path}")

if __name__ == "__main__":
    main()

# Execution Examples:
# -------------------
# 1. Run analysis with default settings (reads from results/hypothesis3/
#    and outputs to results/hypothesis3/plots/):
#
#    PYTHONPATH=. .venv/bin/python3 src/evalsAndTournaments/hypothesis3/analyze_hypothesis3.py
#
# 2. Run analysis with custom input and output directories:
#
#    PYTHONPATH=. .venv/bin/python3 src/evalsAndTournaments/hypothesis3/analyze_hypothesis3.py \
#        --results_dir results/hypothesis3_custom \
#        --output_dir results/hypothesis3_custom/plots
