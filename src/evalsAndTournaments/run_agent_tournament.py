import os
import sys
import json
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure PYTHONPATH includes the current directory
sys.path.append(os.getcwd())

from src.chess.agents import (
    RandomAgent,
    HeuristicAgent,
    MLPAgent,
    TransformerAgent,
)
from src.chess.arena import play_matchup

# default paths
TEST_MODEL_PATH = None


def parse_agent_dir(path):
    dirname = os.path.basename(path)
    config = {
        "embed_dim": 64,
        "num_blocks": 3,
        "representation": "simple",
        "factorized_policy": False,
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


def get_ablation_agents(seed_dir, device="cpu"):
    if not os.path.exists(seed_dir):
        print(f"[ERROR] Ablation seed directory {seed_dir} not found.")
        sys.exit(1)

    agents_list = []
    # Find all subdirectories containing best_model.pth
    for child in os.listdir(seed_dir):
        child_path = os.path.join(seed_dir, child)
        if os.path.isdir(child_path):
            model_path = os.path.join(child_path, "best_model.pth")
            if os.path.exists(model_path):
                config = parse_agent_dir(child_path)

                # Determine descriptive name from folder name
                parts = child.split("_")
                repr_str = "simple" if "simple" in parts else "spatial"
                fact_str = "fact" if "fact" in parts else "nofact"
                name = f"{repr_str}_{fact_str}"

                agents_list.append(
                    {
                        "name": name,
                        "type": "transformer",
                        "path": model_path,
                        "config": config,
                    }
                )

    # Sort agents by name to be consistent
    agents_list.sort(key=lambda x: x["name"])
    return agents_list


def plot_win_rate_matrix(
    matrix, agent_names, save_path, title="Win Rate Matrix", colorbar_label="Win Rate"
):
    # Convert to array and set diagonal to NaN
    matrix = np.array(matrix, dtype=float)
    np.fill_diagonal(matrix, np.nan)

    num_agents = len(agent_names)
    
    # Scale figure size, ticks, text size, and numeric format dynamically
    if num_agents > 15:
        plt.figure(figsize=(10, 8), dpi=150)
        fontsize_tick = 7
        fontsize_text = 6
        diagonal_text = "-"
        val_format = lambda v: f"{int(round(v * 100))}%"
        title_fontsize = 12
    else:
        plt.figure(figsize=(9, 7), dpi=150)
        fontsize_tick = 9
        fontsize_text = 9
        diagonal_text = "-"
        val_format = lambda v: f"{v * 100:.1f}%"
        title_fontsize = 14

    # Copy colormap and set bad values to lightgray
    cmap = plt.colormaps.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgray")

    im = plt.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)
    plt.colorbar(im, label=colorbar_label)

    # Tick labels
    plt.xticks(np.arange(num_agents), agent_names, rotation=45, ha="right", fontsize=fontsize_tick)
    plt.yticks(np.arange(num_agents), agent_names, fontsize=fontsize_tick)

    # Cell text annotations
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                plt.text(
                    j,
                    i,
                    diagonal_text,
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=fontsize_text,
                )
            else:
                val = matrix[i, j]
                # Use white text for dark colors, black for light
                color = "white" if abs(val - 0.5) > 0.3 else "black"
                plt.text(
                    j,
                    i,
                    val_format(val),
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                    fontsize=fontsize_text,
                )

    plt.title(title, fontsize=title_fontsize, pad=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Win rate matrix heatmap saved to {save_path}")


def estimate_elo_bt(
    win_rate_matrix,
    agent_names,
    base_agent="RandomAgent",
    base_elo=1000.0,
    num_games=100,
):
    """This is needed since we dont have elos, only observations.
    we want the elo values that maximize the likelihood of the observations,
    but this is not solvable analytically. thus, we use an iterative approach trough MLE
    """

    num_agents = len(agent_names)
    # Reconstruct score matrix: W[i, j] = score of i against j
    W = win_rate_matrix * num_games

    # Bradley-Terry MLE via minorization-maximization (MM)
    # Add a tiny smoothing factor to prevent 0 or infinite Elos
    W_smoothed = W + 0.1

    p = np.ones(num_agents)
    for _ in range(2000):
        p_new = np.zeros(num_agents)
        for i in range(num_agents):
            W_i = np.sum(W_smoothed[i]) - W_smoothed[i, i]
            denom = 0.0
            for j in range(num_agents):
                if i != j:
                    denom += (W_smoothed[i, j] + W_smoothed[j, i]) / (p[i] + p[j])
            p_new[i] = W_i / denom if denom > 0 else 1e-5
        p = p_new / np.mean(p_new)

    elo = 400 * np.log10(p)

    # Anchor to base_agent
    if base_agent in agent_names:
        base_idx = agent_names.index(base_agent)
        offset = base_elo - elo[base_idx]
        elo = elo + offset

    return elo.tolist()


def main():
    parser = argparse.ArgumentParser(description="Auto-run a round-robin tournament for Minichess agents")
    parser.add_argument("--trial", action="store_true", help="Run in trial mode (RandomAgent vs Test Model)")
    parser.add_argument("--seed", type=int, default=1, help="Seed directory inside ablations root to use for final run (default: 1)")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games per matchup (default: 100)")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature for move selection (default: 0.1)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run models on")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save logs and plots (must be an inner directory within results/tournaments)")
    parser.add_argument("--ablation_root", type=str, default="results/ablations/ablations_dk64_n3", help="Ablations root directory (default: results/ablations/ablations_dk64_n3)")
    parser.add_argument("--seed_prefix", type=str, default="h2_ablation_bigbatch", help="Seed folder prefix within the ablation root (default: h2_ablation_bigbatch)")

    args = parser.parse_args()

    # Ensure output_dir is an inner directory within results/tournaments
    abs_output = os.path.abspath(args.output_dir)
    abs_base = os.path.abspath("results/tournaments")
    if abs_output == abs_base or not abs_output.startswith(abs_base + os.sep):
        parser.error(
            "The output directory must be an inner directory within 'results/tournaments' "
            "(e.g., 'results/tournaments/my_tournament')."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Define agents
    if args.trial:
        print("[*] Running in TRIAL mode...")
        agent_configs = [
            {"name": "RandomAgent", "type": "random", "path": None, "config": None},
            {
                "name": "HeuristicAgent",
                "type": "heuristic",
                "path": None,
                "config": None,
            },
        ]
        num_games = args.num_games if args.num_games != 100 else 6
    else:
        seed_dir = os.path.join(args.ablation_root, f"{args.seed_prefix}_seed_{args.seed}")
        print(
            f"[*] Running in ABLATION mode using seed: {args.seed} from {seed_dir}..."
        )
        agent_configs = get_ablation_agents(seed_dir, device=args.device)

        # Add MLP baseline agent
        agent_configs.append(
            {
                "name": "MLP_Baseline",
                "type": "mlp",
                "path": "experiments/exp1_mlp_transf/mlp_big/best_model.pt",
                "config": {"hidden_size": 1024, "result_mode": "regression"},
            }
        )

        # Add RandomAgent
        agent_configs.append(
            {"name": "RandomAgent", "type": "random", "path": None, "config": None}
        )

        # Add HeuristicAgent
        agent_configs.append(
            {
                "name": "HeuristicAgent",
                "type": "heuristic",
                "path": None,
                "config": None,
            }
        )

        # Sort alphabetically so they are consistently ordered
        agent_configs.sort(key=lambda x: x["name"])
        num_games = args.num_games

    num_agents = len(agent_configs)
    if num_agents == 0:
        print("[ERROR] No agents loaded. Check paths and formats.")
        sys.exit(1)

    agent_names = [a["name"] for a in agent_configs]
    print(f"[*] Loaded {num_agents} agents:")
    for i, a in enumerate(agent_configs):
        print(f"  {i}: {a['name']} (Type: {a['type']})")

    # Initialize matrices
    win_rate_matrix_with_draws = np.full((num_agents, num_agents), 0.0)
    win_rate_matrix_no_draws = np.full((num_agents, num_agents), 0.0)

    # Store policy entropies per matchup to compute average per agent
    agent_entropies = {name: [] for name in agent_names}

    # Run tournament for all pairs (i, j) with i < j
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            a_cfg = agent_configs[i]
            b_cfg = agent_configs[j]

            # Instantiate agent A
            if a_cfg["type"] == "random":
                agent_a = RandomAgent()
            elif a_cfg["type"] == "heuristic":
                agent_a = HeuristicAgent()
            elif a_cfg["type"] == "mlp":
                agent_a = MLPAgent(
                    model_path=a_cfg["path"],
                    hidden_size=a_cfg["config"].get("hidden_size", 1024),
                    result_mode=a_cfg["config"].get("result_mode", "regression"),
                    device=args.device
                )
            else:
                agent_a = TransformerAgent(
                    model_path=a_cfg["path"],
                    config_args=a_cfg["config"],
                    device=args.device
                )
            agent_a.name = a_cfg["name"]

            # Instantiate agent B
            if b_cfg["type"] == "random":
                agent_b = RandomAgent()
            elif b_cfg["type"] == "heuristic":
                agent_b = HeuristicAgent()
            elif b_cfg["type"] == "mlp":
                agent_b = MLPAgent(
                    model_path=b_cfg["path"],
                    hidden_size=b_cfg["config"].get("hidden_size", 1024),
                    result_mode=b_cfg["config"].get("result_mode", "regression"),
                    device=args.device
                )
            else:
                agent_b = TransformerAgent(
                    model_path=b_cfg["path"],
                    config_args=b_cfg["config"],
                    device=args.device
                )
            agent_b.name = b_cfg["name"]

            # Save log file path for the matchup
            suffix = "_trial" if args.trial else f"_seed_{args.seed}_extended"
            log_filename = f"matchup_{agent_a.name}_vs_{agent_b.name}{suffix}.json"
            log_path = os.path.join(args.output_dir, log_filename)

            # Run the tournament
            results = play_matchup(
                agent_a,
                agent_b,
                num_games=num_games,
                max_moves=100,
                temperature=args.temp,
                save_log=log_path,
            )

            # Calculate win rate of A against B (including draws: wins + 0.5*draws)
            score_a = results["agent1_wins"] + 0.5 * results["draws"]
            win_rate_ab = score_a / num_games
            win_rate_ba = 1.0 - win_rate_ab

            win_rate_matrix_with_draws[i, j] = win_rate_ab
            win_rate_matrix_with_draws[j, i] = win_rate_ba

            # Calculate decisive win rate (excluding draws: wins_a / (wins_a + wins_b))
            wins_a = results["agent1_wins"]
            wins_b = results["agent2_wins"]
            total_decisive = wins_a + wins_b
            if total_decisive > 0:
                wr_ab_no_draws = wins_a / total_decisive
                wr_ba_no_draws = wins_b / total_decisive
            else:
                wr_ab_no_draws = 0.5
                wr_ba_no_draws = 0.5

            win_rate_matrix_no_draws[i, j] = wr_ab_no_draws
            win_rate_matrix_no_draws[j, i] = wr_ba_no_draws

            # Accumulate policy entropies (if agent is not random/heuristic which have trivial/empty entropy)
            if a_cfg["type"] not in ["random", "heuristic"]:
                agent_entropies[agent_a.name].append(results["avg_entropy1"])
            if b_cfg["type"] not in ["random", "heuristic"]:
                agent_entropies[agent_b.name].append(results["avg_entropy2"])

            print(
                f"[Matchup] {agent_a.name} vs {agent_b.name}: Wins A={wins_a}, Wins B={wins_b}, Draws={results['draws']}"
            )

    # Compute average policy entropy per agent
    avg_entropies = {}
    for name in agent_names:
        if agent_entropies[name]:
            avg_entropies[name] = float(np.mean(agent_entropies[name]))
        else:
            avg_entropies[name] = 0.0  # for random/heuristic agents

    # Estimate ELO
    elo_ratings = estimate_elo_bt(
        win_rate_matrix_with_draws,
        agent_names,
        base_agent="RandomAgent",
        base_elo=1000.0,
        num_games=num_games,
    )

    # Plot and save heatmaps
    suffix = "_trial" if args.trial else f"_seed_{args.seed}_extended"
    plot_path_draws = os.path.join(
        args.output_dir, f"win_rate_matrix_with_draws{suffix}.png"
    )
    plot_win_rate_matrix(
        win_rate_matrix_with_draws,
        agent_names,
        plot_path_draws,
        title=f"Win Rate Matrix with Draws ({num_games} games/matchup)",
        colorbar_label="Win Rate (Wins + 0.5*Draws)",
    )

    plot_path_no_draws = os.path.join(
        args.output_dir, f"win_rate_matrix_no_draws{suffix}.png"
    )
    plot_win_rate_matrix(
        win_rate_matrix_no_draws,
        agent_names,
        plot_path_no_draws,
        title=f"Win Rate Matrix ignoring Draws ({num_games} games/matchup)",
        colorbar_label="Decisive Win Rate (Wins / Decisive Games)",
    )

    # Save matrix data to JSON
    json_path = os.path.join(args.output_dir, f"win_rate_results{suffix}.json")
    results_data = {
        "agents": agent_names,
        "matrix_with_draws": win_rate_matrix_with_draws.tolist(),
        "matrix_no_draws": win_rate_matrix_no_draws.tolist(),
        "elo_ratings": elo_ratings,
        "policy_entropies": avg_entropies,
        "num_games_per_matchup": num_games,
        "temperature": args.temp,
    }
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"[*] Win rate matrix data saved to {json_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f" TOURNAMENT SUMMARY TABLE (Seed {args.seed}) ")
    print("=" * 70)
    print(
        f"  {'Agent Name':22s} | {'Type':12s} | {'Elo Rating':10s} | {'Avg Policy Entropy':18s}"
    )
    print("-" * 70)
    for name, elo in zip(agent_names, elo_ratings):
        cfg = next(a for a in agent_configs if a["name"] == name)
        entropy_str = (
            f"{avg_entropies[name]:.4f}"
            if cfg["type"] not in ["random", "heuristic"]
            else "N/A"
        )
        print(f"  {name:22s} | {cfg['type']:12s} | {elo:10.1f} | {entropy_str:18s}")
    print("=" * 70 + "\n")

    # Save text summary
    summary_path = os.path.join(args.output_dir, f"tournament_summary{suffix}.txt")
    with open(summary_path, "w") as f:
        f.write("=== TOURNAMENT SUMMARY REPORT ===\n\n")
        f.write(
            f"Parameters: num_games_per_matchup={num_games}, temperature={args.temp}\n\n"
        )
        f.write(
            f"{'Agent Name':22s} | {'Type':12s} | {'Elo Rating':10s} | {'Avg Policy Entropy':18s}\n"
        )
        f.write("-" * 70 + "\n")
        for name, elo in zip(agent_names, elo_ratings):
            cfg = next(a for a in agent_configs if a["name"] == name)
            entropy_str = (
                f"{avg_entropies[name]:.4f}"
                if cfg["type"] not in ["random", "heuristic"]
                else "N/A"
            )
            f.write(
                f"{name:22s} | {cfg['type']:12s} | {elo:10.1f} | {entropy_str:18s}\n"
            )
    print(f"[*] Text summary report saved to {summary_path}")


if __name__ == "__main__":
    import time
    from datetime import timedelta

    start_time = time.time()
    main()
    end_time = time.time()
    print(f"[*] Total execution time: {timedelta(seconds=end_time - start_time)}")
    print(f"[*] Total execution time: {(end_time - start_time) / 60:.2f} minutes")