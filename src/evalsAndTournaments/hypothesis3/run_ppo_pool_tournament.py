import os
import sys
import json
import glob
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Ensure PYTHONPATH includes the current directory
sys.path.append(os.getcwd())

from src.chess.agents import (
    RandomAgent,
    HeuristicAgent,
)
from src.chess.agents.transformer import TransformerAgent
from src.chess.arena import play_matchup
from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig
from src.evalsAndTournaments.run_agent_tournament import plot_win_rate_matrix, estimate_elo_bt

def plot_termination_reasons_stacked(agent_reasons, agent_names, save_path):
    """Plot a stacked bar chart of game termination reasons for each agent."""
    categories = ["checkmate", "stalemate", "insufficient_material", "50_move_rule", "3_repetition_rule", "max_moves"]
    category_labels = {
        "checkmate": "Checkmate",
        "stalemate": "Stalemate",
        "insufficient_material": "Insufficient Material",
        "50_move_rule": "50-Move Rule",
        "3_repetition_rule": "3-Repetition Rule",
        "max_moves": "Max Moves"
    }
    
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#ccb974"]
    
    # Prepare data
    data = {cat: [] for cat in categories}
    for name in agent_names:
        total = sum(agent_reasons[name].values())
        for cat in categories:
            val = agent_reasons[name].get(cat, 0)
            percentage = (val / total * 100) if total > 0 else 0
            data[cat].append(percentage)
            
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bottom = np.zeros(len(agent_names))
    
    for i, cat in enumerate(categories):
        ax.bar(agent_names, data[cat], label=category_labels[cat], bottom=bottom, color=colors[i], width=0.5, alpha=0.9)
        bottom += np.array(data[cat])
        
    ax.set_ylabel("Percentage of Game Outcomes (%)", fontsize=11)
    ax.set_title("Game Termination Reasons Breakdown per Agent", fontsize=12, pad=12)
    ax.set_xticks(np.arange(len(agent_names)))
    ax.set_xticklabels(agent_names, rotation=30, ha="right", fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_checkpoint_agent(checkpoint_path, encoder_config, device, name):
    """Instantiate and load a policy checkpoint into a TransformerAgent with use_lookahead=False."""
    return TransformerAgent(checkpoint_path, encoder_config, device=device, name=name, use_lookahead=False)

def main():
    parser = argparse.ArgumentParser(description="Run Gardner Minichess PPO Final Round-Robin Tournament.")
    parser.add_argument("--results_dir", type=str, default="results/hypothesis3", help="Directory where seed logs are stored")
    parser.add_argument("--seed", type=int, default=42, help="Seed directory to pull checkpoints from")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games per matchup")
    parser.add_argument("--temp", type=float, default=0.15, help="Temperature for move selection")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run models on")
    parser.add_argument("--output_dir", type=str, default="results/hypothesis3/final_tournament", help="Directory to save output files")
    parser.add_argument("--variant", type=str, default="both", choices=["both", "tabula_rasa", "pretrained"], help="Which variant to include in the tournament")
    parser.add_argument("--num_checkpoints", type=int, default=20, help="Number of checkpoints to select from the pool")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. PPO Architecture Configuration (64dk, 3 blocks, spatial)
    encoder_config = EncoderConfig(
        embed_dim=64,
        num_heads=8,
        num_blocks=3,
        batch_size=1,
        policy_size=704,
        mlp_expand_factor=4,
        representation="spatial",
        use_factorized_policy=False,
        attn_backend="math",
        autocast_mode="none"
    )
    
    # 2. Gather baselines
    agents_map = {
        "RandomAgent": RandomAgent(),
        "HeuristicAgent": HeuristicAgent()
    }
    
    variants = [args.variant] if args.variant != "both" else ["tabula_rasa", "pretrained"]
    for variant in variants:
        pool_dir = os.path.join(args.results_dir, f"{variant}_seed_{args.seed}", "pool")
        if not os.path.exists(pool_dir):
            print(f"[INFO] No pool directory found for variant={variant}, seed={args.seed}. Skipping.")
            continue
            
        checkpoints = sorted(glob.glob(os.path.join(pool_dir, "checkpoint_*.pth")))
        if not checkpoints:
            print(f"[INFO] No checkpoints found in {pool_dir}. Skipping.")
            continue
            
        # Select uniformly distributed checkpoints from the pool
        total_ckpts = len(checkpoints)
        if total_ckpts <= args.num_checkpoints:
            selected_indices = list(range(total_ckpts))
        else:
            selected_indices = np.linspace(0, total_ckpts - 1, args.num_checkpoints, dtype=int).tolist()
            
        selected_ckpts = {}
        for idx in selected_indices:
            path = checkpoints[idx]
            filename = os.path.basename(path)
            # Find checkpoint number like checkpoint_0050.pth -> it50
            num_match = re.search(r"checkpoint_(\d+)\.pth", filename)
            if num_match:
                ckpt_id = f"it{int(num_match.group(1))}"
            else:
                ckpt_id = f"idx{idx}"
            selected_ckpts[ckpt_id] = path
            
        for ckpt_id, path in selected_ckpts.items():
            agent_name = f"{variant}_{ckpt_id}"
            print(f"[INFO] Loading checkpoint agent: {agent_name} from {os.path.basename(path)}")
            try:
                agent = load_checkpoint_agent(path, encoder_config, args.device, agent_name)
                agents_map[agent_name] = agent
            except Exception as e:
                print(f"[ERROR] Failed to load checkpoint {agent_name}: {e}")
                
    agent_names = list(agents_map.keys())
    num_agents = len(agent_names)
    print(f"\n[INFO] Starting round-robin tournament with {num_agents} agents: {agent_names}")
    
    # Win rate and decisive win rate matrices
    win_rate_matrix_with_draws = np.zeros((num_agents, num_agents))
    win_rate_matrix_no_draws = np.zeros((num_agents, num_agents))
    
    # Accumulate termination reasons per agent
    categories = ["checkmate", "stalemate", "insufficient_material", "50_move_rule", "3_repetition_rule", "max_moves"]
    agent_reasons = {name: {cat: 0 for cat in categories} for name in agent_names}
    
    # Run matchups
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                win_rate_matrix_with_draws[i, j] = 0.5
                win_rate_matrix_no_draws[i, j] = 0.5
                continue
                
            agent_a = agents_map[agent_names[i]]
            agent_b = agents_map[agent_names[j]]
            
            # Save matchup JSON
            log_path = os.path.join(args.output_dir, f"matchup_{agent_a.name}_vs_{agent_b.name}.json")
            
            print(f"[Matchup] Running: {agent_a.name} vs {agent_b.name}...")
            results = play_matchup(
                agent_a,
                agent_b,
                num_games=args.num_games,
                max_moves=100,
                temperature=args.temp,
                save_log=log_path
            )
            
            # Calculate win rate of A against B
            wins_a = results["agent1_wins"]
            wins_b = results["agent2_wins"]
            draws = results["draws"]
            
            score_a = wins_a + 0.5 * draws
            win_rate_ab = score_a / args.num_games
            win_rate_matrix_with_draws[i, j] = win_rate_ab
            
            total_decisive = wins_a + wins_b
            if total_decisive > 0:
                win_rate_matrix_no_draws[i, j] = wins_a / total_decisive
            else:
                win_rate_matrix_no_draws[i, j] = 0.5
                
            # Accumulate termination reasons
            reasons = results.get("reasons", {})
            for cat in categories:
                count = reasons.get(cat, 0)
                agent_reasons[agent_a.name][cat] += count
                agent_reasons[agent_b.name][cat] += count
                
    # Estimate ELO
    elo_ratings = estimate_elo_bt(
        win_rate_matrix_with_draws,
        agent_names,
        base_agent="RandomAgent",
        base_elo=1000.0,
        num_games=args.num_games
    )
    
    # Plot win rate matrices using shared run_agent_tournament utility
    plot_win_rate_matrix(
        win_rate_matrix_with_draws,
        agent_names,
        os.path.join(args.output_dir, "win_rate_matrix_with_draws.png"),
        title=f"Tournament Win Rate Matrix with Draws ({args.num_games} games/matchup)",
        colorbar_label="Win Rate (Wins + 0.5*Draws)"
    )
    
    plot_win_rate_matrix(
        win_rate_matrix_no_draws,
        agent_names,
        os.path.join(args.output_dir, "win_rate_matrix_no_draws.png"),
        title=f"Tournament Decisive Win Rate Matrix ignoring Draws ({args.num_games} games/matchup)",
        colorbar_label="Decisive Win Rate (Wins / Decisive Games)"
    )
    
    # Plot stacked bar chart of game termination reasons
    plot_reasons_path = os.path.join(args.output_dir, "game_termination_reasons.png")
    plot_termination_reasons_stacked(agent_reasons, agent_names, plot_reasons_path)
    print(f"[*] Game termination reasons stacked bar chart saved to {plot_reasons_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print(f" TOURNAMENT SUMMARY TABLE (Seed {args.seed}) ")
    print("=" * 70)
    print(f"  {'Agent Name':25s} | {'Elo Rating':12s}")
    print("-" * 70)
    for name, elo in zip(agent_names, elo_ratings):
        print(f"  {name:25s} | {elo:12.1f}")
    print("=" * 70 + "\n")
    
    # Save JSON results
    json_path = os.path.join(args.output_dir, "tournament_results.json")
    results_data = {
        "agents": agent_names,
        "matrix_with_draws": win_rate_matrix_with_draws.tolist(),
        "matrix_no_draws": win_rate_matrix_no_draws.tolist(),
        "elo_ratings": elo_ratings,
        "num_games_per_matchup": args.num_games,
        "temperature": args.temp,
        "reasons": agent_reasons
    }
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"[*] Win rate matrix data saved to {json_path}")
    
    # Save text report
    report_path = os.path.join(args.output_dir, "tournament_report.txt")
    with open(report_path, "w") as f:
        f.write("=== POST-PPO ROUND-ROBIN TOURNAMENT REPORT ===\n\n")
        f.write(f"Parameters: num_games_per_matchup={args.num_games}, temperature={args.temp}, seed={args.seed}\n\n")
        f.write(f"{'Agent Name':25s} | {'Elo Rating':12s}\n")
        f.write("-" * 50 + "\n")
        for name, elo in zip(agent_names, elo_ratings):
            f.write(f"{name:25s} | {elo:12.1f}\n")
    print(f"[*] Text summary report saved to {report_path}")

if __name__ == "__main__":
    main()
