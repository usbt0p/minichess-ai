'''
CLI util for pitting two agents against each other
'''

import argparse
import torch
from src.utils.utils import set_seed
from src.training.utils import decode_move_indices

from src.chess.agents.random import RandomAgent
from src.chess.agents.mlp import MLPAgent
from src.chess.agents.transformer import TransformerAgent

from src.chess.arena import play_matchup

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Minichess models in a tournament setting")
    
    # Agent 1 Config
    parser.add_argument("--agent1_type", type=str, choices=["mlp", "transformer", "random"], default="random")
    parser.add_argument("--agent1_path", type=str, default="")
    parser.add_argument("--agent1_dim", type=int, default=64)
    parser.add_argument("--agent1_blocks", type=int, default=3)
    parser.add_argument("--agent1_repr", type=str, choices=["simple", "spatial"], default="simple")
    parser.add_argument("--agent1_factorized", action="store_true")
    
    # Agent 2 Config
    parser.add_argument("--agent2_type", type=str, choices=["mlp", "transformer", "random"], default="random")
    parser.add_argument("--agent2_path", type=str, default="")
    parser.add_argument("--agent2_dim", type=int, default=64)
    parser.add_argument("--agent2_blocks", type=int, default=3)
    parser.add_argument("--agent2_repr", type=str, choices=["simple", "spatial"], default="simple")
    parser.add_argument("--agent2_factorized", action="store_true")
    
    # Tournament parameters
    parser.add_argument("--num_games", type=int, default=20, help="Number of games to play (even number recommended)")
    parser.add_argument("--max_moves", type=int, default=100, help="Maximum number of halfmoves per game before declaring draw")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature for move selection sampling (0.0 for greedy)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load neural networks ('cpu' or 'cuda')")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--save_log", type=str, default="", help="Path to save detailed tournament JSON log")
    
    args = parser.parse_args()
    
    set_seed(args.seed)

    # Example: run best model against random agent for 10 games with greedy policy
    '''
    python src/chess/agentVSagent.py \
        --agent1_type transformer \
        --agent1_path "/home/usbt0p/TFG/experiments/test_value_refactor/20260620_013832_d4_val_trnsf_dk64_n3_value_refactor_spatial_nofact_dk64_depth3_lr3.00e-03_bs512/best_model.pth" \
        --agent1_dim 64 --agent1_blocks 3 \
        --agent1_repr "spatial" \
        --agent2_type random \
        --num_games 10 \
        --temp 0.0 \
        --device cuda
    '''

    # Instantiate Agent 1
    if args.agent1_type == "random":
        agent1 = RandomAgent()
    elif args.agent1_type == "mlp":
        config_1 = {
            "hidden_size": 512 if "small" in args.agent1_path else 1024,
            "result_mode": "regression"
        }
        agent1 = MLPAgent(args.agent1_path, hidden_size=config_1["hidden_size"], device=args.device)
    elif args.agent1_type == "transformer":
        # TODO this usage of the config is wrong. we already have a src.models.transformerEncoder.EncoderConfig class and should take advantage of it.
        # this introduces complexity, coupling and duplication
        # move encoder config and training config out of the model agent if needed,
        # or make a factory or something...
        config_1 = {
            "embed_dim": args.agent1_dim,
            "num_blocks": args.agent1_blocks,
            "representation": args.agent1_repr,
            "factorized_policy": args.agent1_factorized
        }
        agent1 = TransformerAgent(args.agent1_path, config_1, device=args.device)
        
    # Instantiate Agent 2
    if args.agent2_type == "random":
        agent2 = RandomAgent()
    elif args.agent2_type == "mlp":
        config_2 = {
            "hidden_size": 512 if "small" in args.agent2_path else 1024,
            "result_mode": "regression"
        }
        agent2 = MLPAgent(args.agent2_path, hidden_size=config_2["hidden_size"], device=args.device)
    elif args.agent2_type == "transformer":
        config_2 = {
            "embed_dim": args.agent2_dim,
            "num_blocks": args.agent2_blocks,
            "representation": args.agent2_repr,
            "factorized_policy": args.agent2_factorized
        }
        agent2 = TransformerAgent(args.agent2_path, config_2, device=args.device)
    
    results = play_matchup(
        agent1, 
        agent2, 
        num_games=args.num_games, 
        max_moves=args.max_moves, 
        temperature=args.temp, 
        save_log=args.save_log if args.save_log else None
    )

    print(results)
