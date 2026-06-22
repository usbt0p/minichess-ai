import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyffish
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.models.fnnPromotionMasking import BaselineNet
from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig
from src.models.dataset_parser import uci_to_index, parse_fen_to_features
from src.utils.utils import set_seed
from src.training.utils import decode_move_indices

# Mapping of pieces for FEN parsing
PIECE_MAP = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}

def index_to_uci(idx: int) -> str:
    """Decodes a move index (0-703) back to a UCI move string like "a1a2" or "a1b2q"."""
    from_sq_tensor, to_sq_tensor, promo_class_tensor = decode_move_indices(
        torch.tensor([idx], dtype=torch.long), device="cpu"
    )
    f_sq = from_sq_tensor.item()
    t_sq = to_sq_tensor.item()
    p_class = promo_class_tensor.item()
    
    file_from = f_sq % 5
    rank_from = f_sq // 5
    file_to = t_sq % 5
    rank_to = t_sq // 5
    
    move_str = f"{chr(ord('a') + file_from)}{rank_from + 1}{chr(ord('a') + file_to)}{rank_to + 1}"
    if p_class > 0:
        promo_char = ['q', 'r', 'b', 'n'][(p_class - 1) % 4]
        move_str += promo_char
    return move_str

class IllegalMoveError(Exception):
    def __init__(self, move, legal_moves=None, message="Illegal move encountered"):
        self.message = message
        self.move = move
        self.legal_moves = legal_moves
        super().__init__(
            f"{self.message}: {self.move}. "
            f"Legal moves: {len(legal_moves) if legal_moves is not None else 'unknown'}"
        )

# TODO actually use this
@dataclass
class FenParts:
    fen: str
    active_player: str
    halfmove: int
    repetition: int
    
    def __init__(self, fen: str):
        parts = fen.split(" ") 
        self.fen = fen
        self.active_player = parts[1]
        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.repetition = int(parts[5]) if len(parts) > 5 else 0


class ChessAgent(ABC):
    """Abstract base class for a chess agent"""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_move(self, fen: str, legal_moves: list, temperature: float = 1.0):
        """Selects a move from the legal moves"""
        pass

class RandomAgent(ChessAgent):
    def __init__(self):
        self.name = "RandomAgent"
        
    def select_move(self, fen: str, legal_moves: list, temperature: float = 1.0):
        return random.choice(legal_moves), 0.0

class ModelAgent(ChessAgent):
    def __init__(self, model_type, model_path, config_args, device="cpu"):
        self.model = None
        self.model_type = model_type
        self.device = device
        self.name = os.path.basename(model_path).replace(".pth", "").replace(".pt", "")
        
        if model_type == "mlp":
            self.model = BaselineNet(
                hidden_size=config_args.get("hidden_size", 512),
                result_mode=config_args.get("result_mode", "regression")
            )
        elif model_type == "transformer":
            encoder_config = EncoderConfig(
                embed_dim=config_args["embed_dim"],
                num_heads=8,
                num_blocks=config_args["num_blocks"],
                batch_size=1, # TODO using higher bs by parallelizing games would be better
                policy_size=704,
                mlp_expand_factor=config_args.get("mlp_expand", 4),
                representation=config_args.get("representation", "simple"),
                use_factorized_policy=config_args.get("factorized_policy", False),
                attn_backend="math",
                autocast_mode="none"
            )
            self.representation = encoder_config.representation
            self.model = MiniChessTransformerEncoder(encoder_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Load state dict
        # TODO maybe better to use MiniChessTransformerEncoder.from_pretrained()?
        print(f"[INFO] Loading checkpoint '{model_path}' onto {device}...")
        state_dict = torch.load(model_path, map_location=device)
        clean_state_dict = {}
        for k, v in state_dict.items():
            # strip "_orig_mod." added by compiled torch.compile() call
            if k.startswith("_orig_mod."):
                clean_state_dict[k[10:]] = v
            else:
                clean_state_dict[k] = v
        self.model.load_state_dict(clean_state_dict)
        self.model.to(device)
        self.model.eval() # turn off dropout, normalization, etc.

    def select_move(self, fen: str, legal_moves: list, temperature: float = 1.0):
        if not legal_moves:
            return None, 0.0
            
        # Parse FEN
        # only the board repr of the fen. well need parts for other stuff
        # TODO the "feature creation" should be abstracted to a private method and called withing select move
        parts = fen.split(" ") 
        board_features = np.full(25, 12, dtype=np.uint8)
        parse_fen_to_features(parts[0], PIECE_MAP, board_features)
        
        board_tensor = torch.from_numpy(board_features).long().to(self.device)
        halfmove = int(parts[4]) if len(parts) > 4 else 0
        halfmove_tensor = torch.tensor([halfmove], dtype=torch.long, device=self.device)
        repetition_tensor = torch.tensor([0], dtype=torch.long, device=self.device)
        
        # Build features based on model type
        # TODO separate this logic. create another agent for mlp and other for transformer
        if self.model_type == "mlp":
            # create one-hot encoding of the board
            one_hot = torch.zeros((25, 13), dtype=torch.float32, device=self.device)
            one_hot.scatter_(1, board_tensor.unsqueeze(1), 1.0) 
            features = one_hot.flatten().unsqueeze(0)
            
            # Policy mask
            mask = torch.zeros(704, dtype=torch.bool, device=self.device)
            legal_indices = [uci_to_index(m, promotions=True) for m in legal_moves]
            mask[legal_indices] = True
            
            with torch.no_grad():
                policy_logits, _ = self.model(features, mask.unsqueeze(0))
                policy_logits = policy_logits.squeeze(0)
        else:
            # Transformer features
            if self.representation == "spatial":
                active_player = 1 if parts[1] == 'w' else 0 # TODO important to check if this is right
                active_player_tensor = torch.tensor([active_player], dtype=torch.long, device=self.device)
                features = torch.cat([board_tensor, repetition_tensor, halfmove_tensor, active_player_tensor], dim=0).unsqueeze(0)
            else:
                features = torch.cat([board_tensor, repetition_tensor, halfmove_tensor], dim=0).unsqueeze(0)
                
            with torch.no_grad():
                # Policy head outputs logits of size 704
                policy_logits, value_pred = self.model(features)
                policy_logits = policy_logits.squeeze(0)
                
        # Mask out illegal moves
        legal_indices = [uci_to_index(m, promotions=True) for m in legal_moves]
        masked_logits = torch.full_like(policy_logits, -1e9)
        masked_logits[legal_indices] = policy_logits[legal_indices]
        
        # calculate policy entropy (shannon entropy of the legal moves probability distribution)
        # measures how much the model is "sure" about its move
        probs = F.softmax(policy_logits[legal_indices], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        
        # TODO this is a problem. The usual way lis to use the value to orient the sampling.
        # But to do that we have to do tree search of at least depth 1.
        # The thing is: is there a way of computing, from the current board tensor and,
        # given all the legal moves, all of the possible boards in constant time? that is, 
        # "spawning" the children boards of depth 1

        # TODO the thing is it would probably be good to have both the value and the policy
        # as methods for selecting moves. but that has to be implemented and correctly abstracted
        # for now the policy is enough...

        # TODO the best thing to do might be to have policy pick top-k moves, calculate their fens, and then use the value to select the best one
  
        if temperature == 0.0:
            # Greedy selection
            best_idx = torch.argmax(masked_logits).item()
        else:
            # Stochastic sampling, like llms do
            scaled_logits = masked_logits / temperature
            probs = F.softmax(scaled_logits[legal_indices], dim=-1)
            selected_local_idx = torch.multinomial(probs, 1).item()
            best_idx = legal_indices[selected_local_idx]
        # TODO also would be good to have an option that returns the top-k moves
            
        move_uci = index_to_uci(best_idx)
        # if decoded move is somehow invalid, fail loudly
        if move_uci not in legal_moves:
            raise IllegalMoveError(move_uci, legal_moves)
            
        return move_uci, entropy

def get_game_status(start_fen: str, movelist: list) -> tuple:
    """
    Determines if the game has ended and returns the result.
    Returns: (ended: bool, result: str)
      result: "white" (White won), "black" (Black won), "draw" (Draw), or "ongoing"
    """
    current_fen = pyffish.get_fen("gardner", start_fen, movelist)
    
    # check legal moves
    legal = pyffish.legal_moves("gardner", current_fen, [])
    if not legal:
        in_check = pyffish.gives_check("gardner", current_fen, [])
        active_player = current_fen.split(" ")[1]
        if in_check:
            # Checkmate: active player loses, opponent wins
            return True, "black" if active_player == 'w' else "white"
        else:
            # Stalemate: draw
            return True, "draw"
            
    # check insufficient material
    insufficient = pyffish.has_insufficient_material("gardner", start_fen, movelist)
    if insufficient == (True, True): # if neither player has enough material to win
        return True, "draw"
        
    # check repetition or 50-move rule
    opt_ended, _ = pyffish.is_optional_game_end("gardner", start_fen, movelist)
    if opt_ended:
        return True, "draw"
        
    return False, "ongoing"

def play_game(agent_white : ChessAgent, agent_black : ChessAgent, max_moves=150, temperature=0.1):
    pyffish.set_option("UCI_Variant", "gardner")
    start_fen = pyffish.start_fen("gardner")
    
    movelist = []
    entropies_white = []
    entropies_black = []
    
    while True:
        # Check current game status using full move history for repetition & 50-move rules
        ended, result = get_game_status(start_fen, movelist)
        if ended:
            # TODO return also the moves, so we have a complete game log
            return result, len(movelist), entropies_white, entropies_black
            
        # Hard limit to prevent infinite loops (max_moves treated as max half-moves)
        if len(movelist) >= max_moves:
            return "draw", len(movelist), entropies_white, entropies_black
            
        current_fen = pyffish.get_fen("gardner", start_fen, movelist)
        legal = pyffish.legal_moves("gardner", current_fen, [])
        
        # Select active agent
        parts = current_fen.split(" ")
        active_player = parts[1]
        
        if active_player == 'w':
            move, ent = agent_white.select_move(current_fen, legal, temperature)
            entropies_white.append(ent)
        else:
            move, ent = agent_black.select_move(current_fen, legal, temperature)
            entropies_black.append(ent)
            
        movelist.append(move)

def run_tournament(agent1, agent2, num_games=20, max_moves=150, temperature=0.1):
    print(f"\n=== Tournament: {agent1.name} vs {agent2.name} ({num_games} games) ===")
    
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    entropies_agent1 = []
    entropies_agent2 = []
    
    for game_idx in range(num_games):
        # Alternate colors
        if game_idx % 2 == 0:
            white, black = agent1, agent2
            agent1_color = "white"
        else:
            white, black = agent2, agent1
            agent1_color = "black"
            
        winner, moves, ent_w, ent_b = play_game(white, black, max_moves, temperature)
        
        # Collect entropies
        if agent1_color == "white":
            entropies_agent1.extend(ent_w)
            entropies_agent2.extend(ent_b)
        else:
            entropies_agent1.extend(ent_b)
            entropies_agent2.extend(ent_w)
            
        if winner == "draw":
            draws += 1
            result_str = "Draw"
        elif winner == agent1_color:
            agent1_wins += 1
            result_str = f"{agent1.name} won"
        else:
            agent2_wins += 1
            result_str = f"{agent2.name} won"
            
        print(f"  Game {game_idx + 1:02d}: Winner: {result_str} | Moves: {moves}")
        
    # Calculate win rates
    total_games = num_games
    win_rate1 = agent1_wins / total_games * 100
    win_rate2 = agent2_wins / total_games * 100
    draw_rate = draws / total_games * 100
    
    # Calculate Elo difference using the standard formula
    # E_A = 1 / (1 + 10^((R_B - R_A)/400))
    # Let's approximate the performance rating difference
    # TODO idk if this is right, i need an intuition on how the elo works
    score1 = agent1_wins + 0.5 * draws
    p1 = score1 / total_games
    if p1 >= 0.99:
        elo_diff = 400
    elif p1 <= 0.01:
        elo_diff = -400
    else:
        elo_diff = -400 * np.log10((1 - p1) / p1)
        
    avg_entropy1 = np.mean(entropies_agent1) if entropies_agent1 else 0.0
    avg_entropy2 = np.mean(entropies_agent2) if entropies_agent2 else 0.0
    
    print("\n" + "="*50)
    print(" TOURNAMENT RESULTS ")
    print("="*50)
    print(f"{agent1.name:25s}: {agent1_wins} wins ({win_rate1:.1f}%)")
    print(f"{agent2.name:25s}: {agent2_wins} wins ({win_rate2:.1f}%)")
    print(f"Draws                    : {draws} ({draw_rate:.1f}%)")
    print("-"*50)
    print(f"Approx Elo Difference (Model1 - Model2): {elo_diff:+.1f}")
    print(f"Average Policy Entropy ({agent1.name}): {avg_entropy1:.4f}")
    print(f"Average Policy Entropy ({agent2.name}): {avg_entropy2:.4f}")
    print("="*50 + "\n")
    
    return {
        "agent1_wins": agent1_wins,
        "agent2_wins": agent2_wins,
        "draws": draws,
        "elo_diff": elo_diff,
        "avg_entropy1": avg_entropy1,
        "avg_entropy2": avg_entropy2
    }

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
    
    args = parser.parse_args()
    
    set_seed(args.seed)

    # Example: run two random agents one against the other for 10 games
    '''
    python src/chess/evaluate_tournament.py --agent1_type random --agent2_type random --num_games 10
    '''

    # Example: run best model against random agent for 10 games with greedy policy
    '''
    python src/chess/evaluate_tournament.py \
        --agent1_type transformer \
        --agent1_path "/home/usbt0p/TFG/experiments/test_value_refactor/20260620_013832_d4_val_trnsf_dk64_n3_value_refactor_spatial_nofact_dk64_depth3_lr3.00e-03_bs512/best_model.pth" \
        --agent1_dim 64 --agent1_blocks 3 \
        --agent1_repr "spatial" \
        --agent2_type random \
        --num_games 10 \
        --temp 0.0 \
        --device cuda
    '''

    # Example: run two best models against each other for 20 games
    '''
    python src/chess/evaluate_tournament.py --agent1_type transformer --agent1_path "../models/transformer_d4_best_model.pt" --agent1_dim 128 --agent1_blocks 6 --agent2_type transformer --agent2_path "../models/transformer_d2_best_model.pt" --agent2_dim 128 --agent2_blocks 6 --num_games 20 --temp 0.1 --device cuda
    '''
    
    # Instantiate Agent 1
    if args.agent1_type == "random":
        agent1 = RandomAgent()
    else:
        # TODO this usage of the config is wrong. we already have a src.models.transformerEncoder.EncoderConfig class and should take advantage of it.
        # this introduces complexity, coupling and duplication
        # move encoder config and training config out of the model agent if needed,
        # or make a factory or something...
        config_1 = {
            "hidden_size": 512 if "small" in args.agent1_path else 1024,
            "embed_dim": args.agent1_dim,
            "num_blocks": args.agent1_blocks,
            "representation": args.agent1_repr,
            "factorized_policy": args.agent1_factorized
        }
        agent1 = ModelAgent(args.agent1_type, args.agent1_path, config_1, device=args.device)
        
    # Instantiate Agent 2
    if args.agent2_type == "random":
        agent2 = RandomAgent()
    else:
        config_2 = {
            "hidden_size": 512 if "small" in args.agent2_path else 1024,
            "embed_dim": args.agent2_dim,
            "num_blocks": args.agent2_blocks,
            "representation": args.agent2_repr,
            "factorized_policy": args.agent2_factorized
        }
        agent2 = ModelAgent(args.agent2_type, args.agent2_path, config_2, device=args.device)
    
    # TODO similar to before, make some kind of factory for tournaments that handles the underlying agents and their configurations, but keeping them loosely coupled
    results = run_tournament(agent1, agent2, num_games=args.num_games, max_moves=args.max_moves, temperature=args.temp)

    print(results)
