import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import pyffish

from src.chess.agents.base import ChessAgent, PIECE_MAP, IllegalMoveError, FenParts
from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig
from src.models.dataset_parser import uci_to_index, index_to_uci, parse_fen_to_features

class TransformerAgent(ChessAgent):
    def __init__(self, model_path: str, config_args: dict, device: str = "cpu", name: str = None):
        if name is None:
            name = os.path.basename(model_path).replace(".pth", "").replace(".pt", "")
        super().__init__(name=name)
        self.device = device
        
        # Build EncoderConfig if dict is passed
        if isinstance(config_args, dict):
            encoder_config = EncoderConfig(
                embed_dim=config_args["embed_dim"],
                num_heads=8,
                num_blocks=config_args["num_blocks"],
                batch_size=1,
                policy_size=704,
                mlp_expand_factor=config_args.get("mlp_expand", 4),
                representation=config_args.get("representation", "simple"),
                use_factorized_policy=config_args.get("factorized_policy", False),
                attn_backend="math",
                autocast_mode="none"
            )
        elif isinstance(config_args, EncoderConfig):
            encoder_config = config_args
        else:
            raise TypeError("config_args must be a dict or EncoderConfig instance")
            
        self.representation = encoder_config.representation
        self.model = MiniChessTransformerEncoder(encoder_config)
        
        print(f"[INFO] Loading Transformer checkpoint '{model_path}' onto {device}...")
        state_dict = torch.load(model_path, map_location=device)
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."): # in case the model was saved with torch.compile
                clean_state_dict[k[10:]] = v
            else:
                clean_state_dict[k] = v
        self.model.load_state_dict(clean_state_dict)
        self.model.to(device)
        self.model.eval()

    def select_move(self, fen: str, legal_moves: list, temperature: float = 1.0, repetition: int = 0):
        """
        Select a move using the Transformer agent. Uses both policy head to select top 6 moves and
        value head to select the best move among them, so equivalent to a 1-ply search.

        Args:
            fen: The current FEN string.
            legal_moves: A list of legal moves.
            temperature: The temperature for sampling.
            repetition: The number of repetitions.

        Returns:
            The selected move.
        """
        if not legal_moves:
            return None, 0.0, []
            
        # Parse FEN
        fen_parts = FenParts(fen)
        board_features = np.full(25, 12, dtype=np.uint8)
        parse_fen_to_features(fen_parts.fen_board, PIECE_MAP, board_features)
        
        board_tensor = torch.from_numpy(board_features).long().to(self.device)
        halfmove = int(fen_parts.halfmove)
        halfmove_tensor = torch.tensor([halfmove], dtype=torch.long, device=self.device)
        repetition_tensor = torch.tensor([repetition], dtype=torch.long, device=self.device)
        
        # Build features
        if self.representation == "spatial":
            active_player = 1 if fen_parts.active_player == 'w' else 0
            active_player_tensor = torch.tensor([active_player], dtype=torch.long, device=self.device)
            features = torch.cat([board_tensor, repetition_tensor, halfmove_tensor, active_player_tensor], dim=0).unsqueeze(0)
        else:
            features = torch.cat([board_tensor, repetition_tensor, halfmove_tensor], dim=0).unsqueeze(0)
            
        with torch.no_grad():
            outputs = self.model(features)
            if len(outputs) == 5:
                policy_logits, value_pred, _, _, _ = outputs
            else:
                policy_logits, value_pred = outputs
            policy_logits = policy_logits.squeeze(0)
            
        # Mask out illegal moves
        legal_indices = [uci_to_index(m, promotions=True) for m in legal_moves]
        masked_logits = torch.full_like(policy_logits, -1e9)
        masked_logits[legal_indices] = policy_logits[legal_indices]
        
        # calculate policy entropy
        probs = F.softmax(policy_logits[legal_indices], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        
        # Extract top-6 probabilities
        move_probs = sorted(zip(legal_moves, probs.tolist()), key=lambda x: x[1], reverse=True)
        top_6 = [{"move": m, "prob": round(p, 4)} for m, p in move_probs[:6]]

        # 1-step lookahead using Value Head
        k = min(6, len(legal_moves))
        top_k_moves_info = move_probs[:k]
        top_k_moves = [m for m, p in top_k_moves_info]
        
        # Generate FENs and features for each child position
        batch_features = []
        for move in top_k_moves:
            child_fen_str = pyffish.get_fen("gardner", fen, [move])
            child_board, child_player, child_halfmove, _ = FenParts(child_fen_str)
            
            child_board_features = np.full(25, 12, dtype=np.uint8)
            parse_fen_to_features(child_board, PIECE_MAP, child_board_features)
            
            child_board_tensor = torch.from_numpy(child_board_features).long().to(self.device)
            child_halfmove_tensor = torch.tensor([child_halfmove], dtype=torch.long, device=self.device)
            child_repetition_tensor = torch.tensor([repetition], dtype=torch.long, device=self.device)
            
            if self.representation == "spatial":
                child_active_player = 1 if child_player == 'w' else 0
                child_active_player_tensor = torch.tensor([child_active_player], dtype=torch.long, device=self.device)
                child_features = torch.cat([child_board_tensor, child_repetition_tensor, child_halfmove_tensor, child_active_player_tensor], dim=0)
            else:
                child_features = torch.cat([child_board_tensor, child_repetition_tensor, child_halfmove_tensor], dim=0)
            batch_features.append(child_features)
            
        features_batch = torch.stack(batch_features, dim=0)
        
        with torch.no_grad():
            outputs = self.model(features_batch)
            if len(outputs) == 5:
                _, value_preds, _, _, _ = outputs
            else:
                _, value_preds = outputs
                    
        value_preds = value_preds.squeeze(-1)
        utilities = -value_preds
        
        if temperature > 0.0:
            val_probs = F.softmax(utilities / temperature, dim=-1)
            best_move_idx = torch.multinomial(val_probs, num_samples=1).item()
        else:
            best_move_idx = torch.argmin(value_preds).item()
            
        best_move = top_k_moves[best_move_idx]
            
        if best_move not in legal_moves:
            raise IllegalMoveError(best_move, legal_moves)
            
        return best_move, entropy, top_6

    def select_move_ppo(self, fen: str, legal_moves: list, temperature: float = 1.0, repetition: int = 0):
        """
        PPO Rollout action selection. Only works with spatial representation.
        Completely bypasses lookahead. Samples from policy head and returns
        move_uci, move_index, log_prob, value_pred.

        This is needed since PPO needs to generate trajectories using the policy and value separately as actor and critic.
        If the agent selects using search, the loss is non differentiable w.r.t the weights.
        """
        if not legal_moves:
            return None, -1, 0.0, 0.0
            
        # Parse FEN
        fen_parts = FenParts(fen)
        board_features = np.full(25, 12, dtype=np.uint8)
        parse_fen_to_features(fen_parts.fen_board, PIECE_MAP, board_features)
        
        board_tensor = torch.from_numpy(board_features).long().to(self.device)
        halfmove = int(fen_parts.halfmove)
        halfmove_tensor = torch.tensor([halfmove], dtype=torch.long, device=self.device)
        repetition_tensor = torch.tensor([repetition], dtype=torch.long, device=self.device)
        
        # Build features (PPO assumes spatial representation)
        active_player = 1 if fen_parts.active_player == 'w' else 0
        active_player_tensor = torch.tensor([active_player], dtype=torch.long, device=self.device)
        features = torch.cat([board_tensor, repetition_tensor, halfmove_tensor, active_player_tensor], dim=0).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(features)
            if len(outputs) == 5:
                policy_logits, value_pred, _, _, _ = outputs
            else:
                policy_logits, value_pred = outputs
            policy_logits = policy_logits.squeeze(0)
            value_val = value_pred.squeeze(-1).squeeze(0).item()
            
        # Mask out illegal moves
        legal_indices = [uci_to_index(m, promotions=True) for m in legal_moves]
        
        # Softmax over legal moves
        legal_logits = policy_logits[legal_indices]
        if temperature > 0.0:
            probs = F.softmax(legal_logits / temperature, dim=-1)
            # Sample using torch.multinomial
            sampled_idx = torch.multinomial(probs, 1).item()
        else:
            probs = F.softmax(legal_logits, dim=-1)
            sampled_idx = torch.argmax(probs).item()
            
        best_move = legal_moves[sampled_idx]
        best_move_idx = legal_indices[sampled_idx]
        
        mask = torch.full((704,), -1e9, device=self.device)
        mask[legal_indices] = policy_logits[legal_indices]
        if temperature > 0.0:
            probs_all = F.softmax(mask / temperature, dim=-1)
        else:
            probs_all = torch.zeros(704, device=self.device)
            probs_all[best_move_idx] = 1.0
            
        log_prob = torch.log(probs_all[best_move_idx] + 1e-9).item()
        
        return best_move, best_move_idx, log_prob, value_val
