import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import pyffish

from src.chess.agents.base import ChessAgent, PIECE_MAP, IllegalMoveError, FenParts
from src.models.fnnPromotionMasking import BaselineNet
from src.models.dataset_parser import uci_to_index, parse_fen_to_features

class MLPAgent(ChessAgent):
    def __init__(self, model_path: str, hidden_size: int = 512, result_mode: str = "regression", device: str = "cpu", name: str = None):
        if name is None:
            name = os.path.basename(model_path).replace(".pth", "").replace(".pt", "")
        super().__init__(name=name)
        self.device = device
        self.model = BaselineNet(hidden_size=hidden_size, result_mode=result_mode)
        
        print(f"[INFO] Loading MLP checkpoint '{model_path}' onto {device}...")
        state_dict = torch.load(model_path, map_location=device)
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                clean_state_dict[k[10:]] = v
            else:
                clean_state_dict[k] = v
        self.model.load_state_dict(clean_state_dict)
        self.model.to(device)
        self.model.eval()

    def select_move(self, fen: str, legal_moves: list, temperature: float = 1.0):
        if not legal_moves:
            return None, 0.0, []
            
        # Parse FEN
        fen_parts = FenParts(fen)
        board_features = np.full(25, 12, dtype=np.uint8)
        parse_fen_to_features(fen_parts.fen_board, PIECE_MAP, board_features)
        
        board_tensor = torch.from_numpy(board_features).long().to(self.device)
        
        # Create one-hot encoding of the board
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
            
        # Mask out illegal moves
        masked_logits = torch.full_like(policy_logits, -1e9)
        masked_logits[legal_indices] = policy_logits[legal_indices]
        
        # Calculate policy entropy
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
            child_fen = FenParts(child_fen_str)
            
            child_board_features = np.full(25, 12, dtype=np.uint8)
            parse_fen_to_features(child_fen.fen_board, PIECE_MAP, child_board_features)
            child_board_tensor = torch.from_numpy(child_board_features).long().to(self.device)
            
            child_one_hot = torch.zeros((25, 13), dtype=torch.float32, device=self.device)
            child_one_hot.scatter_(1, child_board_tensor.unsqueeze(1), 1.0)
            child_features = child_one_hot.flatten()
            batch_features.append(child_features)
            
        features_batch = torch.stack(batch_features, dim=0)
        
        with torch.no_grad():
            _, value_preds = self.model(features_batch, mask=None)
            
        value_preds = value_preds.squeeze(-1) # remove extra batch dim
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
