import random
import numpy as np
from src.chess.agents.base import ChessAgent, PIECE_MAP, FenParts
from src.models.dataset_parser import parse_fen_to_features

PIECE_VALUES = {
    0: 1, 1: 3, 2: 3, 3: 5, 4: 9, 5: 1000,
    6: 1, 7: 3, 8: 3, 9: 5, 10: 9, 11: 1000
}

class HeuristicAgent(ChessAgent):
    def __init__(self):
        super().__init__(name="HeuristicAgent")
        
    def select_move(self, fen: str, legal_moves: list, *args, **kwargs):
        if not legal_moves:
            return None, 0.0, []
            
        fen_parts = FenParts(fen)
        board_features = np.full(25, 12, dtype=np.uint8)
        parse_fen_to_features(fen_parts.fen_board, PIECE_MAP, board_features)

        best_moves = []
        best_score = -1
        
        for move in legal_moves:
            file_to = ord(move[2]) - ord('a')
            rank_to = int(move[3]) - 1
            idx_to = rank_to * 5 + file_to
            
            target_piece = board_features[idx_to]
            if target_piece != 12:
                score = PIECE_VALUES.get(int(target_piece), 0)
            else:
                score = 0
                
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
                
        selected_move = random.choice(best_moves)
        return selected_move, 0.0, []
