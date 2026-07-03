import random
from src.chess.agents.base import ChessAgent

class RandomAgent(ChessAgent):
    def __init__(self):
        super().__init__(name="RandomAgent")
        
    def select_move(self, fen: str, legal_moves: list, temperature: float = 1.0):
        if not legal_moves:
            return None, 0.0, []
        return random.choice(legal_moves), 0.0, []
