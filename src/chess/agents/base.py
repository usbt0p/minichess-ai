from abc import ABC, abstractmethod
from dataclasses import dataclass

# Mapping of pieces for FEN parsing
PIECE_MAP = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}

class IllegalMoveError(Exception):
    def __init__(self, move, legal_moves=None, message="Illegal move encountered"):
        self.message = message
        self.move = move
        self.legal_moves = legal_moves
        super().__init__(
            f"{self.message}: {self.move}. "
            f"Legal moves: {len(legal_moves) if legal_moves is not None else 'unknown'}"
        )

@dataclass
class FenParts:
    """
    Parses the FEN string into its components.
    fen_full: full FEN string
    fen_board: board representation by rows
    active_player: w for white to move, b for black to move
    halfmove: number of halfmoves since last capture or pawn move
    fullmove: incremented by 1 after black moves
    """
    fen_full: str
    fen_board: str
    active_player: str 
    halfmove: int 
    fullmove: int 
    
    def __init__(self, fen: str):
        self.fen_full = fen
        parts = fen.split(" ") 
        self.fen_board = parts[0]
        self.active_player = parts[1]
        # intermediate attributes like castling and en passant not useful for us
        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove = int(parts[5]) if len(parts) > 5 else 0

class ChessAgent(ABC):
    """Abstract base class for a chess agent"""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_move(self, fen: str, legal_moves: list, temperature: float = 1.0):
        """Selects a move from the legal moves.
        Returns:
            tuple: (selected_move, entropy, top_6_moves_probs)
        """
        pass
