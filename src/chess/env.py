import torch
import numpy as np
import pyffish
from src.chess.arena import get_game_status_with_reason
from src.chess.agents.base import FenParts, PIECE_MAP
from src.models.dataset_parser import parse_fen_to_features

class MinichessEnv:
    """Stateful wrapper around pyffish for 5x5 Gardner Minichess RL loops."""
    
    def __init__(self, opponent=None):
        # eat the stdout of pyffish.set_option
        self.opponent = opponent
        pyffish.set_option("UCI_Variant", "gardner")
        self.start_fen = pyffish.start_fen("gardner")
        self.reset()

    def reset(self):
        """Reset the environment to the initial state."""
        self.current_fen = self.start_fen
        self.movelist = []
        
        # Tracks (board, player) strings since the last irreversible move
        parts = FenParts(self.current_fen)
        self.state_history = [(parts.fen_board, parts.active_player)]
        
        if self.opponent is not None:
            import random
            self.player_color = random.choice(['w', 'b'])
            # If player is Black, White (opponent) makes first move
            if self.player_color == 'b':
                legal = self.get_legal_moves()
                opp_move, _, _ = self.opponent.select_move(self.current_fen, legal)
                self._step_raw(opp_move)
                
        return self.current_fen

    def get_legal_moves(self):
        '''Get legal moves in UCI format.'''
        return pyffish.legal_moves("gardner", self.current_fen, [])

    def _get_repetition_count(self):
        if not self.state_history:
            return 0
        # we count how many times the current (board, player) state has appeared before.
        # if it has appeared once before, the count is 1.
        # if it has appeared twice before, the count is 2.
        return max(0, self.state_history.count(self.state_history[-1]) - 1)

    def _step_raw(self, action_uci: str):
        """Applies a move and updates board state internally without executing opponent logic."""
        self.movelist.append(action_uci)
        
        # Advance state via engine
        self.current_fen = pyffish.get_fen("gardner", self.current_fen, [action_uci])
        
        # Parse internal FEN structure for clocks and repetition resets
        board, player, halfmove, _ = FenParts(self.current_fen)
        
        if halfmove == 0:
            self.state_history = []
        self.state_history.append((board, player))

    def step(self, action_uci: str):
        '''Play a move in the environment.
        Args:
            action_uci: move in UCI format
        Returns:
            fen: new FEN
            reward: reward for the move
            done: whether the game is over
        '''
        # 1. Play the agent's move
        self._step_raw(action_uci)
        current_rep = self._get_repetition_count()
        ended, outcome, reason = get_game_status_with_reason(
            self.start_fen, self.movelist, current_fen=self.current_fen, current_repetition=current_rep
        )
        
        player_active = self.current_fen.split(" ")[1]
        just_moved = 'b' if player_active == 'w' else 'w'
        
        if ended or self.opponent is None:
            reward = 0.0
            if ended:
                if outcome == "white":
                    target_color = self.player_color if self.opponent is not None else just_moved
                    reward = 1.0 if target_color == 'w' else -1.0
                elif outcome == "black":
                    target_color = self.player_color if self.opponent is not None else just_moved
                    reward = 1.0 if target_color == 'b' else -1.0
                elif outcome == "draw":
                    reward = 0.0
            return self.current_fen, reward, ended

        # 2. Opponent steps
        legal = self.get_legal_moves()
        opp_move, _, _ = self.opponent.select_move(self.current_fen, legal)
        self._step_raw(opp_move)
        
        current_rep = self._get_repetition_count()
        ended, outcome, reason = get_game_status_with_reason(
            self.start_fen, self.movelist, current_fen=self.current_fen, current_repetition=current_rep
        )
        
        reward = 0.0
        if ended:
            if outcome == "white":
                reward = 1.0 if self.player_color == 'w' else -1.0
            elif outcome == "black":
                reward = 1.0 if self.player_color == 'b' else -1.0
            elif outcome == "draw":
                reward = 0.0
                
        return self.current_fen, reward, ended


def batch_parse_fens(fens: list[str], repetitions: list[int] = None, device: str = "cpu") -> torch.Tensor:
    """
    Parses a list of FENs into a batched spatial features tensor (shape: (N, 28)).
    
    Args:
        fens (list[str]): List of FEN strings to parse.
        repetitions (list[int], optional): List of repetition counts corresponding to each FEN. Defaults to None.
        device (str, optional): Device to move the tensor to. Defaults to "cpu".
    Returns:
        torch.Tensor: Batched spatial features tensor (shape: (N, 28)).

        Each feature row contains:
        [0..24] - board pieces representation
        [25]    - repetition count
        [26]    - halfmove clock
        [27]    - active player (1 for white, 0 for black)
    """
    N = len(fens)
    features = torch.zeros((N, 28), dtype=torch.long, device=device)
    
    for i, fen in enumerate(fens):
        fen_parts = FenParts(fen)
        
        # Parse board in place
        board_features = np.full(25, 12, dtype=np.uint8)
        parse_fen_to_features(fen_parts.fen_board, PIECE_MAP, board_features)
        
        features[i, :25] = torch.from_numpy(board_features).long().to(device)
        
        rep = repetitions[i] if repetitions is not None else 0
        features[i, 25] = rep
        features[i, 26] = int(fen_parts.halfmove)
        
        active_player = 1 if fen_parts.active_player == 'w' else 0
        features[i, 27] = active_player
            
    return features
