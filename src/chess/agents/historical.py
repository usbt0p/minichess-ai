import glob
import os
import random

import torch
import torch.nn.functional as F

from src.chess.agents.base import ChessAgent
from src.models.dataset_parser import parse_fens_to_tensor, uci_to_index
from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig


class HistoricalAgent(ChessAgent):
    """
    Opponent agent that randomly loads one of the historical self-play checkpoints
    saved in the checkpoint pool to ensure the policy trains against previous versions of itself.
    This prevents exploiting weaknesses in the current policy and reward hacking.
    """
    def __init__(self, encoder_config: EncoderConfig, pool_dir: str, device: str = "cpu", name: str = "historical_agent"):
        super().__init__(name=name)
        self.encoder_config = encoder_config
        self.device = device
        self.pool_dir = pool_dir
        self.representation = encoder_config.representation
        self.model = None
        self.load_random_checkpoint()
        
    def load_random_checkpoint(self):
        """Randomly select a checkpoint from pool_dir and load it into the model."""
        if not self.pool_dir or not os.path.exists(self.pool_dir):
            raise FileNotFoundError(f"Pool directory {self.pool_dir} not found.")
        checkpoints = glob.glob(os.path.join(self.pool_dir, "*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.pool_dir}.")
        checkpoint_path = random.choice(checkpoints)

        self.model = MiniChessTransformerEncoder.from_pretrained(
            checkpoint_path, config=self.encoder_config, device=self.device
        )
        self.model.eval()

    def select_move(self, fen: str, legal_moves: list, temperature: float = 1.0, repetition: int = 0):
        """Select a move from legal_moves based on policy head probabilities."""
        if not legal_moves:
            return None, 0.0, []
        if self.model is None:
            raise ValueError("HistoricalAgent could not load model for move selection.")
        
        features = parse_fens_to_tensor([fen], [repetition], self.representation, self.device)
            
        with torch.no_grad():
            policy_logits, _ = self.model(features)
            policy_logits = policy_logits.squeeze(0)
            
        legal_indices = [uci_to_index(m) for m in legal_moves]
        legal_logits = policy_logits[legal_indices]
        
        if temperature > 0.0:
            probs = F.softmax(legal_logits / temperature, dim=-1)
            idx = torch.multinomial(probs, 1).item()
        else:
            probs = F.softmax(legal_logits, dim=-1)
            idx = torch.argmax(probs).item()
            
        best_move = legal_moves[idx]
        return best_move, 0.0, []
