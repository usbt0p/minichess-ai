import os

import torch
import torch.nn.functional as F
import pyffish

from src.chess.agents.base import ChessAgent, IllegalMoveError
from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig
from src.models.dataset_parser import uci_to_index, parse_fens_to_tensor


class TransformerAgent(ChessAgent):
    """
    ChessAgent that uses a trained MiniChessTransformerEncoder for move selection.
    Exposes 2 methods for move selection:
    1. `select_move`: Uses a 1-step lookahead using the value head among the top-k moves suggested by the policy head. Can't be used directly in PPO.
    2. `select_move_ppo`: Samples from the policy head directly, without lookahead. Can be used directly in PPO.
    """

    def __init__(
        self, model_path: str, config_args: dict, device: str = "cpu", name: str = None
    ):
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
                autocast_mode="none",
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
            if k.startswith(
                "_orig_mod."
            ):  # in case the model was saved with torch.compile
                clean_state_dict[k[10:]] = v
            else:
                clean_state_dict[k] = v
        self.model.load_state_dict(clean_state_dict)
        self.model.to(device)
        self.model.eval()

    def select_move(
        self, fen: str, legal_moves: list, temperature: float = 1.0, repetition: int = 0
    ):
        """
        Select a move using the Transformer agent. Uses both policy head to select top 6 moves and
        value head to select the best move among them, so equivalent to a 1-ply search.

        Args:
            fen: The current FEN string.
            legal_moves: A list of legal moves.
            temperature: The temperature for sampling.
            repetition: The number of repetitions.

        Returns:
            The selected move, policy entropy, and top-6 moves details.
        """
        if not legal_moves:
            return None, 0.0, []

        features = parse_fens_to_tensor(
            [fen], [repetition], self.representation, self.device
        )

        with torch.no_grad():
            policy_logits, _ = self.model(features)
            policy_logits = policy_logits.squeeze(0)

        # Mask out illegal moves
        legal_indices = [uci_to_index(m, promotions=True) for m in legal_moves]
        masked_logits = torch.full_like(policy_logits, -1e9)
        masked_logits[legal_indices] = policy_logits[legal_indices]

        # calculate policy entropy
        probs = F.softmax(policy_logits[legal_indices], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

        # Extract top-6 probabilities
        move_probs = sorted(
            zip(legal_moves, probs.tolist()), key=lambda x: x[1], reverse=True
        )
        top_6 = [{"move": m, "prob": round(p, 4)} for m, p in move_probs[:6]]

        # 1-step lookahead using Value Head
        k = min(6, len(legal_moves))
        top_k_moves_info = move_probs[:k]
        top_k_moves = [m for m, p in top_k_moves_info]

        # Generate FENs and features for each child position
        batch_fens = []
        batch_reps = []
        for move in top_k_moves:
            child_fen_str = pyffish.get_fen("gardner", fen, [move])
            batch_fens.append(child_fen_str)
            batch_reps.append(repetition)

        features_batch = parse_fens_to_tensor(
            batch_fens, batch_reps, self.representation, self.device
        )

        # get values for each of the top-k moves
        with torch.no_grad():
            _, value_preds = self.model(features_batch)

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

    def select_move_ppo(
        self, fen: str, legal_moves: list, temperature: float = 1.0, repetition: int = 0
    ):
        """
        PPO Rollout action selection. Only works with spatial representation.
        Completely bypasses lookahead. Samples from policy head and returns
        move_uci, move_index, log_prob, value_pred.

        This is needed since PPO needs to generate trajectories using the policy and value separately as actor and critic.
        If the agent selects using search, the loss is non differentiable w.r.t the weights.
        
        Returns: 
            - best_move: The best move selected by the agent.
            - best_move_idx: The index of the best move.
            - log_prob: The log probability of the best move.
            - value_val: The value prediction for the best move.
        """
        if not legal_moves:
            return None, -1, 0.0, 0.0

        # PPO assumes spatial representation
        features = parse_fens_to_tensor([fen], [repetition], "spatial", self.device)

        with torch.no_grad():
            policy_logits, value_pred = self.model(features)
            policy_logits = policy_logits.squeeze(0)
            value_val = value_pred.squeeze(-1).squeeze(0).item()

        # Mask out illegal moves
        legal_indices = [uci_to_index(m, promotions=True) for m in legal_moves]

        # Softmax over legal moves
        legal_logits = policy_logits[legal_indices]
        if temperature > 0.0:
            probs = F.softmax(legal_logits / temperature, dim=-1)
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


class InMemoryTransformerAgent(ChessAgent):
    """
    ChessAgent wrapper for in-memory model instances, primarily used in PPO environments and evaluations.
    Bypasses value-lookahead search for fast evaluation and training rollouts.
    """

    def __init__(self, model, encoder_config, device="cpu", name="current_model"):
        super().__init__(name=name)
        self.model = model
        self.device = device
        self.representation = encoder_config.representation

    def select_move(
        self, fen: str, legal_moves: list, temperature: float = 0.1, repetition: int = 0
    ):
        """Select a move by sampling from the policy head directly (no search lookahead)."""
        if not legal_moves:
            return None, 0.0, []

        features = parse_fens_to_tensor(
            [fen], [repetition], self.representation, self.device
        )

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
