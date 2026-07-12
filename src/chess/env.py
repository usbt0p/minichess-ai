import glob
import multiprocessing as mp
import os
import random
import sys

import torch
import torch.nn.functional as F
import pyffish

from src.chess.arena import get_game_status_with_reason
from src.chess.agents.base import FenParts
from src.chess.agents.random import RandomAgent
from src.chess.agents.transformer import TransformerAgent
from src.chess.agents.historical import HistoricalAgent
from src.models.dataset_parser import parse_fens_to_tensor, uci_to_index
from src.models.transformerEncoder import MiniChessTransformerEncoder


class MinichessEnv:
    """Stateful wrapper around pyffish for 5x5 Gardner Minichess RL loops."""
    
    def __init__(self, opponent=None):
        # TODO eat the stdout of pyffish.set_option
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
            self.player_color = random.choice(['w', 'b'])
            # If player is Black, White (opponent) makes first move
            if self.player_color == 'b':
                legal = self.get_legal_moves()
                opp_move, _, _ = self.opponent.select_move(self.current_fen, legal)
                self._step_raw(opp_move)
                
        return self.current_fen

    def get_legal_moves(self):
        """Get legal moves in UCI format."""
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
        """Play a move in the environment.
        Args:
            action_uci: move in UCI format
        Returns:
            fen: new FEN
            reward: reward for the move
            done: whether the game is over
        """
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
        
        # Check game state again after opponent steps since it may have ended it
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
    Kept for backward compatibility after refactors.
    """
    if repetitions is None:
        repetitions = [0] * len(fens)
    return parse_fens_to_tensor(fens, repetitions, "spatial", device)


def batch_select_moves_self_play(
    model: MiniChessTransformerEncoder,
    representation: str,
    fens: list[str],
    legal_moves_list: list[list[str]],
    repetitions: list[int],
    device="cpu",
):
    """
    Select moves in parallel/batch for self-play opponents across environment workers.

    Args:
        model: Policy-value model.
        representation (str): "spatial" or "simple".
        fens (list[str]): List of current board FENs.
        legal_moves_list (list[list[str]]): Legal moves per environment.
        repetitions (list[int]): Repetition count per environment.
        device (str): Destination device for forward pass.

    Returns:
        list[str]: Selected move per environment.
    """
    if not fens:
        return []

    features_batch = parse_fens_to_tensor(fens, repetitions, representation, device)

    with torch.no_grad():
        policy_logits, _ = model(features_batch)

    best_moves = []
    for i, legal_moves in enumerate(legal_moves_list):
        logits = policy_logits[i]
        legal_indices = [uci_to_index(m) for m in legal_moves]
        legal_logits = logits[legal_indices]

        probs = F.softmax(legal_logits, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        best_moves.append(legal_moves[idx])

    return best_moves

def _worker(conn, count, opponent_types, encoder_config, pool_dir):
    # we have to limit the thread number to 1 to prevent problemss with multithreading and
    # multiprocessing spawning lots of threads, creating performance problems and OOM
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    pyffish.set_option("UCI_Variant", "gardner")

    # Instantiate the shared self-play model if config is provided
    self_play_model = None
    if encoder_config is not None:
        self_play_model = MiniChessTransformerEncoder(encoder_config)
        self_play_model.to("cpu")
        self_play_model.eval()

    opponents = []
    for opp_type in opponent_types:
        if opp_type == "random":
            opponents.append(RandomAgent())
        elif opp_type == "self_play":
            opponents.append(
                TransformerAgent(self_play_model, encoder_config, device="cpu", use_lookahead=False)
            )
        elif opp_type == "historical":
            opponents.append(HistoricalAgent(encoder_config, pool_dir))
        else:
            raise ValueError(f"Unknown opponent type: {opp_type}")

    # Passive MinichessEnv instances with opponent=None
    envs = [MinichessEnv(opponent=None) for _ in range(count)]
    player_colors = ['w'] * count

    def handle_reset_opponent_first_move(idx):
        """
        Set up the player colors and handle first-move logic when the environment is reset.
        If the training agent is assigned Black ('b'), the opponent plays White ('w')
        and must make the first move before returning the observation state to the agent.
        """
        env = envs[idx]
        opp = opponents[idx]
        color = random.choice(['w', 'b'])
        player_colors[idx] = color
        env.player_color = color

        if color == 'b':
            # opponent is White and plays first!
            legal = env.get_legal_moves()
            opp_move, _, _ = opp.select_move(env.current_fen, legal)
            env._step_raw(opp_move)

    # Initial resets color setup
    for i in range(count):
        handle_reset_opponent_first_move(i)

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                results = []

                # Phase 1: Apply agent's move to each env
                for env, action in zip(envs, data):
                    next_fen, reward, ended = env.step(action)
                    rep = env._get_repetition_count()
                    results.append([next_fen, reward, ended, rep])

                # Phase 2: Play the opponent's move for any active env
                self_play_indices = []
                self_play_fens = []
                self_play_reps = []
                self_play_legal_moves = []

                for idx, env in enumerate(envs):
                    if not results[idx][2]: # not ended
                        opp = opponents[idx]
                        if isinstance(opp, TransformerAgent) and not opp.use_lookahead:
                            self_play_indices.append(idx)
                            self_play_fens.append(env.current_fen)
                            self_play_reps.append(env._get_repetition_count())
                            self_play_legal_moves.append(env.get_legal_moves())
                        else:
                            # Random/Historical opponents run sequentially
                            legal = env.get_legal_moves()
                            opp_move, _, _ = opp.select_move(env.current_fen, legal)
                            env._step_raw(opp_move)

                            ended, result, reason = get_game_status_with_reason(env.start_fen, env.movelist, env.current_fen, env._get_repetition_count())
                            opp_reward = 0.0
                            if ended:
                                if result == player_colors[idx]:
                                    opp_reward = 1.0
                                elif result == ("black" if player_colors[idx] == "w" else "white"):
                                    opp_reward = -1.0

                            results[idx][0] = env.current_fen
                            results[idx][1] = opp_reward
                            results[idx][2] = ended
                            results[idx][3] = env._get_repetition_count()

                # Batch select moves for all SelfPlay opponents
                if self_play_indices:
                    opp_moves = batch_select_moves_self_play(
                        self_play_model,
                        encoder_config.representation,
                        self_play_fens,
                        self_play_legal_moves,
                        self_play_reps,
                        device="cpu"
                    )

                    for idx, opp_move in zip(self_play_indices, opp_moves):
                        env = envs[idx]
                        env._step_raw(opp_move)

                        ended, result, reason = get_game_status_with_reason(env.start_fen, env.movelist, env.current_fen, env._get_repetition_count())
                        opp_reward = 0.0
                        if ended:
                            if result == player_colors[idx]:
                                opp_reward = 1.0
                            elif result == ("black" if player_colors[idx] == "w" else "white"):
                                opp_reward = -1.0

                        results[idx][0] = env.current_fen
                        results[idx][1] = opp_reward
                        results[idx][2] = ended
                        results[idx][3] = env._get_repetition_count()

                # Phase 3: If ended, reset and play first move if player is Black
                for idx, env in enumerate(envs):
                    if results[idx][2]: # ended
                        opp = opponents[idx]
                        if hasattr(opp, "load_random_checkpoint"):
                            opp.load_random_checkpoint()
                        env.reset()
                        handle_reset_opponent_first_move(idx)
                        results[idx][0] = env.current_fen
                        results[idx][3] = env._get_repetition_count()

                conn.send([tuple(r) for r in results])

            elif cmd == "reset":
                for idx, env in enumerate(envs):
                    env.reset()
                    handle_reset_opponent_first_move(idx)
                fens = [env.current_fen for env in envs]
                conn.send(fens)

            elif cmd == "get_legal_moves":
                moves = [env.get_legal_moves() for env in envs]
                conn.send(moves)

            elif cmd == "update_weights":
                clean_state_dict = {}
                for k, v in data.items():
                    if k.startswith("_orig_mod."):
                        clean_state_dict[k[10:]] = v
                    else:
                        clean_state_dict[k] = v
                self_play_model.load_state_dict(clean_state_dict)
                conn.send("ok")

            elif cmd == "close":
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] Worker error: {e}", file=sys.stderr)
    finally:
        conn.close()


class ParallelVectorEnv:
    """Vectorized environment wrapping multiple MinichessEnv instances across processes."""

    def __init__(self, num_envs, opponent_fn=None, encoder_config=None, pool_dir=None, num_workers=8, opponent_mode="random"):
        self.num_envs = num_envs
        self.num_workers = min(num_workers, num_envs)

        envs_per_worker = num_envs // self.num_workers
        rem = num_envs % self.num_workers

        self.parent_conns = []
        self.processes = []
        self.worker_sizes = []

        ctx = mp.get_context("spawn")

        start_idx = 0
        for w_idx in range(self.num_workers):
            count = envs_per_worker + (1 if w_idx < rem else 0)
            self.worker_sizes.append(count)

            # Assign opponent type for each env of this worker
            opponent_types = []
            for j in range(count):
                env_idx = start_idx + j
                if opponent_mode == "random":
                    opponent_types.append("random")
                elif opponent_mode == "self_play_pool":
                    # 80% self play, 20% historical
                    if env_idx < int(0.8 * num_envs):
                        opponent_types.append("self_play")
                    else:
                        opponent_types.append("historical")
                else:
                    raise ValueError(f"Unknown opponent mode: {opponent_mode}")

            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child_conn, count, opponent_types, encoder_config, pool_dir))
            p.daemon = True
            p.start()

            self.parent_conns.append(parent_conn)
            self.processes.append(p)
            start_idx += count

    def reset(self):
        for conn in self.parent_conns:
            conn.send(("reset", None))
        
        fens = []
        for conn in self.parent_conns:
            fens.extend(conn.recv())
        return fens

    def get_legal_moves(self):
        for conn in self.parent_conns:
            conn.send(("get_legal_moves", None))
        
        moves = []
        for conn in self.parent_conns:
            moves.extend(conn.recv())
        return moves

    def step(self, actions):
        idx = 0
        for conn, size in zip(self.parent_conns, self.worker_sizes):
            conn.send(("step", actions[idx : idx + size]))
            idx += size

        results = []
        for conn in self.parent_conns:
            results.extend(conn.recv())
        return results

    def update_opponent_weights(self, state_dict):
        for conn in self.parent_conns:
            conn.send(("update_weights", state_dict))
        for conn in self.parent_conns:
            conn.recv()

    def close(self):
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
            except IOError:
                pass
        for p in self.processes:
            p.join(timeout=1.0)
