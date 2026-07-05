import pytest
import torch
import torch.optim as optim
import pyffish
from src.chess.env import MinichessEnv, batch_parse_fens
from src.chess.agents.random import RandomAgent
from src.models.dataset_parser import uci_to_index, index_to_uci
from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig
from src.ppo.ppo import PPOTrainer, PPOConfig

@pytest.fixture(autouse=True)
def setup_pyffish():
    pyffish.set_option("UCI_Variant", "gardner")

def test_opponent_env_wrapper():
    """Verify that MinichessEnv wrapper works statefully in single-agent opponent MDP mode."""
    opponent = RandomAgent()
    env = MinichessEnv(opponent=opponent)
    
    start_fen = env.reset()
    assert start_fen == env.current_fen
    assert env.player_color in ('w', 'b')
    
    # If the player is White, the starting position is White's turn
    if env.player_color == 'w':
        assert env.current_fen.split(" ")[1] == 'w'
    else:
        # If the player is Black, the opponent must have already made a move, so it's Black's turn
        assert env.current_fen.split(" ")[1] == 'b'
        assert len(env.movelist) == 1

    # Take one legal step for the agent
    legal = env.get_legal_moves()
    assert len(legal) > 0
    next_fen, reward, ended = env.step(legal[0])
    
    # After the agent steps (and opponent steps in response), the turn is back to the agent's color
    if not ended:
        assert next_fen.split(" ")[1] == env.player_color

def test_o1_lookup_correctness():
    """Verify O(1) indexing matches original mathematical coordinates."""
    # Test random selection of normal and promotion moves
    test_cases = ["e2e3", "a1b2", "b1c3", "e4e5q", "a2b1r", "c2d1b"]
    for move in test_cases:
        idx = uci_to_index(move)
        assert index_to_uci(idx) == move

def test_ppo_trainer_rollout_and_training():
    """Verify PPOTrainer collects rollouts and successfully computes gradients."""
    device = "cpu"
    ppo_config = PPOConfig(
        num_envs=2,
        rollout_steps=4,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        device=device
    )
    
    encoder_config = EncoderConfig(
        embed_dim=8,
        num_heads=2,
        num_blocks=1,
        batch_size=1,
        policy_size=704,
        mlp_expand_factor=2,
        representation="spatial",
        use_factorized_policy=False,
        attn_backend="math",
        autocast_mode="none"
    )
    
    model = MiniChessTransformerEncoder(encoder_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=ppo_config.lr)
    trainer = PPOTrainer(model, optimizer, ppo_config)
    
    opponent = RandomAgent()
    envs = [MinichessEnv(opponent=opponent) for _ in range(ppo_config.num_envs)]
    
    current_fens = [env.reset() for env in envs]
    current_repetitions = [0] * ppo_config.num_envs
    episode_rewards = [0.0] * ppo_config.num_envs
    completed_rewards = []
    
    # 1. Collect trajectories
    batch = trainer.collect_rollouts(
        envs, current_fens, current_repetitions, episode_rewards, completed_rewards
    )
    
    assert batch.observations.shape == (4, 2, 28)
    assert batch.actions.shape == (4, 2)
    assert batch.rewards.shape == (4, 2)
    
    # 2. Train update
    final_obs = batch_parse_fens(current_fens, repetitions=current_repetitions, device=device)
    final_dones = torch.tensor([0.0] * ppo_config.num_envs)
    
    metrics = trainer.train_step(batch, final_obs, final_dones)
    
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "total_loss" in metrics
    assert metrics["total_loss"] is not None
