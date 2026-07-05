import time
import torch
import torch.optim as optim
import argparse
import os
import sys
from src.chess.env import MinichessEnv, batch_parse_fens
from src.chess.agents.random import RandomAgent
from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig
from src.ppo.ppo import PPOTrainer, PPOConfig
from src.utils.utils import Tee

def train_ppo():
    parser = argparse.ArgumentParser(description="Run PPO training for Gardner Minichess.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint model weights (.pth)")
    parser.add_argument("--iterations", type=int, default=60, help="Number of training iterations")
    parser.add_argument("--log_file", type=str, default=None, help="File to log the training print outputs")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the best and last models")
    args = parser.parse_args()

    if args.log_file:
        # Redirect stdout and stderr to the log file as well as displaying on screen
        tee = Tee(args.log_file)
        sys.stdout = tee
        sys.stderr = tee

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # 1. Hyperparameters & Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running PPO training on device: {device}")
    if args.checkpoint:
        print(f"[INFO] Initializing model with checkpoint: {args.checkpoint}")
    else:
        print(f"[INFO] Initializing model from scratch.")

    ppo_config = PPOConfig(
        num_envs=128,
        rollout_steps=128,
        epochs=6,
        batch_size=256,
        lr=2e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        c1_value=0.5,
        c2_entropy=0.01,
        max_grad_norm=0.5,
        device=device
    )

    # 2. Model Config (64dk, 3 blocks/depth, 8 heads)
    encoder_config = EncoderConfig(
        embed_dim=64,
        num_heads=8,
        num_blocks=3,
        batch_size=1,
        policy_size=704,
        mlp_expand_factor=4,
        representation="spatial",
        use_factorized_policy=False,
        attn_backend="math",
        autocast_mode="none"
    )
    
    model = MiniChessTransformerEncoder(encoder_config).to(device)
    # model = torch.compile(model)

    # Load checkpoint weights if provided
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):  # in case model was compiled
                clean_state_dict[k[10:]] = v
            else:
                clean_state_dict[k] = v
        model.load_state_dict(clean_state_dict)
        print(f"[INFO] Loaded checkpoint successfully.")

    optimizer = optim.AdamW(model.parameters(), lr=ppo_config.lr)
    trainer = PPOTrainer(model, optimizer, ppo_config)

    # 3. Initialize wrapped environments
    # Each environment is wrapped with a RandomAgent opponent
    opponent = RandomAgent()
    envs = [MinichessEnv(opponent=opponent) for _ in range(ppo_config.num_envs)]
    
    current_fens = [env.reset() for env in envs]
    current_repetitions = [0] * ppo_config.num_envs
    episode_rewards = [0.0] * ppo_config.num_envs
    completed_episode_rewards = []

    # 4. Training Loop
    num_iterations = args.iterations
    print("\n=== Starting PPO Training ===")
    print(f"Total iterations: {num_iterations}")
    print(f"Environments: {ppo_config.num_envs} | Rollout steps per env: {ppo_config.rollout_steps}")
    print("---------------------------------------------------------")

    best_reward = None
    total_start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        start_time = time.time()

        # Collect trajectory rollouts
        batch = trainer.collect_rollouts(
            envs, current_fens, current_repetitions, episode_rewards, completed_episode_rewards, time=True
        )

        # Get final state observations and dones to bootstrap values
        final_obs = batch_parse_fens(current_fens, repetitions=current_repetitions, device=device)
        final_dones = torch.tensor([0.0] * ppo_config.num_envs)

        # Perform PPO optimization update
        metrics = trainer.train_step(batch, final_obs, final_dones, time=True)

        elapsed = time.time() - start_time
        
        # Calculate reward metrics
        avg_reward = 0.0
        has_episodes = len(completed_episode_rewards) > 0
        if has_episodes:
            avg_reward = sum(completed_episode_rewards) / len(completed_episode_rewards)
            # Limit the rewards window to the most recent 100 episodes
            completed_episode_rewards = completed_episode_rewards[-100:]

        print(f"Iteration {iteration:03d}/{num_iterations:03d} [{elapsed:.2f}s]")
        print(f"  Avg Episode Reward (recent 100): {avg_reward:+.3f}")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f} | Value Loss: {metrics['value_loss']:.4f} | Entropy: {metrics['entropy']:.4f}")
        print(f"  Total Loss:  {metrics['total_loss']:.4f}")

        # Save model with the best reward
        if args.save_dir and has_episodes:
            if best_reward is None or avg_reward > best_reward:
                best_reward = avg_reward
                best_model_path = os.path.join(args.save_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"  [SAVE] New best model saved to {best_model_path} with reward: {best_reward:+.3f}")

        print("---------------------------------------------------------")

    total_elapsed = time.time() - total_start_time
    print(f"\n=== PPO Training Finished ===")
    print(f"Total training time: {total_elapsed / 60:.2f} minutes ({total_elapsed:.2f} seconds)")

    if args.save_dir:
        last_model_path = os.path.join(args.save_dir, "last_model.pth")
        torch.save(model.state_dict(), last_model_path)
        print(f"[SAVE] Last model saved to {last_model_path}")

if __name__ == "__main__":
    train_ppo()
