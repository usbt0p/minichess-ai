import argparse
import logging
import os
import random
import sys
import time
from statistics import mean, stdev

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.chess.agents.heuristic import HeuristicAgent
from src.chess.agents.random import RandomAgent
from src.chess.agents.transformer import InMemoryTransformerAgent
from src.chess.arena import play_matchup
from src.chess.env import MinichessEnv, ParallelVectorEnv, batch_parse_fens
from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig
from src.ppo.ppo import PPOTrainer, PPOConfig
from src.utils.utils import Tee

# Set up logging to output to stdout so that it is captured by the Tee redirector
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("train_ppo")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for Gardner Minichess PPO training.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run PPO training for Gardner Minichess.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint model weights (.pth)")
    parser.add_argument("--iterations", type=int, default=60, help="Number of training iterations")
    parser.add_argument("--log_file", type=str, default=None, help="File to log the training print outputs")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the best and last models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tb_dir", type=str, default=None, help="Directory to save TensorBoard logs")
    parser.add_argument("--pool_dir", type=str, default=None, help="Directory for the opponent checkpoint pool")
    parser.add_argument("--opponent_mode", type=str, default="random", choices=["random", "self_play_pool"], help="Opponent type in environments")
    parser.add_argument("--eval_interval", type=int, default=5, help="Interval (iterations) to run evaluations")
    parser.add_argument("--checkpoint_save_interval", type=int, default=5, help="Interval (iterations) to save pool checkpoints")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of worker processes")
    parser.add_argument("--num_envs", type=int, default=100, help="Number of concurrent environments")
    parser.add_argument("--rollout_steps", type=int, default=100, help="Number of rollout steps per environment per iteration")
    parser.add_argument("--batch_size", type=int, default=1024, help="Minibatch size for optimization updates")
    parser.add_argument("--epochs", type=int, default=8, help="Number of optimization epochs per iteration")
    parser.add_argument("--eval_games", type=int, default=40, help="Number of games per evaluation tournament")
    return parser.parse_args()


def setup_boilerplate(args: argparse.Namespace) -> Tee:
    """
    Handle folder creation, seed setting, and Tee logging setup.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
        
    Returns:
        Tee: Instantiated Tee log redirector, or None if no log file is specified.
    """
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Redirect stdout and stderr to the log file as well as displaying on screen
    tee = None
    if args.log_file:
        tee = Tee(args.log_file)
        sys.stdout = tee
        sys.stderr = tee

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    if args.pool_dir:
        os.makedirs(args.pool_dir, exist_ok=True)

    return tee


def run_evaluation_vs_baseline(
    model: torch.nn.Module, 
    encoder_config: EncoderConfig, 
    baseline_name: str, 
    num_games: int = 60, 
    temp: float = 0.1, 
    device: str = "cpu", 
    save_log: str = None
) -> dict:
    """
    Evaluate the current training policy against a specific baseline agent.
    
    Args:
        model (torch.nn.Module): Current policy model.
        encoder_config (EncoderConfig): Model configuration.
        baseline_name (str): 'random' or 'heuristic'.
        num_games (int): Number of games to play in the matchup.
        temp (float): Exploration temperature.
        device (str): Inference device.
        save_log (str): Optional path to save detailed game-by-game JSON log.
        
    Returns:
        dict: Summary results of the evaluation matchup.
    """
    agent_model = InMemoryTransformerAgent(model, encoder_config, device=device, name="current_model")
    if baseline_name == "random":
        agent_baseline = RandomAgent()
    elif baseline_name == "heuristic":
        agent_baseline = HeuristicAgent()
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")
        
    results = play_matchup(agent_model, agent_baseline, num_games=num_games, temperature=temp, save_log=save_log)
    return results


class PPOTrainingRunner:
    """
    Orchestration class for setup, loop, checkpointing, and evaluations of Gardner Minichess PPO.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.writer = None
        self.envs = None
        self.model = None
        self.trainer = None
        
        # Track active environment state
        self.current_fens = None
        self.current_repetitions = None
        self.episode_rewards = None
        self.completed_episode_rewards = []
        self.episode_lengths = None
        self.completed_episode_lengths = []
        
        self.best_reward = None
        self.total_start_time = None

    def setup(self):
        """Initialize configurations, models, environments, optimizer, and logging writers."""
        if self.args.tb_dir:
            os.makedirs(self.args.tb_dir, exist_ok=True)
            self.writer = SummaryWriter(self.args.tb_dir)

        logger.info(f"Running PPO training on device: {self.device}")
        logger.info(f"Using random seed: {self.args.seed}")
        if self.args.checkpoint:
            logger.info(f"Initializing model with checkpoint: {self.args.checkpoint}")
        else:
            logger.info("Initializing model from scratch.")

        # 1. Hyperparameters & Config
        ppo_config = PPOConfig(
            num_envs=self.args.num_envs,
            num_workers=self.args.num_workers,
            rollout_steps=self.args.rollout_steps,
            epochs=self.args.epochs,
            batch_size=self.args.batch_size,
            lr=2e-4,
            gamma=0.99, # Gamma is the discount factor: higher means we value long term rewards more.
            # lambda trades off bias and variance of the value function. 
            # depending on how much we trust the value function we adjust lambda:
            # - lambda is close to 1: we trust the value function more and use more variance, 0 bias
            # - lambda is close to 0: we trust the value function less and use less variance.
            gae_lambda=0.75, 
            clip_eps=0.2,
            c1_value=0.5,
            c2_entropy=0.01,
            max_grad_norm=0.5,
            device=self.device
        )

        # 2. Model Config (64dk, 3 blocks/depth, 8 heads)
        self.encoder_config = EncoderConfig(
            embed_dim=64,
            num_heads=8,
            num_blocks=3,
            batch_size=1,
            policy_size=704,
            mlp_expand_factor=4,
            representation="spatial",
            use_factorized_policy=False,
            attn_backend="math",
            autocast_mode="none",
            mlp_dropout=0.1,
            mha_dropout=0.1,
            embed_dropout=0.1,
            value_head_dropout=0.1
        )
        
        # Load checkpoint weights if provided
        if self.args.checkpoint:
            self.model = MiniChessTransformerEncoder.from_pretrained(
                self.args.checkpoint, config=self.encoder_config, device=self.device
            )
            logger.info("Loaded checkpoint successfully.")
        else:
            self.model = MiniChessTransformerEncoder(self.encoder_config).to(self.device)

        # Save initial model to the pool if we are in pool mode before starting envs
        if self.args.pool_dir and self.args.opponent_mode == "self_play_pool":
            initial_pool_path = os.path.join(self.args.pool_dir, "checkpoint_0000.pth")
            torch.save(self.model.state_dict(), initial_pool_path)
            logger.info(f"[POOL] Saved initial model to pool: {initial_pool_path}")

        optimizer = optim.AdamW(self.model.parameters(), lr=ppo_config.lr)
        self.trainer = PPOTrainer(self.model, optimizer, ppo_config)

        # 3. Initialize wrapped environments
        if ppo_config.num_workers > 0:
            self.envs = ParallelVectorEnv(
                num_envs=ppo_config.num_envs, 
                opponent_fn=RandomAgent if self.args.opponent_mode == "random" else None,
                encoder_config=self.encoder_config if self.args.opponent_mode == "self_play_pool" else None,
                pool_dir=self.args.pool_dir if self.args.opponent_mode == "self_play_pool" else None,
                num_workers=ppo_config.num_workers,
                opponent_mode=self.args.opponent_mode
            )
        else:
            self.envs = MinichessEnv(opponent=RandomAgent())
        
        self.current_fens = self.envs.reset()
        self.current_repetitions = [0] * ppo_config.num_envs
        self.episode_rewards = [0.0] * ppo_config.num_envs
        self.episode_lengths = [0] * ppo_config.num_envs

    def run_evaluations(self, iteration: int):
        """
        Evaluate model against random and heuristic agents and log results to TensorBoard.
        
        Args:
            iteration (int): Current training iteration index.
        """
        logger.info(f"\n=== Running Evaluation Tournament at Iteration {iteration:03d} ===")
        eval_random_log = os.path.join(self.args.save_dir, f"eval_random_iter_{iteration:03d}.json") if self.args.save_dir else None
        results_random = run_evaluation_vs_baseline(
            self.model, self.encoder_config, "random", 
            num_games=self.args.eval_games, temp=0.1, device=self.device, save_log=eval_random_log
        )
        
        eval_heuristic_log = os.path.join(self.args.save_dir, f"eval_heuristic_iter_{iteration:03d}.json") if self.args.save_dir else None
        results_heuristic = run_evaluation_vs_baseline(
            self.model, self.encoder_config, "heuristic", 
            num_games=self.args.eval_games, temp=0.1, device=self.device, save_log=eval_heuristic_log
        )
        
        wr_random = (results_random["agent1_wins"] + 0.5 * results_random["draws"]) / float(self.args.eval_games)
        elo_random = results_random["elo_diff"]
        
        wr_heuristic = (results_heuristic["agent1_wins"] + 0.5 * results_heuristic["draws"]) / float(self.args.eval_games)
        elo_heuristic = results_heuristic["elo_diff"]
        
        logger.info(f"  [EVAL] vs Random: Winrate = {wr_random*100:.1f}%, Elo diff = {elo_random:+.1f}")
        logger.info(f"  [EVAL] vs Heuristic: Winrate = {wr_heuristic*100:.1f}%, Elo diff = {elo_heuristic:+.1f}")
        
        if self.writer:
            self.writer.add_scalar("Evaluation_Random/winrate", wr_random, iteration)
            self.writer.add_scalar("Evaluation_Random/elo", elo_random, iteration)
            self.writer.add_scalar("Evaluation_Random/mean_game_length", results_random["avg_game_length"], iteration)
            for reason, count in results_random["reasons"].items():
                self.writer.add_scalar(f"Evaluation_Random_Reasons/{reason}", count, iteration)
                
            self.writer.add_scalar("Evaluation_Heuristic/winrate", wr_heuristic, iteration)
            self.writer.add_scalar("Evaluation_Heuristic/elo", elo_heuristic, iteration)
            self.writer.add_scalar("Evaluation_Heuristic/mean_game_length", results_heuristic["avg_game_length"], iteration)
            for reason, count in results_heuristic["reasons"].items():
                self.writer.add_scalar(f"Evaluation_Heuristic_Reasons/{reason}", count, iteration)

    def train(self):
        """Run the main multi-iteration PPO trajectory collection and policy update loop."""
        # 4. Training Loop
        num_iterations = self.args.iterations
        logger.info("\n=== Starting PPO Training ===")
        logger.info(f"Total iterations: {num_iterations}")
        logger.info(f"Environments: {self.args.num_envs} | Rollout steps per env: {self.args.rollout_steps}")
        logger.info("---------------------------------------------------------")

        self.total_start_time = time.time()

        try:
            for iteration in range(1, num_iterations + 1):
                start_time = time.time()

                # Collect trajectory rollouts
                logger.info(f"\n[DEBUG] Iteration {iteration}: Collecting rollouts...")
                batch = self.trainer.collect_rollouts(
                    self.envs, self.current_fens, self.current_repetitions, self.episode_rewards, self.completed_episode_rewards,
                    episode_lengths=self.episode_lengths, completed_episode_lengths=self.completed_episode_lengths,
                    time=True,
                )

                # Get final state observations and dones to bootstrap values
                final_obs = batch_parse_fens(self.current_fens, repetitions=self.current_repetitions, device=self.device)
                final_dones = torch.tensor([0.0] * self.args.num_envs)

                # Perform PPO optimization update
                logger.info(f"[DEBUG] Iteration {iteration}: Training step...")
                metrics = self.trainer.train_step(batch, final_obs, final_dones, time=True)

                elapsed = time.time() - start_time
                
                # Calculate reward metrics
                avg_reward = 0.0
                var_reward = 0.0
                avg_len = 0.0
                has_episodes = len(self.completed_episode_rewards) > 0
                if has_episodes:
                    avg_reward = mean(self.completed_episode_rewards)
                    if len(self.completed_episode_rewards) > 1:
                        var_reward = stdev(self.completed_episode_rewards) ** 2
                    # Limit the rewards window to the most recent 100 episodes
                    self.completed_episode_rewards = self.completed_episode_rewards[-100:]
                if len(self.completed_episode_lengths) > 0:
                    avg_len = mean(self.completed_episode_lengths)
                    self.completed_episode_lengths = self.completed_episode_lengths[-100:]

                logger.info(f"Iteration {iteration:03d}/{num_iterations:03d} [{elapsed:.2f}s]")
                logger.info(f"  Avg Episode Reward (recent 100): {avg_reward:+.3f} (variance: {var_reward:.4f})")
                logger.info(f"  Policy Loss: {metrics['policy_loss']:.4f} | Value Loss: {metrics['value_loss']:.4f} | Entropy: {metrics['entropy']:.4f}")
                logger.info(f"  Total Loss:  {metrics['total_loss']:.4f}")

                # Tensorboard logs
                if self.writer:
                    self.writer.add_scalar("PPO/policy_loss", metrics['policy_loss'], iteration)
                    self.writer.add_scalar("PPO/value_loss", metrics['value_loss'], iteration)
                    self.writer.add_scalar("PPO/entropy", metrics['entropy'], iteration)
                    self.writer.add_scalar("PPO/total_loss", metrics['total_loss'], iteration)
                    self.writer.add_scalar("PPO/avg_reward", avg_reward, iteration)
                    self.writer.add_scalar("PPO/reward_variance", var_reward, iteration)
                    if len(self.completed_episode_lengths) > 0:
                        self.writer.add_scalar("PPO/avg_game_length", avg_len, iteration)

                # Save model with the best reward
                if self.args.save_dir and has_episodes:
                    if self.best_reward is None or avg_reward > self.best_reward:
                        self.best_reward = avg_reward
                        best_model_path = os.path.join(self.args.save_dir, "best_model.pth")
                        torch.save(self.model.state_dict(), best_model_path)
                        logger.info(f"  [SAVE] New best model saved to {best_model_path} with reward: {self.best_reward:+.3f}")

                # Update worker opponents' weights if in self play mode
                if hasattr(self.envs, "update_opponent_weights") and self.args.opponent_mode == "self_play_pool":
                    self.envs.update_opponent_weights(self.model.state_dict())

                # Save checkpoint to the pool periodically
                if self.args.pool_dir and self.args.opponent_mode == "self_play_pool" and iteration % self.args.checkpoint_save_interval == 0:
                    pool_path = os.path.join(self.args.pool_dir, f"checkpoint_{iteration:04d}.pth")
                    torch.save(self.model.state_dict(), pool_path)
                    logger.info(f"  [POOL] Saved checkpoint to pool: {pool_path}")

                # Run evaluation tournament
                if iteration % self.args.eval_interval == 0:
                    self.run_evaluations(iteration)

                logger.info("---------------------------------------------------------")

            total_elapsed = time.time() - self.total_start_time
            logger.info("\n=== PPO Training Finished ===")
            logger.info(f"Total training time: {total_elapsed / 60:.2f} minutes ({total_elapsed:.2f} seconds)")

            if self.args.save_dir:
                last_model_path = os.path.join(self.args.save_dir, "last_model.pth")
                torch.save(self.model.state_dict(), last_model_path)
                logger.info(f"[SAVE] Last model saved to {last_model_path}")
        finally:
            logger.info("[INFO] Closing environments...")
            if hasattr(self.envs, "close"):
                self.envs.close()
            if self.writer:
                self.writer.close()


def train_ppo():
    """Main function parsing arguments, setting up logs, and running PPO training."""
    args = parse_args()
    setup_boilerplate(args)
    
    runner = PPOTrainingRunner(args)
    runner.setup()
    runner.train()


if __name__ == "__main__":
    train_ppo()
