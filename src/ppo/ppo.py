import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from src.chess.env import batch_parse_fens
from src.models.dataset_parser import uci_to_index, index_to_uci
from src.utils.utils import time_this

@dataclass
class RolloutBuffer:
    observations: torch.Tensor  # (T, N, 28)
    actions: torch.Tensor       # (T, N)
    logprobs: torch.Tensor      # (T, N)
    rewards: torch.Tensor       # (T, N)
    dones: torch.Tensor         # (T, N)
    values: torch.Tensor        # (T, N)
    masks: torch.Tensor         # (T, N, 704)

@dataclass
class PPOConfig:
    num_envs: int = 16
    rollout_steps: int = 128
    epochs: int = 4
    batch_size: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    c1_value: float = 0.5
    c2_entropy: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cuda"

class PPOTrainer:
    def __init__(self, model, optimizer, config: PPOConfig):
        self.model = model
        self.optimizer = optimizer
        self.config = config

    @time_this
    def collect_rollouts(self, envs, current_fens, current_repetitions, episode_rewards, completed_episode_rewards, print_breakdown=True):
        """
        Collect trajectories by running the vectorized environment loop for T steps.
        Updates in-place: current_fens, current_repetitions, episode_rewards, completed_episode_rewards.
        """
        import time
        self.model.eval()
        T = self.config.rollout_steps
        N = self.config.num_envs
        device = self.config.device

        # Allocate trajectory buffers
        obs_buf = torch.zeros((T, N, 28), dtype=torch.long)
        action_buf = torch.zeros((T, N), dtype=torch.long)
        logprob_buf = torch.zeros((T, N))
        reward_buf = torch.zeros((T, N))
        done_buf = torch.zeros((T, N))
        value_buf = torch.zeros((T, N))
        mask_buf = torch.zeros((T, N, 704), dtype=torch.bool)

        # we'll be doing some timing to optimize bottlenecks
        # sadly, this means we have to do cuda.synchronize() at multiple points
        t_gpu = 0.0
        t_cpu = 0.0

        for step in range(T):
            # 1. Batched feature extraction from current FENs
            t0 = time.time()
            features = batch_parse_fens(current_fens, repetitions=current_repetitions, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            t_gpu += time.time() - t0

            # 2. Batched model inference to select the next actions
            t0 = time.time()
            obs_buf[step] = features.cpu()
            with torch.no_grad():
                outputs = self.model(features)
                if len(outputs) == 5:
                    policy_logits, value_pred, _, _, _ = outputs
                else:
                    policy_logits, value_pred = outputs
                # shape: (N, 704) and (N, 1)
                policy_logits = policy_logits.to(device)
                value_pred = value_pred.squeeze(-1).to(device)

            value_buf[step] = value_pred.cpu()
            if device == "cuda":
                torch.cuda.synchronize()
            t_gpu += time.time() - t0

            # 3. Batch action selection across all environments
            t1 = time.time()
            legal_moves_batch = []
            for i, env in enumerate(envs):
                legal_moves = env.get_legal_moves()
                # if an env terminates, reset it and start a new episode
                if not legal_moves:
                    current_fens[i] = env.reset()
                    current_repetitions[i] = 0
                    episode_rewards[i] = 0.0
                    legal_moves = env.get_legal_moves()
                legal_moves_batch.append(legal_moves)

            # Construct batched mask on CPU first
            batched_mask = torch.zeros((N, 704), dtype=torch.bool)
            for i, legal_moves in enumerate(legal_moves_batch):
                legal_indices = [uci_to_index(m) for m in legal_moves]
                batched_mask[i, legal_indices] = True
            
            mask_buf[step] = batched_mask

            # Move mask to device and perform batched logits masking
            batched_mask = batched_mask.to(device)
            masked_logits = torch.full((N, 704), -1e9, device=device)
            masked_logits = torch.where(batched_mask, policy_logits, masked_logits)

            probs = torch.softmax(masked_logits, dim=-1)

            # Batched action sampling and log probability computation on device
            action_indices = torch.multinomial(probs, 1).squeeze(-1) # shape: (N,)
            action_buf[step] = action_indices.cpu()

            action_probs = probs.gather(1, action_indices.unsqueeze(-1)).squeeze(-1)
            logprob_buf[step] = torch.log(action_probs + 1e-9).cpu()

            # 4. Step each environment sequentially
            for i, env in enumerate(envs):
                action_idx = action_indices[i].item()
                move_uci = index_to_uci(action_idx)
                next_fen, reward, ended = env.step(move_uci)

                reward_buf[step, i] = reward
                done_buf[step, i] = float(ended)
                episode_rewards[i] += reward

                if ended:
                    completed_episode_rewards.append(episode_rewards[i])
                    current_fens[i] = env.reset()
                    current_repetitions[i] = 0
                    episode_rewards[i] = 0.0
                else:
                    current_fens[i] = next_fen
                    current_repetitions[i] = env._get_repetition_count()
            t_cpu += time.time() - t1

        if print_breakdown:
            print(f"\t   [collect_rollouts] GPU/Parsing/Inference: {t_gpu:.3f}s | CPU Env stepping/sampling: {t_cpu:.3f}s")

        return RolloutBuffer(
            observations=obs_buf,
            actions=action_buf,
            logprobs=logprob_buf,
            rewards=reward_buf,
            dones=done_buf,
            values=value_buf,
            masks=mask_buf
        )

    @time_this
    def train_step(self, batch: RolloutBuffer, next_obs: torch.Tensor, next_dones: torch.Tensor, print_breakdown=True) -> dict:
        """
        Perform PPO training epoch steps.
        """
        import time
        self.model.train()
        T = self.config.rollout_steps
        N = self.config.num_envs
        device = self.config.device

        # 1. Compute bootstrap value for the final step
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model(next_obs.to(device))
            if len(outputs) == 5:
                _, next_values, _, _, _ = outputs
            else:
                _, next_values = outputs
            next_values = next_values.squeeze(-1).cpu()  # (N,)
        if device == "cuda":
            torch.cuda.synchronize()
        t_gpu_bootstrap = time.time() - t0

        # 2. Generalized Advantage Estimation (GAE)
        t1 = time.time()
        advantages = torch.zeros((T, N))
        lastgaelam = 0
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - next_dones.cpu()
                nextvalues = next_values
            else:
                nextnonterminal = 1.0 - batch.dones[t + 1]
                nextvalues = batch.values[t + 1]
            delta = batch.rewards[t] + self.config.gamma * nextvalues * nextnonterminal - batch.values[t]
            advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + batch.values

        # Flatten trajectory buffers
        b_obs = batch.observations.reshape(-1, 28)
        b_actions = batch.actions.reshape(-1)
        b_logprobs = batch.logprobs.reshape(-1)
        b_masks = batch.masks.reshape(-1, 704)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = batch.values.reshape(-1)

        # Standardize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        t_cpu_gae = time.time() - t1

        # 3. PPO Optimization Epochs
        t2 = time.time()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        updates_count = 0

        dataset_size = T * N
        for epoch in range(self.config.epochs): # TODO actually this matters quite a lot
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.config.batch_size):
                end = start + self.config.batch_size
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx].to(device)
                mb_actions = b_actions[mb_idx].to(device)
                mb_old_logprobs = b_logprobs[mb_idx].to(device)
                mb_masks = b_masks[mb_idx].to(device)
                mb_advantages = b_advantages[mb_idx].to(device)
                mb_returns = b_returns[mb_idx].to(device)

                # Forward pass
                outputs = self.model(mb_obs)
                if len(outputs) == 5:
                    policy_logits, value_pred, _, _, _ = outputs
                else:
                    policy_logits, value_pred = outputs
                value_pred = value_pred.squeeze(-1)

                # Mask logits to calculate log probability of actions
                masked_logits = torch.full_like(policy_logits, -1e9)
                masked_logits = torch.where(mb_masks, policy_logits, masked_logits)
                
                probs = torch.softmax(masked_logits, dim=-1)
                
                # New log probability of chosen actions
                action_probs = probs.gather(1, mb_actions.unsqueeze(-1)).squeeze(-1)
                new_logprobs = torch.log(action_probs + 1e-9)

                # Policy entropy
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()

                # Probability ratio. Instead of division, subtract in log space and then exponentiate
                logratio = new_logprobs - mb_old_logprobs
                ratio = torch.exp(logratio)

                # Clipped surrogate objective using the two surrogates
                # policy loss = -E_t[ min(ratio_t * A_t, clip(ratio_t, 1-eps, 1+eps) * A_t) ]

                # advantage is the expected reward of taking an action minus the "baseline value"
                # surr1 is prob ratio (likelyhood of the action under new policy / likelyhood of action under old policy) times the advantage
                # surr2 is the same but ratio is clipped between 1-eps and 1+eps
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss: simple MSE
                value_loss = F.mse_loss(value_pred, mb_returns)

                # Total loss
                loss = policy_loss + self.config.c1_value * value_loss - self.config.c2_entropy * entropy

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                total_loss += loss.item()
                updates_count += 1
        if device == "cuda":
            torch.cuda.synchronize()
        t_gpu_opt = time.time() - t2

        if print_breakdown:
            print(f"\t   [train_step] GPU bootstrap: {t_gpu_bootstrap:.3f}s | CPU GAE: {t_cpu_gae:.3f}s | GPU Optimization: {t_gpu_opt:.3f}s")

        avg_policy_loss = total_policy_loss / updates_count
        avg_value_loss = total_value_loss / updates_count
        avg_entropy_loss = total_entropy_loss / updates_count
        avg_total_loss = total_loss / updates_count

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy_loss,
            "total_loss": avg_total_loss
        }
