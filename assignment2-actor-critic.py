from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from time import time

@dataclass
class ActorCriticConfig:
    # Optimization
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    
    gamma: float = 0.996
    grad_clip_norm: Optional[float] = 1.0

    alpha_w: float = 0.5
    alpha_teta: float = 1
    # Variance reduction
    normalize_returns: bool = True

    # Baseline
    value_coef: float = 1.0

    # Training loop
    # batch_episodes: int = 10       
    max_steps_per_episode: int = 10_000
    total_episodes: int = 750
    log_every: int = 10

    # Networks
    policy_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    value_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])

    # Repro
    seed: int = 0
    device: str = "cpu"

    # Logging
    log_to_tensorboard: bool = True


class MLP(nn.Module):
    """Policy network for discrete action spaces: outputs action logits."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int]):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Returns logits for a Categorical distribution
        return self.net(obs)
    
class ActorCriticAgent:
    def __init__(self, env: gym.Env, config: ActorCriticConfig):
        self.env = env
        self.cfg = config
        self.device = torch.device(config.device)

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.env.reset(seed=config.seed)

        # Logging
        if config.log_to_tensorboard:
            self.writer = SummaryWriter()



        # Validate spaces (this implementation is for Discrete actions)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("This implementation supports only gymnasium.spaces.Discrete action spaces.")

        # Flatten observation if it's Box/ndarray-like
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError("This implementation expects a Box observation space (continuous vector).")

        obs_dim = int(np.prod(obs_space.shape))
        act_dim = env.action_space.n

        self.policy_network = MLP(obs_dim, act_dim, self.cfg.policy_hidden_sizes).to(self.device)
        self.value_network = MLP(obs_dim, 1, self.cfg.value_hidden_sizes).to(self.device)

        self.policy_opt = optim.Adam(self.policy_network.parameters(), lr=self.cfg.policy_lr)
        self.value_opt = optim.Adam(self.value_network.parameters(), lr=self.cfg.value_lr)
    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)  # flatten
        return torch.from_numpy(obs).to(self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        obs_t = self._obs_to_tensor(obs)
        logits = self.policy_network(obs_t)
        dist = Categorical(logits=logits)
        if deterministic:
            return int(torch.argmax(dist.probs).item())
        return int(dist.sample().item())
    
    @torch.no_grad()
    def evaluate(self, episodes: int = 10, deterministic: bool = True) -> float:
        returns = []
        for _ in range(episodes):
            obs, _ = self.env.reset()
            done = False
            total = 0.0
            steps = 0
            while not done and steps < self.cfg.max_steps_per_episode:
                a = self.act(obs, deterministic=deterministic)
                obs, r, terminated, truncated, _ = self.env.step(a)
                total += float(r)
                steps += 1
                done = terminated or truncated
            returns.append(total)
        return float(np.mean(returns))

    def train(self) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "episode_return": [],
            "episode_len": [],
            "policy_loss": [],
            "value_loss": [],
        }

        for ep_idx in range(1, self.cfg.total_episodes + 1):
            obs, _ = self.env.reset()
            done = False
            I, ep_return, ep_len = 1, 0.0, 0
            pol_losses: List[float] = []
            val_losses: List[float] = []
            deltas, entropies, values = [],[] ,[]
            while not done and (ep_len < self.cfg.max_steps_per_episode):
                obs_t = self._obs_to_tensor(obs)
                # compute value
                v_s = self.value_network(obs_t).squeeze(-1)

                # get actions
                logits = self.policy_network(obs_t)
                dist = Categorical(logits=logits)
                sample_a_t = dist.sample()
                ln_p = dist.log_prob(sample_a_t)
                entropy = dist.entropy()   
                action = int(sample_a_t.item())

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_return += float(reward)
                ep_len += 1
                next_obs_as_tensor = self._obs_to_tensor(next_obs)
                with torch.no_grad():
                    if terminated:
                        v_s_tag = torch.tensor(0.0, device=self.device)
                    else:
                        v_s_tag = self.value_network(next_obs_as_tensor).squeeze(-1)
                    td_target = torch.tensor(reward, dtype=torch.float32) + self.cfg.gamma*v_s_tag

                delta =  td_target - v_s
                I_t = torch.tensor(I, device=self.device)

                self.value_opt.zero_grad(set_to_none=True)
                # minimize td error
                value_loss = self.cfg.alpha_w*delta.pow(2)
                value_loss.backward()

                self.value_opt.step()
                val_losses.append(float(value_loss.item()))

                # maximize reward
                policy_loss = -self.cfg.alpha_teta*delta.detach()*ln_p - 0.001*entropy

                self.policy_opt.zero_grad()
                policy_loss.backward()

                self.policy_opt.step()
                pol_losses.append(float(policy_loss.item()))

                I *= self.cfg.gamma
                obs = next_obs
                
                deltas.append(delta.detach().item())
                entropies.append(entropy.detach().item())
                values.append(v_s.detach().item())
            history["episode_return"].append(ep_return)
            history["episode_len"].append(ep_len)
            history["policy_loss"].append(float(np.mean(pol_losses)) if pol_losses else 0.0)
            history["value_loss"].append(float(np.mean(val_losses)) if val_losses else 0.0)

            if (ep_idx % self.cfg.log_every) == 0:
                recent = history["episode_return"][-self.cfg.log_every:]
                avg_ret = float(np.mean(recent))
                print(
                    f"episodes={ep_idx:5d}  "
                    f"avg_return({len(recent)})={avg_ret:8.3f}  "
                    f"policy_loss={history['policy_loss'][-1]:8.4f}  "
                    f"value_loss={history['value_loss'][-1]:8.4f}"
                )
                if self.cfg.log_to_tensorboard:
                    self.writer.add_scalar("charts/avg_return", avg_ret, ep_idx)
                    self.writer.add_scalar("losses/policy_loss", history["policy_loss"][-1], ep_idx)
                    self.writer.add_scalar("losses/value_loss", history["value_loss"][-1], ep_idx)
                    self.writer.add_scalar("charts/episode_len", history["episode_len"][-1], ep_idx)
                    self.writer.add_scalar("debug/abs_delta", np.mean(np.abs(deltas)), ep_idx)
                    self.writer.add_scalar("debug/entropy", np.mean(entropies), ep_idx)
                    self.writer.add_scalar("debug/value", np.mean(values), ep_idx)

        return history
    

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    cfg = ActorCriticConfig(
        gamma=0.99,
        # policy_lr=3e-4,
        # value_lr=1e-3,        
        total_episodes=470,
        log_every=10,
        seed=0,
        device="cpu",
        log_to_tensorboard=True,
    )

    agent = ActorCriticAgent(env, cfg)
    t0 = time()
    agent.train()
    print(f"Training time: {time() - t0:.2f} seconds")
    print("Evaluation avg return:", agent.evaluate(episodes=20, deterministic=True))
    env.close()
