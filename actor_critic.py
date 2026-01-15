from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import (
    Categorical,
    Normal,
    
)
from torch.utils.tensorboard import SummaryWriter

from utils import (
    TaskKind,
    TaskSpec,
    infer_task_spec
)

from task_wrappers import (
    DiscretizeBoxAction,
    PadObservation,
    MountainCarEnergyShaping,
    ScaleVelocity
)

SUPPORTED_ENVS = ["CartPole-v1", "Acrobot-v1", "MountainCarContinuous-v0"]
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes: List[int]):
        super().__init__()
        sizes = [in_dim] + hidden_sizes + [out_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

@dataclass
class ActorCriticConfig:
    env_id: str = "CartPole-v1"
    discretize_continuous: bool = True
    n_bins: int = 11
    # Optimization
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    gamma: float = 0.99
    grad_clip_norm: Optional[float] = 1.0

    alpha_w: float = 0.5
    alpha_theta: float = 1.0
    entropy_coef: float = 1e-4

    # Training
    max_steps_per_episode: int = 10_000
    total_episodes: int = 500
    log_every: int = 10

    # Networks
    policy_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    value_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])

    # Unified sizes (computed from supported tasks by default)
    obs_dim_max: Optional[int] = None
    policy_out_dim: Optional[int] = None

    # Repro / device
    seed: int = 0
    device: str = "cpu"

    # Logging
    log_to_tensorboard: bool = True

def compute_unified_dims_for_assignment(n_bins: int) -> tuple[int, int]:
    obs_dim_max = 6 # max from all three 
    policy_out_dim = max(2, 3, n_bins)  # CartPole, Acrobot, discretized MountainCar
    return obs_dim_max, policy_out_dim

class UnifiedActorCriticAgent:
    def __init__(self, cfg: ActorCriticConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Infer specs for all supported envs (for unified dims)
        specs = [infer_task_spec(eid, seed=cfg.seed) for eid in SUPPORTED_ENVS]
        self.spec_map = {s.env_id: s for s in specs}
        if cfg.env_id not in self.spec_map:
            raise ValueError(f"Unsupported env_id={cfg.env_id}. Supported: {SUPPORTED_ENVS}")

        # Compute unified sizes unless provided
        obs_dim_max, policy_out_dim = compute_unified_dims_for_assignment(self.cfg.n_bins)
        self.obs_dim_max = cfg.obs_dim_max or obs_dim_max
        self.policy_out_dim = cfg.policy_out_dim or policy_out_dim

        # Build env (with obs padding)
        base_env = gym.make(cfg.env_id)

        self.eval_env = gym.make(cfg.env_id)
        
        if self.cfg.env_id == "MountainCarContinuous-v0" and cfg.discretize_continuous and isinstance(base_env.action_space, gym.spaces.Box):
            print(f"Detected mountain car, activating the reward shaping")
            base_env = ScaleVelocity(base_env)
            base_env = DiscretizeBoxAction(base_env, n_bins=cfg.n_bins)
            base_env = MountainCarEnergyShaping(base_env, gamma=self.cfg.gamma)

        
        if self.cfg.env_id == "MountainCarContinuous-v0":
            self.eval_env = ScaleVelocity(self.eval_env)
            self.eval_env = DiscretizeBoxAction(self.eval_env, n_bins=cfg.n_bins)
        
        self.env = PadObservation(base_env, obs_dim_max=self.obs_dim_max)
        self.eval_env = PadObservation(self.eval_env, obs_dim_max=self.obs_dim_max)
        self.task = self.spec_map[cfg.env_id]

        # Seeds
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env.reset(seed=cfg.seed)

        # Logging
        self.writer = None
        if cfg.log_to_tensorboard:
            self.writer = SummaryWriter(comment=f"_{cfg.env_id}")

        # Networks
        self.policy_net = MLP(self.obs_dim_max, self.policy_out_dim, cfg.policy_hidden_sizes).to(self.device)
        self.value_net = MLP(self.obs_dim_max, 1, cfg.value_hidden_sizes).to(self.device)

        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)
        self.value_opt = optim.Adam(self.value_net.parameters(), lr=cfg.value_lr)

    def save_model(self, path: str):
            """Saves the model weights to disk."""
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'value_net': self.value_net.state_dict(),
                'obs_dim_max': self.obs_dim_max
            }, path)
            print(f"Model saved to {path}")
    
    def load_transfer_model(self, path: str, freeze_hidden: bool = False):
        """
        Loads weights from a source model but RE-INITIALIZES the output layers.
        This enables transfer from Source (e.g. Acrobot) to Target (e.g. CartPole).
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        source_policy = checkpoint['policy_net']
        current_policy = self.policy_net.state_dict()
        
        pretrained_dict = {}
        for k, v in source_policy.items():
            if k not in current_policy:
                continue
                
            if v.size() != current_policy[k].size():
                continue

            # skip output layer weights
            if v.size(0) == self.policy_out_dim:
                print(f"Skipping transfer of output layer: {k}")
                continue
                
            pretrained_dict[k] = v
        
        # Overwrite current weights with filtered source weights
        current_policy.update(pretrained_dict)
        self.policy_net.load_state_dict(current_policy)

        # Explicit Re-initialization
        print("Re-initializing policy output layer...")
        for name, param in self.policy_net.named_parameters():
            if "weight" in name and param.shape[0] == self.policy_out_dim:
                nn.init.orthogonal_(param, gain=0.01) 
            elif "bias" in name and param.shape[0] == self.policy_out_dim:
                nn.init.constant_(param, 0.0)

        # Load Value Network
        source_value = checkpoint['value_net']
        current_value = self.value_net.state_dict()
        pretrained_val = {k: v for k, v in source_value.items() if k in current_value and v.size() == current_value[k].size()}
        current_value.update(pretrained_val)
        self.value_net.load_state_dict(current_value)
        
        print(f"Transfer learning weights loaded from {path}")

        # 4. Freeze Hidden Layers 
        if freeze_hidden:
            print("Freezing hidden layers...")
            for name, param in self.policy_net.named_parameters():
                if not param.shape[0] == self.policy_out_dim:
                    param.requires_grad = False
            
            # Re-init optimizer to only update the head
            self.policy_opt = optim.Adam(filter(lambda p: p.requires_grad, self.policy_net.parameters()), lr=self.cfg.policy_lr)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        self.env.close()

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        return torch.from_numpy(x).to(self.device)
    
    def _dist_and_action(self, obs_t: torch.Tensor, deterministic: bool = False):
        logits_all = self.policy_net(obs_t)

        n_valid = int(self.env.action_space.n)  
        logits = logits_all[:n_valid]
        temp = 2.0
        dist = Categorical(logits=logits/temp)
        if deterministic:
            a = int(torch.argmax(dist.probs).item())
        else:
            a = int(dist.sample().item())

        logp = dist.log_prob(torch.tensor(a, device=self.device))
        entropy = dist.entropy()
        return a, logp, entropy
    
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        obs_t = self._obs_to_tensor(obs)
        a, _, _ = self._dist_and_action(obs_t, deterministic=deterministic)
        return a

    @torch.no_grad()
    def evaluate(self, episodes: int = 10, deterministic: bool = True) -> float:
        rets = []
        for _ in range(episodes):
            obs, _ = self.eval_env.reset()
            done, total, steps = False, 0.0, 0
            while not done and steps < self.cfg.max_steps_per_episode:
                a = self.act(obs, deterministic=deterministic)
                obs, r, terminated, truncated, _ = self.eval_env.step(a)
                total += float(r)
                steps += 1
                done = terminated or truncated
            rets.append(total)
        return float(np.mean(rets))

    def train(self) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "episode_return": [],
            "episode_len": [],
            "policy_loss": [],
            "value_loss": [],
            "success": [],
            "original_return": []
        }
        success = 0
        for ep_idx in range(1, self.cfg.total_episodes + 1):
            obs, _ = self.env.reset()
            done = False
            I, ep_ret, ep_len, orig_r = 1, 0.0, 0, 0.0

            pol_losses, val_losses = [], []
            deltas, entropies, values = [], [], []

            while not done and ep_len < self.cfg.max_steps_per_episode:
  
                obs_t = self._obs_to_tensor(obs)

                # compute value
                v_s = self.value_net(obs_t).squeeze(-1)

                # Policy sample + logp
                env_action, logp, entropy = self._dist_and_action(obs_t, deterministic=False)

                next_obs, reward, terminated, truncated, info = self.env.step(env_action)
                done = terminated or truncated

                ep_ret += float(reward)
                orig_r += float(info.get("orig_reward", ep_ret))
                ep_len += 1

                next_obs_t = self._obs_to_tensor(next_obs)
                with torch.no_grad():
                    # if terminated:
                    if terminated:
                        success += 1
                        v_s_tag = torch.tensor(0.0, device=self.device)
                    else:
                        v_s_tag = self.value_net(next_obs_t).squeeze(-1)
                    td_target = torch.tensor(reward, dtype=torch.float32) + self.cfg.gamma * v_s_tag

                delta = td_target - v_s 
                I_t = torch.tensor(I, device=self.device)

                self.value_opt.zero_grad(set_to_none=True)
                
                # minimize td error
                value_loss = self.cfg.alpha_w * delta.pow(2)
                # value_loss = self.cfg.alpha_w * F.smooth_l1_loss(v_s, td_target)
                value_loss.backward()
                if self.cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.grad_clip_norm)
                self.value_opt.step()
                val_losses.append(float(value_loss.item()))

                # policy_loss = -self.cfg.alpha_theta * delta.detach() * logp  - self.cfg.entropy_coef * entropy
                policy_loss = -I * delta.detach() * logp  - self.cfg.entropy_coef * entropy
                self.policy_opt.zero_grad() 
                policy_loss.backward()

                if self.cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.grad_clip_norm)
                self.policy_opt.step()
                pol_losses.append(float(policy_loss.item()))
                
                I *= self.cfg.gamma
                obs = next_obs

                deltas.append(float(delta.detach().item()))
                entropies.append(float(entropy.detach().item()))
                values.append(float(v_s.detach().item()))
  

            history["episode_return"].append(ep_ret)
            history["original_return"].append(orig_r)
            history["episode_len"].append(ep_len)
            history["policy_loss"].append(float(np.mean(pol_losses)) if pol_losses else 0.0)
            history["value_loss"].append(float(np.mean(val_losses)) if val_losses else 0.0)
            history['success'] = success

            if (ep_idx % self.cfg.log_every) == 0:
                recent = history["episode_return"][-self.cfg.log_every:]
                recent_orig = history["original_return"][-self.cfg.log_every:]
                avg_ret = float(np.mean(recent))
                avg_ret_orig = float(np.mean(recent_orig))
                print(
                    f"env={self.cfg.env_id:>24s}  "
                    f"ep={ep_idx:5d}  "
                    f"avg_return({len(recent)})={avg_ret:8.2f}  "
                    f"org_return({len(recent)})={avg_ret_orig:8.2f}  "
                    f"policy_loss={history['policy_loss'][-1]:9.4f}  "
                    f"value_loss={history['value_loss'][-1]:9.4f} "
                    f"number of success= {history["success"]}"
                )

                if self.writer is not None:
                    self.writer.add_scalar("charts/avg_return", avg_ret, ep_idx)
                    self.writer.add_scalar("charts/episode_len", history["episode_len"][-1], ep_idx)
                    self.writer.add_scalar("losses/policy_loss", history["policy_loss"][-1], ep_idx)
                    self.writer.add_scalar("losses/value_loss", history["value_loss"][-1], ep_idx)
                    self.writer.add_scalar("debug/abs_delta", float(np.mean(np.abs(deltas))) if deltas else 0.0, ep_idx)
                    self.writer.add_scalar("debug/entropy", float(np.mean(entropies)) if entropies else 0.0, ep_idx)
                    self.writer.add_scalar("debug/value", float(np.mean(values)) if values else 0.0, ep_idx)

        return history
