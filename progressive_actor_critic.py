from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils import infer_task_spec

from task_wrappers import (
    DiscretizeBoxAction,
    PadObservation,
    MountainCarEnergyShaping,
    ScaleVelocity
)

from actor_critic import (
    ActorCriticConfig,
    SUPPORTED_ENVS,
    compute_unified_dims_for_assignment
)


class ProgressiveColumn(nn.Module):
    """
    A single column in a Progressive Neural Network.
    Can extract intermediate hidden activations for lateral connections.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes: List[int]):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Build layers separately to allow intermediate access
        self.layers = nn.ModuleList()
        sizes = [in_dim] + hidden_sizes + [out_dim]
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning only final output."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply ReLU to all but last layer
                x = F.relu(x)
        return x
    
    def forward_with_intermediates(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass that also returns hidden layer activations (post-ReLU).
        Returns: (final_output, [h1, h2, ...]) where hi is the activation after layer i.
        """
        intermediates = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Hidden layers
                x = F.relu(x)
                intermediates.append(x)
        return x, intermediates


class ProgressiveTargetColumn(nn.Module):
    """
    Target column in a Progressive Neural Network that receives lateral connections
    from frozen source columns.
    
    From the paper (Rusu et al., 2016), the forward pass for layer i in column k is:
    
        h_i^{(k)} = f( W_i^{(k)} * h_{i-1}^{(k)} + sum_{j<k} U_i^{(k:j)} * h_{i-1}^{(j)} )
    
    Key points:
    - Lateral connections U take activations from layer (i-1) of source columns
    - They feed INTO layer i of the target (combined before the nonlinearity)
    - Only hidden layers receive lateral connections (not the output layer)
    """
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        hidden_sizes: List[int],
        source_columns: List[ProgressiveColumn],
        adapter_scale: float = 1.0
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.source_columns = nn.ModuleList(source_columns)
        self.num_sources = len(source_columns)
        
        # Freeze all source columns
        for col in self.source_columns:
            for param in col.parameters():
                param.requires_grad = False
        
        # Build target layers: layer[i] maps from size[i] to size[i+1]
        # layer 0: in_dim -> hidden[0]
        # layer 1: hidden[0] -> hidden[1]
        # ...
        # layer L: hidden[-1] -> out_dim
        self.layers = nn.ModuleList()
        sizes = [in_dim] + hidden_sizes + [out_dim]
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        
        # Build lateral adapters (U matrices in the paper)
        # For layer i in target, we need lateral connections from layer (i-1) of sources
        # 
        # layer 0 (input -> h0): takes raw input, lateral from source's raw input (not useful, skip)
        # layer 1 (h0 -> h1): lateral from source's h0 (source_intermediates[0])
        # layer 2 (h1 -> h2): lateral from source's h1 (source_intermediates[1])
        # ...
        # output layer: NO lateral connections (task-specific)
        
        self.adapters = nn.ModuleList()
        num_hidden = len(hidden_sizes)
        
        # Create adapters for layers 1 through num_hidden (hidden layers only, not output)
        for target_layer_idx in range(1, num_hidden + 1):
            layer_adapters = nn.ModuleList()
            for src_idx, source_col in enumerate(source_columns):
                # Source layer to use: layer (target_layer_idx - 1) output from source
                # This corresponds to source_intermediates[target_layer_idx - 1]
                source_h_idx = target_layer_idx - 1  # Index into source intermediates
                
                if source_h_idx < len(source_col.hidden_sizes):
                    source_hidden_dim = source_col.hidden_sizes[source_h_idx]
                else:
                    # If source has fewer layers, use the last available
                    source_hidden_dim = source_col.hidden_sizes[-1] if source_col.hidden_sizes else in_dim
                
                # Adapter maps source hidden to target layer's OUTPUT dimension
                target_output_dim = sizes[target_layer_idx + 1]
                adapter = nn.Linear(source_hidden_dim, target_output_dim)
                
                # Initialize with small weights to not disrupt initial learning
                nn.init.normal_(adapter.weight, std=adapter_scale * 0.01)
                nn.init.zeros_(adapter.bias)
                layer_adapters.append(adapter)
            
            self.adapters.append(layer_adapters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get intermediate activations from all source columns
        # source_intermediates[k][i] = activation h_i from source column k (post-ReLU)
        source_intermediates: List[List[torch.Tensor]] = []
        for col in self.source_columns:
            with torch.no_grad():
                _, intermediates = col.forward_with_intermediates(x)
            source_intermediates.append(intermediates)
        
        num_hidden = len(self.hidden_sizes)
        
        # Forward through target column with lateral connections
        for layer_idx, layer in enumerate(self.layers):
            # Compute W_i * h_{i-1}^{target}
            h = layer(x)
            
            # Add lateral connections for hidden layers (layers 1 to num_hidden)
            # Layer 0 has no lateral (processes raw input)
            # Output layer has no lateral (task-specific)
            if 1 <= layer_idx <= num_hidden:
                adapter_list_idx = layer_idx - 1  # Index into self.adapters
                adapters_for_layer = self.adapters[adapter_list_idx]
                
                for src_idx, adapter in enumerate(adapters_for_layer):
                    # Get h_{layer_idx-1} from source column src_idx
                    # This is source_intermediates[src_idx][layer_idx - 1]
                    source_h_idx = layer_idx - 1
                    if source_h_idx < len(source_intermediates[src_idx]):
                        src_h = source_intermediates[src_idx][source_h_idx]
                        # Add U_i * h_{i-1}^{source}
                        h = h + adapter(src_h)
            
            # Apply nonlinearity for hidden layers, not for output
            if layer_idx < len(self.layers) - 1:
                x = F.relu(h)
            else:
                x = h
        
        return x


class ProgressiveNetworkPolicy(nn.Module):
    """
    Complete Progressive Network for policy that wraps source columns and target column.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: List[int],
        source_columns: List[ProgressiveColumn]
    ):
        super().__init__()
        self.target_column = ProgressiveTargetColumn(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_sizes=hidden_sizes,
            source_columns=source_columns
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_column(x)


class ProgressiveActorCriticAgent:
    """
    Progressive Neural Network Agent for transfer learning.
    Uses frozen source columns with lateral connections to a trainable target column.
    """
    def __init__(self, cfg: ActorCriticConfig, source_model_paths: List[str]):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.source_model_paths = source_model_paths

        # Infer specs for all supported envs
        specs = [infer_task_spec(eid, seed=cfg.seed) for eid in SUPPORTED_ENVS]
        self.spec_map = {s.env_id: s for s in specs}
        if cfg.env_id not in self.spec_map:
            raise ValueError(f"Unsupported env_id={cfg.env_id}. Supported: {SUPPORTED_ENVS}")

        # Compute unified sizes
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
            self.writer = SummaryWriter(comment=f"_progressive_{cfg.env_id}")

        # Load source columns (frozen)
        self.source_policy_columns = []
        self.source_value_columns = []
        for path in source_model_paths:
            policy_col, value_col = self._load_source_column(path)
            self.source_policy_columns.append(policy_col)
            self.source_value_columns.append(value_col)
        
        print(f"Loaded {len(self.source_policy_columns)} source columns for progressive network")

        # Build progressive policy network (target column with lateral connections)
        self.policy_net = ProgressiveNetworkPolicy(
            in_dim=self.obs_dim_max,
            out_dim=self.policy_out_dim,
            hidden_sizes=cfg.policy_hidden_sizes,
            source_columns=self.source_policy_columns
        ).to(self.device)
        
        # Value network: also use progressive architecture
        self.value_net = ProgressiveTargetColumn(
            in_dim=self.obs_dim_max,
            out_dim=1,
            hidden_sizes=cfg.value_hidden_sizes,
            source_columns=self.source_value_columns
        ).to(self.device)

        # Only optimize target column parameters (source columns are frozen)
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)
        self.value_opt = optim.Adam(self.value_net.parameters(), lr=cfg.value_lr)

    def _load_source_column(self, path: str) -> Tuple[ProgressiveColumn, ProgressiveColumn]:
        """Load a source model and convert it to a ProgressiveColumn."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Create policy column
        policy_col = ProgressiveColumn(
            in_dim=self.obs_dim_max,
            out_dim=self.policy_out_dim,
            hidden_sizes=self.cfg.policy_hidden_sizes
        ).to(self.device)
        
        # Load weights from MLP format to ProgressiveColumn format
        source_policy = checkpoint['policy_net']
        new_state = {}
        for key, value in source_policy.items():
            if key.startswith('net.'):
                # Convert 'net.0.weight' -> 'layers.0.weight'
                parts = key.split('.')
                orig_idx = int(parts[1])
                # In MLP, layers are interleaved with ReLU, so Linear layers are at 0, 2, 4...
                # In ProgressiveColumn, layers are just 0, 1, 2...
                new_idx = orig_idx // 2
                new_key = f"layers.{new_idx}.{parts[2]}"
                new_state[new_key] = value
        
        policy_col.load_state_dict(new_state)
        
        # Create value column
        value_col = ProgressiveColumn(
            in_dim=self.obs_dim_max,
            out_dim=1,
            hidden_sizes=self.cfg.value_hidden_sizes
        ).to(self.device)
        
        source_value = checkpoint['value_net']
        new_state = {}
        for key, value in source_value.items():
            if key.startswith('net.'):
                parts = key.split('.')
                orig_idx = int(parts[1])
                new_idx = orig_idx // 2
                new_key = f"layers.{new_idx}.{parts[2]}"
                new_state[new_key] = value
        
        value_col.load_state_dict(new_state)
        
        # Freeze source columns
        for param in policy_col.parameters():
            param.requires_grad = False
        for param in value_col.parameters():
            param.requires_grad = False
        
        print(f"Loaded and froze source column from {path}")
        return policy_col, value_col

    def save_model(self, path: str):
        """Saves the progressive model weights to disk."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'obs_dim_max': self.obs_dim_max,
            'source_paths': self.source_model_paths
        }, path)
        print(f"Progressive model saved to {path}")

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        self.env.close()
        self.eval_env.close()

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
                    if terminated:
                        success += 1
                        v_s_tag = torch.tensor(0.0, device=self.device)
                    else:
                        v_s_tag = self.value_net(next_obs_t).squeeze(-1)
                    td_target = torch.tensor(reward, dtype=torch.float32) + self.cfg.gamma * v_s_tag

                delta = td_target - v_s 

                self.value_opt.zero_grad(set_to_none=True)
                
                value_loss = self.cfg.alpha_w * delta.pow(2)
                value_loss.backward()
                if self.cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.grad_clip_norm)
                self.value_opt.step()
                val_losses.append(float(value_loss.item()))

                policy_loss = -I * delta.detach() * logp - self.cfg.entropy_coef * entropy
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
                    f"[Progressive] env={self.cfg.env_id:>24s}  "
                    f"ep={ep_idx:5d}  "
                    f"avg_return({len(recent)})={avg_ret:8.2f}  "
                    f"org_return({len(recent)})={avg_ret_orig:8.2f}  "
                    f"policy_loss={history['policy_loss'][-1]:9.4f}  "
                    f"value_loss={history['value_loss'][-1]:9.4f} "
                    f"success={history['success']}"
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
