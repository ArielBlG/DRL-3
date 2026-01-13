import time
import os
from typing import Dict, List

import numpy as np
import torch

from actor_critic import (
    ActorCriticConfig, 
    UnifiedActorCriticAgent,
)
from progressive_actor_critic import ProgressiveActorCriticAgent

def train_source_networks(cfg_base: ActorCriticConfig, envs: list, models_dir: str = "models"):
    """Train separate source networks for each environment."""
    os.makedirs(models_dir, exist_ok=True)
    
    for env_id in envs:
        model_path = os.path.join(models_dir, f"{env_id.replace('-', '_')}.pt")
        
        # Skip if already trained
        if os.path.exists(model_path):
            print(f"Source model for {env_id} already exists at {model_path}, skipping training...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Training source network for: {env_id}")
        print(f"{'='*60}")
        
        cfg = ActorCriticConfig(
            env_id=env_id,
            gamma=cfg_base.gamma,
            total_episodes=cfg_base.total_episodes,
            entropy_coef=cfg_base.entropy_coef,
            log_every=cfg_base.log_every,
            seed=cfg_base.seed,
            device=cfg_base.device,
            log_to_tensorboard=cfg_base.log_to_tensorboard,
            grad_clip_norm=cfg_base.grad_clip_norm,
            policy_lr=cfg_base.policy_lr,
            value_lr=cfg_base.value_lr,
            n_bins=cfg_base.n_bins,
            policy_hidden_sizes=cfg_base.policy_hidden_sizes,
            value_hidden_sizes=cfg_base.value_hidden_sizes,
        )

        agent = UnifiedActorCriticAgent(cfg)
        agent.train()
        agent.save_model(model_path)
        
        # Evaluate
        eval_return = agent.evaluate(episodes=20, deterministic=True)
        print(f"Source {env_id} - Eval avg return: {eval_return:.2f}")
        
        agent.close()


def run_progressive_transfer(
    cfg_base: ActorCriticConfig,
    source_envs: list,
    target_env: str,
    models_dir: str = "models",
    target_episodes: int = 500
):
    """Run progressive network transfer from source envs to target env."""
    print(f"\n{'='*60}")
    print(f"Progressive Transfer: {source_envs} -> {target_env}")
    print(f"{'='*60}")
    
    # Build source model paths
    source_paths = [
        os.path.join(models_dir, f"{env_id.replace('-', '_')}.pt")
        for env_id in source_envs
    ]
    
    # Verify source models exist
    for path in source_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source model not found: {path}. Train source networks first.")
    
    # Configure for target environment
    cfg = ActorCriticConfig(
        env_id=target_env,
        gamma=cfg_base.gamma,
        total_episodes=target_episodes,
        entropy_coef=cfg_base.entropy_coef,
        log_every=cfg_base.log_every,
        seed=cfg_base.seed,
        device=cfg_base.device,
        log_to_tensorboard=cfg_base.log_to_tensorboard,
        grad_clip_norm=cfg_base.grad_clip_norm,
        policy_lr=cfg_base.policy_lr,
        value_lr=cfg_base.value_lr,
        n_bins=cfg_base.n_bins,
        policy_hidden_sizes=cfg_base.policy_hidden_sizes,
        value_hidden_sizes=cfg_base.value_hidden_sizes,
    )
    
    t0 = time.perf_counter()
    
    # Create progressive agent
    agent = ProgressiveActorCriticAgent(cfg, source_model_paths=source_paths)
    
    # Train
    history = agent.train()
    
    elapsed = time.perf_counter() - t0
    print(f"\nTraining took {elapsed:.2f} seconds")
    
    # Evaluate on target task
    eval_return = agent.evaluate(episodes=20, deterministic=True)
    print(f"Progressive {target_env} - Eval avg return: {eval_return:.2f}")
    
    # Evaluate source columns on their respective source tasks (baseline)
    print(f"\n--- Evaluating frozen SOURCE columns on source tasks (baseline) ---")
    source_baseline_evals = evaluate_frozen_source_columns(agent, source_envs, cfg_base, episodes=20)
    
    # Evaluate target column on source tasks (transfer preservation test)
    print(f"\n--- Evaluating TARGET column on source tasks (preservation test) ---")
    target_on_source_evals = evaluate_target_on_source_tasks(agent, source_envs, cfg_base, episodes=20)
    
    # Save progressive model
    save_name = f"progressive_{'_'.join([e.split('-')[0] for e in source_envs])}_to_{target_env.split('-')[0]}.pt"
    agent.save_model(os.path.join(models_dir, save_name))
    
    agent.close()
    
    return history, eval_return, source_baseline_evals, target_on_source_evals


def evaluate_frozen_source_columns(
    agent: ProgressiveActorCriticAgent,
    source_envs: List[str],
    cfg_base: ActorCriticConfig,
    episodes: int = 20
) -> Dict[str, float]:
    """
    Evaluate each frozen SOURCE column on its original source environment.
    This is the baseline - how well the original trained models perform.
    """
    from torch.distributions import Categorical
    import gymnasium as gym
    from task_wrappers import DiscretizeBoxAction, PadObservation, ScaleVelocity
    
    results = {}
    
    for src_idx, env_id in enumerate(source_envs):
        source_policy = agent.source_policy_columns[src_idx]
        
        # Create environment for this source task
        env = gym.make(env_id)
        if env_id == "MountainCarContinuous-v0":
            env = ScaleVelocity(env)
            env = DiscretizeBoxAction(env, n_bins=cfg_base.n_bins)
        env = PadObservation(env, obs_dim_max=agent.obs_dim_max)
        
        rets = []
        for _ in range(episodes):
            obs, _ = env.reset()
            done, total, steps = False, 0.0, 0
            while not done and steps < cfg_base.max_steps_per_episode:
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(agent.device)
                with torch.no_grad():
                    logits_all = source_policy(obs_t)
                    n_valid = int(env.action_space.n)
                    logits = logits_all[:n_valid]
                    dist = Categorical(logits=logits / 2.0)
                    action = int(torch.argmax(dist.probs).item())
                
                obs, r, terminated, truncated, _ = env.step(action)
                total += float(r)
                steps += 1
                done = terminated or truncated
            rets.append(total)
        
        env.close()
        avg_return = float(np.mean(rets))
        results[env_id] = avg_return
        print(f"  Source column {src_idx} on {env_id}: avg return = {avg_return:.2f}")
    
    return results


def evaluate_target_on_source_tasks(
    agent: ProgressiveActorCriticAgent,
    source_envs: List[str],
    cfg_base: ActorCriticConfig,
    episodes: int = 20
) -> Dict[str, float]:
    """
    Evaluate the TARGET column (progressive network) on the source environments.
    This tests whether transfer learning preserves performance on old tasks.
    """
    from torch.distributions import Categorical
    import gymnasium as gym
    from task_wrappers import DiscretizeBoxAction, PadObservation, ScaleVelocity
    
    results = {}
    
    for src_idx, env_id in enumerate(source_envs):
        # Create environment for this source task
        env = gym.make(env_id)
        if env_id == "MountainCarContinuous-v0":
            env = ScaleVelocity(env)
            env = DiscretizeBoxAction(env, n_bins=cfg_base.n_bins)
        env = PadObservation(env, obs_dim_max=agent.obs_dim_max)
        
        rets = []
        for _ in range(episodes):
            obs, _ = env.reset()
            done, total, steps = False, 0.0, 0
            while not done and steps < cfg_base.max_steps_per_episode:
                # Use the TARGET column (progressive policy) to select action
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(agent.device)
                with torch.no_grad():
                    logits_all = agent.policy_net(obs_t)  # Use the progressive network
                    n_valid = int(env.action_space.n)
                    logits = logits_all[:n_valid]
                    dist = Categorical(logits=logits / 2.0)
                    action = int(torch.argmax(dist.probs).item())
                
                obs, r, terminated, truncated, _ = env.step(action)
                total += float(r)
                steps += 1
                done = terminated or truncated
            rets.append(total)
        
        env.close()
        avg_return = float(np.mean(rets))
        results[env_id] = avg_return
        print(f"  Target column on {env_id}: avg return = {avg_return:.2f}")
    
    return results


if __name__ == "__main__":
    # Base configuration
    cfg_base = ActorCriticConfig(
        gamma=0.996,
        total_episodes=650,
        entropy_coef=0.01,
        log_every=10,
        seed=0,
        device="cpu",
        log_to_tensorboard=True,
        grad_clip_norm=None,
        policy_lr=3e-4,
        value_lr=1e-3,
        n_bins=3,
        policy_hidden_sizes=[64, 64],
        value_hidden_sizes=[64, 64],
    )

    # Step 1: Train source networks for all environments
    print("\n" + "="*70)
    print("STEP 1: Training Source Networks")
    print("="*70)
    all_envs = ["Acrobot-v1", "CartPole-v1", "MountainCarContinuous-v0"]
    train_source_networks(cfg_base, all_envs, models_dir="models")

    # Step 2: Progressive transfer experiments
    print("\n" + "="*70)
    print("STEP 2: Progressive Network Transfer Experiments")
    print("="*70)
    
    # Experiment 1: {Acrobot, MountainCar} -> CartPole
    print("\n--- Experiment 1: {Acrobot, MountainCar} -> CartPole ---")
    history1, eval1, source_baseline1, target_on_source1 = run_progressive_transfer(
        cfg_base=cfg_base,
        source_envs=["Acrobot-v1", "MountainCarContinuous-v0"],
        target_env="CartPole-v1",
        models_dir="models",
        target_episodes=650
    )
    
    # Experiment 2: {CartPole, Acrobot} -> MountainCar
    print("\n--- Experiment 2: {CartPole, Acrobot} -> MountainCar ---")
    history2, eval2, source_baseline2, target_on_source2 = run_progressive_transfer(
        cfg_base=cfg_base,
        source_envs=["CartPole-v1", "Acrobot-v1"],
        target_env="MountainCarContinuous-v0",
        models_dir="models",
        target_episodes=500
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nExperiment 1: {{Acrobot, MountainCar}} -> CartPole")
    print(f"  Target task (CartPole) eval return: {eval1:.2f}")
    print(f"\n  Comparison on SOURCE tasks:")
    print(f"  {'Environment':<30} {'Source Column':<15} {'Target Column':<15} {'Preserved?':<10}")
    print(f"  {'-'*70}")
    for env_id in source_baseline1.keys():
        src_ret = source_baseline1[env_id]
        tgt_ret = target_on_source1[env_id]
        preserved = "✓" if tgt_ret >= src_ret * 0.8 else "✗"  # 80% threshold
        print(f"  {env_id:<30} {src_ret:<15.2f} {tgt_ret:<15.2f} {preserved:<10}")
    
    print(f"\nExperiment 2: {{CartPole, Acrobot}} -> MountainCar")
    print(f"  Target task (MountainCar) eval return: {eval2:.2f}")
    print(f"\n  Comparison on SOURCE tasks:")
    print(f"  {'Environment':<30} {'Source Column':<15} {'Target Column':<15} {'Preserved?':<10}")
    print(f"  {'-'*70}")
    for env_id in source_baseline2.keys():
        src_ret = source_baseline2[env_id]
        tgt_ret = target_on_source2[env_id]
        preserved = "✓" if tgt_ret >= src_ret * 0.8 else "✗"  # 80% threshold
        print(f"  {env_id:<30} {src_ret:<15.2f} {tgt_ret:<15.2f} {preserved:<10}")
    
