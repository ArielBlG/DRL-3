from typing import Tuple, Literal, List, Optional
import numpy as np
import gymnasium as gym

from dataclasses import dataclass, field

TaskKind = Literal["discrete", "continuous"]

@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    kind: TaskKind
    obs_dim: int
    # discrete
    n_actions: Optional[int] = None
    # continuous
    act_dim: Optional[int] = None
    act_low: Optional[np.ndarray] = None
    act_high: Optional[np.ndarray] = None


def compute_unified_dims(task_specs: List[TaskSpec]) -> Tuple[int, int]:
    obs_dim_max = max(t.obs_dim for t in task_specs)

    max_discrete = 0
    max_cont = 0
    for t in task_specs:
        if t.kind == "discrete":
            max_discrete = max(max_discrete, int(t.n_actions))
        else:
            max_cont = max(max_cont, int(t.act_dim))

    # Policy head output dimension must be identical for all tasks.
    # Discrete needs max_discrete logits.
    # Continuous needs 2*act_dim (mean, log_std) => use 2*max_cont.
    policy_out_dim = max(max_discrete, 2 * max_cont)
    return obs_dim_max, policy_out_dim


def infer_task_spec(env_id: str, seed: int = 0) -> TaskSpec:
    env = gym.make(env_id)
    try:
        env.reset(seed=seed)
        obs_space = env.observation_space
        act_space = env.action_space

        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError(f"{env_id}: expected Box observation space, got {type(obs_space)}")

        obs_dim = int(np.prod(obs_space.shape))

        if isinstance(act_space, gym.spaces.Discrete):
            return TaskSpec(env_id=env_id, kind="discrete", obs_dim=obs_dim, n_actions=act_space.n)

        if isinstance(act_space, gym.spaces.Box):
            act_dim = int(np.prod(act_space.shape))
            low = np.array(act_space.low, dtype=np.float32).reshape(-1)
            high = np.array(act_space.high, dtype=np.float32).reshape(-1)
            return TaskSpec(
                env_id=env_id,
                kind="continuous",
                obs_dim=obs_dim,
                act_dim=act_dim,
                act_low=low,
                act_high=high,
            )

        raise ValueError(f"{env_id}: unsupported action space {type(act_space)}")
    finally:
        env.close()