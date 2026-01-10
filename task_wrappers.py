
import gymnasium as gym
import numpy as np

class PadObservation(gym.ObservationWrapper):
    """
    Pads flat observation vectors with zeros up to obs_dim_max.
    Crucial for Transfer Learning: Allows CartPole (4 dims) to feed into Acrobot Net (6 dims).
    """
    def __init__(self, env: gym.Env, obs_dim_max: int):
        super().__init__(env)
        self.obs_dim_max = obs_dim_max

        assert isinstance(env.observation_space, gym.spaces.Box)
        orig = env.observation_space
    
        
        low = np.full((obs_dim_max,), -np.inf, dtype=np.float32)
        high = np.full((obs_dim_max,), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(obs_dim_max,), dtype=np.float32)

    def observation(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if x.shape[0] == self.obs_dim_max:
            return x
        # If smaller, pad with zeros
        if x.shape[0] < self.obs_dim_max:
            out = np.zeros((self.obs_dim_max,), dtype=np.float32)
            out[: x.shape[0]] = x
            return out
        return x

class DiscretizeBoxAction(gym.ActionWrapper):
    """
    Turns a 1D continuous Box action env into a Discrete(n_bins) env by mapping
    discrete indices to fixed continuous actions (bin centers).
    """
    def __init__(self, env: gym.Env, n_bins: int = 7):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), "Expected Box action space"
        assert int(np.prod(env.action_space.shape)) == 1, "This wrapper supports 1D actions only"

        self.n_bins = n_bins
        low = float(env.action_space.low.reshape(-1)[0])
        high = float(env.action_space.high.reshape(-1)[0])

        # Bin centers in [low, high]
        # self.bin_values = np.linspace(low, high, n_bins, dtype=np.float32)
        self.bin_values = np.array(
            # [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            # [-1.0, -0.5, 0.1, 0.5, 1.0],
            [-1.0, 0, 1.0],
            dtype=np.float32
        )
        # New discrete action space
        self.action_space = gym.spaces.Discrete(n_bins)

    def action(self, act_idx: int):
        # Map index -> continuous action array of shape (1,)
        a = self.bin_values[int(act_idx)]
        return np.array([a], dtype=np.float32)

class MountainCarEnergyShaping(gym.Wrapper):
    """
    Potential-based shaping + explicit end bonuses to avoid time-limit plateaus.
    Works with obs = [position, velocity, ...]. Uses first two entries.
    """
    def __init__(self, env, gamma: float, c: float = 10.0,
                 success_bonus: float = 100.0,
                 timeout_penalty: float = 50.0):
        super().__init__(env)
        self.gamma = float(gamma)
        self.c = float(c)
        self.success_bonus = float(success_bonus)
        self.timeout_penalty = float(timeout_penalty)
        self._last_obs = None

    def _phi(self, obs):
        x = float(obs[0])
        v = float(obs[1])
        return np.sin(3.0 * x) + 0.5 * (v * v)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        assert self._last_obs is not None, "Call reset() before step()."
        obs2, r, terminated, truncated, info = self.env.step(action)
        obs2 = np.asarray(obs2, dtype=np.float32)

        phi_s  = self._phi(self._last_obs)
        phi_s2 = self._phi(obs2)

        shaped = float(r) + self.gamma * self.c * phi_s2 - self.c * phi_s

        info = dict(info)
        info.setdefault("orig_reward", float(r))
        self._last_obs = obs2
        return obs2, shaped, terminated, truncated, info
    

class ScaleVelocity(gym.ObservationWrapper):
    def observation(self, obs):
        new_obs = np.array(obs, dtype=np.float32)
        new_obs[1] *= 15.0  # Boost velocity signal
        return new_obs