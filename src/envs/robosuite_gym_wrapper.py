import gym
import numpy as np
import robosuite as suite
from gym import spaces
from src.utils.obs import ObsV1

class RobosuiteGymWrapper(gym.Env):
    def __init__(self,
                 env_name="Lift",
                 robots="Panda",
                 use_camera_obs=False,
                 has_renderer=False,
                 has_offscreen_renderer=False,
                 obs_v1_params=None):
        super().__init__()
        self.rs_env = suite.make(
            env_name=env_name,
            robots=robots,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
        )
        self.obs_builder = ObsV1(**(obs_v1_params or {}))
        raw = self.rs_env.reset()
        flat = self.obs_builder.flatten(raw)
        obs_dim = flat.shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_dim = None
        if hasattr(self.rs_env, "action_dim"):
            action_dim = int(self.rs_env.action_dim)
        elif hasattr(self.rs_env, "action_space") and hasattr(self.rs_env.action_space, "shape"):
            action_dim = int(self.rs_env.action_space.shape[0])
        else:
            try:
                sample_action = self.rs_env.action_spec
                action_dim = int(np.prod(sample_action.shape))
            except Exception:
                raise ValueError("Cannot determine action_dim automatically.")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def reset(self):
        raw = self.rs_env.reset()
        return self.obs_builder.flatten(raw)

    def step(self, action):
        raw_obs, reward, done, info = self.rs_env.step(action)
        obs = self.obs_builder.flatten(raw_obs)
        return obs, reward, done, info

    def close(self):
        self.rs_env.close()
