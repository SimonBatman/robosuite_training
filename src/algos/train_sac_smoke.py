import os, time
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.envs.robosuite_gym_wrapper import RobosuiteGymWrapper

SEED = 1000
np.random.seed(SEED)
torch.manual_seed(SEED)

def make_env(rank=0, seed=0):
    def _init():
        env = RobosuiteGymWrapper(
            env_name="Lift",
            robots="Panda",
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            obs_v1_params={"include_cube_quat": True, "normalize": False}
        )
        try:
            env.seed(seed + rank)
        except Exception:
            pass
        return env
    return _init

def train(total_timesteps=10000, n_envs=1, logdir="./logs/sac_lift_smoke"):
    os.makedirs(logdir, exist_ok=True)
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, SEED) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, SEED)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_obs=10.)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logdir,
        learning_rate=3e-4,
        buffer_size=int(1e6),
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        device=device,
        seed=SEED
    )
    checkpoint_cb = CheckpointCallback(save_freq=5000, save_path=os.path.join(logdir, "checkpoints"),
                                       name_prefix="sac_smoke")
    eval_env = DummyVecEnv([make_env(0, SEED+100)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)
    eval_cb = EvalCallback(eval_env, best_model_save_path=os.path.join(logdir, "best"),
                           log_path=os.path.join(logdir, "eval_log"), eval_freq=5000,
                           n_eval_episodes=5, deterministic=True, render=False)
    try:
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])
    finally:
        model.save(os.path.join(logdir, "final_model"))
        env.close()
        eval_env.close()

if __name__ == "__main__":
    train()
