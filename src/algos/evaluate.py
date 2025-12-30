import numpy as np
from stable_baselines3 import SAC
from src.envs.robosuite_gym_wrapper import RobosuiteGymWrapper

def evaluate(model_path, n_episodes=20):
    env = RobosuiteGymWrapper(env_name="Lift", robots="Panda", use_camera_obs=False)
    model = SAC.load(model_path, env=env)
    successes = 0
    rewards = []
    for ep in range(n_episodes):
        o = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a, _ = model.predict(o, deterministic=True)
            o, r, done, info = env.step(a)
            ep_ret += r
        rewards.append(ep_ret)
        succ = info.get("success", False)
        if succ:
            successes += 1
    env.close()
    print(f"Avg reward: {np.mean(rewards):.3f}, Success rate: {successes}/{n_episodes} = {successes/n_episodes:.3f}")

if __name__ == "__main__":
    evaluate("logs/sac_lift_smoke/final_model.zip", n_episodes=20)
