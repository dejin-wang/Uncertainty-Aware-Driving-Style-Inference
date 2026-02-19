import ray
from ray.rllib.algorithms.ppo import PPOConfig
from register_env1 import register_custom_env
import numpy as np
import gymnasium as gym
import csv
import custom_model
import json

def obs_to_jsonable(obs):
    """Convert obs dict into JSON-safe pure Python types."""
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out

ray.init()
register_custom_env()

config = (
    PPOConfig()
    .environment("CustomEnv")
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=1)
    .training(model={"custom_model": "custom_fcnet"})
)

trainer = config.build()
trainer.restore("ppo_model_checkpoint/ppo_model_9815")

env = gym.make("CustomEnv")
obs, _ = env.reset()

csv_path = "PPO_obs_and_true_actions_FULL.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Step", "TrueAction", "ObsDict"])

    num_episodes = 200
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done:

            # deterministic action
            true_action = trainer.compute_single_action(obs, explore=False)

            # convert obs to jsonable version
            obs_jsonable = obs_to_jsonable(obs)
            obs_json = json.dumps(obs_jsonable)

            writer.writerow([
                ep,
                step,
                true_action.tolist(),
                obs_json
            ])

            obs, reward, terminated, truncated, _ = env.step(true_action)
            done = terminated or truncated
            step += 1

print(f"Saved to {csv_path}")
ray.shutdown()
