import ray
from ray.rllib.algorithms.ppo import PPOConfig
from register_env1 import register_custom_env
import numpy as np
import gymnasium as gym
import csv
import custom_model

# Initialize Ray
ray.init()

# Register the custom environment
register_custom_env()

# Configure PPO
config = (
    PPOConfig()
    .environment("CustomEnv")
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=1)
    .training(
        train_batch_size=4000,
        sgd_minibatch_size=256,
        num_sgd_iter=10,
        lr=1e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        grad_clip=0.2,
        model={
            "custom_model": "custom_fcnet",   
        },
    )
)

# Build and restore trainer
trainer = config.build()
checkpoint_path = "ppo_model_checkpoint/ppo_model_9815"
trainer.restore(checkpoint_path)

# Use native gym env for rendering
env = gym.make("CustomEnv", render_mode="rgb_array")
obs, _ = env.reset()

reward_set = []
num_episodes =100
total_rewards = []
all_features = []


with open("PPO_obs_and_rewards_feature.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(["Episode", "Step", "Reward", "Obs(flat)", "Style", "Features"])

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        episode_reward_set = []
        step = 0

        while not done:
            action = trainer.compute_single_action(obs, explore=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)


            vehicle = env.unwrapped.env.unwrapped.vehicle
            features = env.unwrapped.features(vehicle, action)

            all_features.append(features)

            done = terminated or truncated
            total_reward += reward
            episode_reward_set.append(total_reward)

            writer.writerow([
                episode,
                step,
                reward,
                next_obs["obs"].flatten().tolist(),
                next_obs["style"].tolist(),
                features
            ])

            obs = next_obs
            step += 1

        total_rewards.append(total_reward)
        reward_set.append(episode_reward_set)
        print(f"Episode {episode + 1}: Total reward = {total_reward}")

# Output stats
print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards)}")
print(f"Max reward over {num_episodes} episodes: {np.max(total_rewards)}")
print(f"Min reward over {num_episodes} episodes: {np.min(total_rewards)}")


all_features = np.array(all_features, dtype=np.float32)
if all_features.ndim == 1:
    mean_features = np.mean(all_features)
else:
    mean_features = np.mean(all_features, axis=0)

print(f"Mean of features across all steps:\n{mean_features}")

# Shutdown Ray
ray.shutdown()
