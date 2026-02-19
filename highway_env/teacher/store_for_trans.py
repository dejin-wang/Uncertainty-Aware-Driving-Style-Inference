import ray
from ray.rllib.algorithms.ppo import PPOConfig
from register_env1 import register_custom_env
import numpy as np
import gymnasium as gym
import csv
import custom_model
import numpy as np



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
            "custom_model": "custom_fcnet",   # 
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
num_episodes = 300000
total_rewards = []




# 
with open("PPO_obs_and_rewards100_6_20.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # 
    writer.writerow(["Episode", "Step", "Reward", "Obs(flat)", "Style"])

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        episode_reward_set = []
        step = 0
        z6= np.random.uniform(0.75, 1.0)


        while not done:
            noise = np.random.normal(loc=0.0, scale=np.sqrt(0.25*(1 - z6) * np.array([1.0, 0.1], dtype=np.float32)))

            action = trainer.compute_single_action(obs, explore=False)
            # action = trainer.compute_single_action(obs, explore=False)+noise
            obs, reward, terminated, truncated, _ = env.step(action)
            print(obs)
            done = terminated or truncated
            total_reward += reward
            episode_reward_set.append(total_reward)


            writer.writerow([
                episode,
                step,
                reward,
                obs["obs"].flatten().tolist(),
                obs["style"].tolist()
            ])

            step += 1

        total_rewards.append(total_reward)
        reward_set.append(episode_reward_set)
        print(f"Episode {episode + 1}: Total reward = {total_reward}")

# Output stats
print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards)}")
print(f"Max reward over {num_episodes} episodes: {np.max(total_rewards)}")
print(f"Min reward over {num_episodes} episodes: {np.min(total_rewards)}")

# Shutdown Ray
ray.shutdown()
