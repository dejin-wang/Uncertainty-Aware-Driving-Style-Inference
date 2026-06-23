import os
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from register_env import register_custom_env
import custom_model


class DynamicGammaCallbacks(DefaultCallbacks):
    def on_postprocess_trajectory(
        self, *, worker, episode, agent_id, policy_id,
        policies, postprocessed_batch, original_batches, **kwargs
    ):
        obs_batch = postprocessed_batch["obs"]


        style_batch = obs_batch[:, -6:]
        # print("style shape:", style_batch.shape)
        # print("z5 values:", style_batch[:, 4])

        gamma = float(style_batch[0, 4])


        rewards = postprocessed_batch[SampleBatch.REWARDS]
        values = postprocessed_batch[SampleBatch.VF_PREDS]
        dones = postprocessed_batch[SampleBatch.TERMINATEDS]

        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)

        last_gae = 0.0
        next_value = 0.0
        next_nonterminal = 1.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + gamma * 0.8 * next_nonterminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

            next_value = values[t]
            next_nonterminal = 1.0 - dones[t]

        postprocessed_batch["advantages"] = advantages
        postprocessed_batch["value_targets"] = returns




# -----------------------------
# main
# -----------------------------
def main():
    ray.init()
    register_custom_env()

    config = (
        PPOConfig()
        .environment("CustomEnv")
        .framework("torch")
        .rollouts(num_rollout_workers=40)
        .resources(num_gpus=1)
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=256,
            num_sgd_iter=15,
            lr=1e-6,
            gamma=0.99,    #roverridden dynamically by DynamicGammaCallbacks
            lambda_=0.8,
            clip_param=0.2,
            grad_clip=1,
            model={"custom_model": "custom_fcnet"},
        )
        .callbacks(DynamicGammaCallbacks)
    )

    log_dir = "logs/ppo_custom_env"
    os.makedirs(log_dir, exist_ok=True)

    trainer = config.build()
    # checkpoint_path = "ppo_model_checkpoint1/ppo_model_495"
    # trainer.restore(checkpoint_path)

    for i in range(20000):
        result = trainer.train()
        reward_mean = result["episode_reward_mean"]
        print(f"Iteration {i}: reward_mean = {reward_mean}")

        #
        with open(f"{log_dir}/training_log.txt", "a") as log_file:
            log_file.write(f"Iteration {i}: {result}\n")

        with open(f"{log_dir}/reward_mean.txt", "a") as reward_file:
            reward_file.write(f"{i},{reward_mean}\n")

        #
        if i % 5 == 0:
            checkpoint_dir = "ppo_model_checkpoint1"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"ppo_model_{i}")
            trainer.save(checkpoint_path)

    ray.shutdown()

if __name__ == "__main__":
    main()
