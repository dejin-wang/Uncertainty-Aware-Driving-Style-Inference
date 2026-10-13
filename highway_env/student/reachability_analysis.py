import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import deque
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from register_env1 import register_custom_env
import gymnasium as gym
import custom_model
from crown_simplified import crown_analyze_style  # existing API

# If TimeSeriesTransformer is in another file (recommended):
# from transformer_test import TimeSeriesTransformer

# ---- If not available as import, uncomment a minimal definition ----
class TimeSeriesTransformer(nn.Module):
    def __init__(self, state_dim=16, feature_dim=6, d_model=128, nhead=8, num_layers=3, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, 2 * feature_dim)
        self.d_model = d_model

    def forward(self, x, mask):
        # x: [B, T, state_dim], mask: [B, T] (True for valid)
        B, T, _ = x.shape
        x = self.input_proj(x)

        # sinusoidal positional encoding
        position = torch.arange(T, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device) * -(np.log(10000.0) / self.d_model))
        pe = torch.zeros(T, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x = x + pe.unsqueeze(0)

        x = self.encoder(x, src_key_padding_mask=~mask)  # invert mask for src_key_padding_mask
        x = x * mask.unsqueeze(-1).float()
        pooled = x.sum(1) / mask.sum(1, keepdim=True)
        out = self.output_proj(pooled)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar

def gaussian_head_to_mu_sigma(mu, logvar):
    # mu, logvar: [B, feature_dim]
    sigma = torch.exp(0.5 * logvar)
    return mu, sigma

def load_transformer(checkpoint_path: str, device: torch.device,
                     state_dim: int = 16, feature_dim: int = 6) -> TimeSeriesTransformer:
    model = TimeSeriesTransformer(state_dim=state_dim, feature_dim=feature_dim)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def make_window_tensor(buffer: deque, device: torch.device):
    # buffer contains a list of 1D state vectors (len=state_dim)
    seq = torch.tensor(np.stack(buffer, axis=0), dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, state_dim]
    mask = torch.ones(1, seq.shape[1], dtype=torch.bool, device=device)  # all valid
    return seq, mask


def try_call_crown(model, obs_tensor, style_tensor, eps_vec):
    """
    Tries to call crown_analyze_style with a vector eps (per-dimension).
    If the function only accepts scalar eps, fallback to scalar = max(eps_vec).
    If you have an alternative API crown_analyze_style_bounds(model, obs, lb, ub),
    you can replace this with that call.
    """
    try:
        # Attempt vector eps (if supported)
        return crown_analyze_style(model, obs_tensor, style_tensor, eps_vec)
    except TypeError:
        # Fallback: scalar eps (conservative)
        eps_scalar = float(torch.max(eps_vec).item())
        return crown_analyze_style(model, obs_tensor, style_tensor, eps_scalar)

# ---------------------------
# Initialize Ray and Env
# ---------------------------
ray.init()
register_custom_env()

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
        model={"custom_model": "custom_fcnet"},
    )
)

trainer = config.build()
checkpoint_path = "ppo_model_checkpoint/ppo_model_9815"
trainer.restore(checkpoint_path)

policy = trainer.get_policy()
ppo_model = policy.model  # CustomFCNet instance

# ---------------------------
# Load Transformer (Gaussian head) for style prediction
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf_checkpoint = "transformer_weights_gaussian/epoch_120400.pth"  # << set to your path
style_feature_dim = 6
state_dim = 16
tf_model = load_transformer(tf_checkpoint, device, state_dim=state_dim, feature_dim=style_feature_dim)

# ---------------------------
# Create env and evaluation
# ---------------------------
env = gym.make("CustomEnv", render_mode=None)
obs, _ = env.reset()

# Maintain a state window (trajectory) for the transformer input
WINDOW_MIN = 3
WINDOW_MAX = 90
state_buffer = deque(maxlen=WINDOW_MAX)

# Prime the buffer with the initial observation if available
if "obs" in obs:
    flat_state = np.asarray(obs["obs"]).flatten().astype(np.float32)
    if flat_state.shape[0] != state_dim:
        raise ValueError(f"Expected state_dim={state_dim}, got {flat_state.shape[0]}")
    state_buffer.append(flat_state)

num_episodes = 1

with open("PPO_CROWN_style_analysis.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Episode", "Step", "Reward", "Action", "LowerBound", "UpperBound"])

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state_buffer.clear()

        # push the first state
        flat_state = np.asarray(obs["obs"]).flatten().astype(np.float32)
        state_buffer.append(flat_state)

        done = False
        total_reward = 0.0
        step = 0

        while not done:
            # 1) Compute deterministic action from PPO (unchanged)
            action = trainer.compute_single_action(obs, explore=False)

            # 2) Push current state into buffer BEFORE crown (to reflect "current trajectory")
            flat_state = np.asarray(obs["obs"]).flatten().astype(np.float32)
            state_buffer.append(flat_state)

            # 3) If we have at least WINDOW_MIN frames, run transformer to get (mu, sigma)
            if len(state_buffer) >= WINDOW_MIN:
                # Make [1, T, state_dim] and mask
                x_seq, x_mask = make_window_tensor(state_buffer, device)
                with torch.no_grad():
                    mu, logvar = tf_model(x_seq, x_mask)            # [1, 6] each
                    mu, sigma = gaussian_head_to_mu_sigma(mu, logvar)


                style_mu = mu[0].detach()                           # [6]
                eps_vec = (0.5 * sigma[0].detach()).clamp(min=1e-6) # [6], 2*σ per-dim

                # Replace style by transformer μ; use 2σ as eps (vector)
                style_tensor = style_mu.to(device)
            else:
                # Not enough history yet: fall back to env-provided style with a small eps
                style_tensor = torch.tensor(np.asarray(obs["style"], dtype=np.float32), device=device)
                eps_vec = torch.full_like(style_tensor, 0.1)  # conservative default

            # 4) Prepare obs tensor for CROWN
            obs_tensor = torch.tensor(flat_state, dtype=torch.float32, device=device)

            # 5) Run CROWN with transformer-driven style and eps (vector if supported)
            l_out, u_out = try_call_crown(ppo_model, obs_tensor, style_tensor, eps_vec)

            # --- printing for debug (first two dims) ---
            print("\n================= Step Info =================")
            print(f"Action (actual PPO output): {action}")
            print(f"style_mu (used): {style_tensor.detach().cpu().numpy()}")
            print(f"2*sigma (eps per-dim): {eps_vec.detach().cpu().numpy()}")
            print(f"l_out[:2] (mean lower bounds): {l_out[:2].detach().cpu().numpy()}")
            print(f"u_out[:2] (mean upper bounds): {u_out[:2].detach().cpu().numpy()}")
            print("============================================\n")

            # 6) Step env
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            writer.writerow([
                episode,
                step,
                reward,
                action,
                l_out.detach().cpu().numpy().tolist(),
                u_out.detach().cpu().numpy().tolist()
            ])
            step += 1

        print(f"Episode {episode + 1}: Total reward = {total_reward}")

print("CROWN (transformer-driven style perturbation) analysis completed. Results saved to PPO_CROWN_style_analysis.csv")
ray.shutdown()
