import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import csv
import os


from Transformer import (
    TimeSeriesTransformer,
    load_obs_style_trajectories,
    SlidingTrajectoryDataset,
    collate_fn
)

def build_test_samples(data_file, state_dim=16, feature_dim=6,
                       window_min_len=10, window_max_len=90, max_samples=500):
    all_trajectories = load_obs_style_trajectories(data_file, state_dim, feature_dim)

    test_samples = []
    for traj in all_trajectories:
        traj_len = len(traj)
        if traj_len < window_min_len:
            continue
        start = np.random.randint(0, traj_len - window_min_len + 1)
        end = np.random.randint(start + window_min_len, min(start + window_max_len, traj_len) + 1)
        window = traj[start:end]
        states = [step[0] for step in window]
        target = window[-1][1]
        test_samples.append((torch.tensor(states, dtype=torch.float32),
                             torch.tensor(target, dtype=torch.float32)))
        if len(test_samples) >= max_samples:
            break
    return test_samples

# def build_full_test_samples(data_file, state_dim=16, feature_dim=6, max_samples=500):
#     all_trajectories = load_obs_style_trajectories(data_file, state_dim, feature_dim)
#
#     test_samples = []
#     for traj in all_trajectories:
#         traj_len = len(traj)
#         if traj_len < 2:
#             continue
#
#         states = [step[0] for step in traj]
#         target = traj[-1][1]
#         test_samples.append(
#             (torch.tensor(states, dtype=torch.float32),
#              torch.tensor(target, dtype=torch.float32))
#         )
#         if len(test_samples) >= max_samples:
#             break
#
#     return test_samples


def evaluate(model, test_samples, device, print_examples=5):

    model.eval()
    total_mse = 0.0
    examples_shown = 0

    with torch.no_grad():
        for idx, (states, target) in enumerate(test_samples):
            x = states.unsqueeze(0).to(device)   # (1, T, state_dim)
            y = target.unsqueeze(0).to(device)   # (1, feature_dim)
            T = x.size(1)
            mask = torch.ones(1, T, dtype=torch.bool, device=device)

            out = model(x, mask)  # (1, feature_dim)
            mse = F.mse_loss(out, y).item()
            total_mse += mse

            #
            if examples_shown < print_examples:
                print(f"\n--- Example {examples_shown+1} ---")
                print("Input length:", T)
                print("Target (style):", y.squeeze(0).cpu().numpy())
                print("Prediction    :", out.squeeze(0).cpu().numpy())
                print("MSE           :", mse)
                examples_shown += 1

    avg_mse = total_mse / len(test_samples)
    return avg_mse



if __name__ == "__main__":

    test_file = "PPO_obs_and_rewards_test.csv"
    checkpoint = "transformer_weights1/epoch_7000.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    test_samples = build_test_samples(test_file, state_dim=16, feature_dim=6, max_samples=500)


    # test_samples = build_full_test_samples(test_file, state_dim=16, feature_dim=6, max_samples=100)

    print(f"Loaded {len(test_samples)} test samples")


    model = TimeSeriesTransformer().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"✅ Loaded checkpoint: {checkpoint}")


    avg_mse = evaluate(model, test_samples, device)
    print(f"🧪 Test Avg MSE: {avg_mse:.6f}")
