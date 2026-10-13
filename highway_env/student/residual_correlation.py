"""
Residual correlation matrix on the held-out test set, for the
factorized Gaussian style-inference model.

Computes
    rho_ij = Corr( (z_i - mu_i), (z_j - mu_j) ),   i != j
over many sampled windows, prints the 6x6 matrix, and reports
max |rho_ij| (the number that goes into the reviewer reply).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformer_test1 import (
    TimeSeriesTransformer,
    load_obs_style_trajectories,
    TrajectoryStreamDataset,
    collate_fn,
)


# ---------- Config ----------
TEST_FILE = "PPO_obs_and_rewards_test.csv"
CHECKPOINT = "transformer_weights_gaussian/epoch_120400.pth"

STATE_DIM = 16
FEATURE_DIM = 6

BATCH_SIZE = 128
WINDOW_MIN = 20
WINDOW_MAX = 50
NUM_EVAL_BATCHES = 200            # ~25,600 sampled windows


def collect_residuals(model, loader, device, num_batches):
    model.eval()
    residuals = []
    with torch.no_grad():
        for i, (x, y, m) in enumerate(loader):
            if i >= num_batches:
                break
            x, y, m = x.to(device), y.to(device), m.to(device)
            mu, _ = model(x, m)
            residuals.append((y - mu).cpu().numpy())
    return np.concatenate(residuals, axis=0)              # (N, 6)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading test trajectories ...")
    trajs = load_obs_style_trajectories(
        TEST_FILE, state_dim=STATE_DIM, feature_dim=FEATURE_DIM
    )
    dataset = TrajectoryStreamDataset(
        trajs,
        state_dim=STATE_DIM, feature_dim=FEATURE_DIM,
        window_min_len=WINDOW_MIN, window_max_len=WINDOW_MAX,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = TimeSeriesTransformer().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    print(f"Loaded checkpoint: {CHECKPOINT}")

    residuals = collect_residuals(model, loader, device, NUM_EVAL_BATCHES)
    print(f"\nCollected {residuals.shape[0]:,} residuals over "
          f"{residuals.shape[1]} style dimensions.")

    corr = np.corrcoef(residuals, rowvar=False)

    print("\nResidual correlation matrix  rho_ij = Corr((z_i - mu_i), (z_j - mu_j)):")
    print("         " + "    ".join(f" z{j+1}" for j in range(FEATURE_DIM)))
    for i in range(FEATURE_DIM):
        cells = "  ".join(f"{corr[i, j]:+0.3f}" for j in range(FEATURE_DIM))
        print(f"   z{i+1}    {cells}")

    off_diag = corr.copy()
    np.fill_diagonal(off_diag, 0.0)
    max_abs = float(np.max(np.abs(off_diag)))
    i_max, j_max = np.unravel_index(np.argmax(np.abs(off_diag)), off_diag.shape)

    print(f"\nMax |rho_ij| (off-diagonal) = {max_abs:.4f}  "
          f"(between z{i_max+1} and z{j_max+1})")
    print("Claim in reviewer reply: max |rho_ij| < 0.21")
    print("Status:", "PASS" if max_abs < 0.21 else "REVISE the claim")


if __name__ == "__main__":
    main()
