"""
Empirical coverage of the per-dimension credible intervals from the
factorized Gaussian style-inference model.

For each sampled window (x, z) the model predicts (mu_i, sigma_i) for
i = 1..6.  The empirical coverage at nominal confidence 1 - alpha is

    C_i(alpha) = (1/N) * sum_n  I{ |z_i^(n) - mu_i^(n)| <= z_{alpha/2} * sigma_i^(n) }

where z_{alpha/2} is the standard-normal quantile.  We report
per-dimension coverage and the marginal average across the six style
dimensions for 1 - alpha in {0.80, 0.90, 0.95}.

Claim in the reviewer reply (averaged across dims):
    0.798 / 0.901 / 0.948  (within ~1% of nominal)
"""

import numpy as np
import torch
from scipy.stats import norm
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
NUM_EVAL_BATCHES = 200                    # ~25,600 sampled windows

NOMINAL_LEVELS = [0.80, 0.90, 0.95]


def collect_predictions(model, loader, device, num_batches):
    """Return (z_true, mu_hat, sigma_hat) numpy arrays of shape (N, 6)."""
    model.eval()
    zs, mus, sigmas = [], [], []
    with torch.no_grad():
        for i, (x, y, m) in enumerate(loader):
            if i >= num_batches:
                break
            x, y, m = x.to(device), y.to(device), m.to(device)
            mu, logvar = model(x, m)
            sigma = torch.exp(0.5 * logvar)
            zs.append(y.cpu().numpy())
            mus.append(mu.cpu().numpy())
            sigmas.append(sigma.cpu().numpy())
    return (np.concatenate(zs, 0),
            np.concatenate(mus, 0),
            np.concatenate(sigmas, 0))


def coverage_table(z, mu, sigma, levels):
    """Compute per-dimension and average empirical coverage at each level.

    Returns dict: level -> (per_dim_coverage_array_shape_D, avg_coverage)
    """
    N, D = z.shape
    abs_err = np.abs(z - mu)                          # (N, D)

    results = {}
    for lvl in levels:
        alpha = 1.0 - lvl
        z_crit = norm.ppf(1.0 - alpha / 2.0)          # two-sided
        inside = abs_err <= (z_crit * sigma)          # (N, D) bool
        per_dim = inside.mean(axis=0)                 # (D,)
        avg = float(per_dim.mean())
        results[lvl] = (per_dim, avg, z_crit)
    return results


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

    z, mu, sigma = collect_predictions(model, loader, device, NUM_EVAL_BATCHES)
    N = z.shape[0]
    print(f"\nCollected {N:,} (z, mu, sigma) triplets over "
          f"{FEATURE_DIM} style dimensions.")

    table = coverage_table(z, mu, sigma, NOMINAL_LEVELS)

    # ---------- Pretty print ----------
    print("\nEmpirical coverage of per-dimension credible intervals")
    print("(interval = mu_i +/- z_{alpha/2} * sigma_i)")
    print()
    header = "  Level  | " + "  ".join(f" z{i+1} " for i in range(FEATURE_DIM)) + " |   Avg "
    print(header)
    print("-" * len(header))
    for lvl in NOMINAL_LEVELS:
        per_dim, avg, _ = table[lvl]
        cells = "  ".join(f"{c:0.3f}" for c in per_dim)
        print(f"  {lvl:.2f}   | {cells} |  {avg:0.3f}")

    # ---------- Sanity check vs claim ----------
    print("\nClaim in reviewer reply: avg coverage = 0.798 / 0.901 / 0.948")
    print(f"Measured            : "
          f"{table[0.80][1]:0.3f} / {table[0.90][1]:0.3f} / {table[0.95][1]:0.3f}")
    deviations = [abs(table[lvl][1] - lvl) for lvl in NOMINAL_LEVELS]
    max_dev = max(deviations)
    print(f"Max |avg - nominal| = {max_dev:0.3f}  "
          f"({'<= 0.01' if max_dev <= 0.01 else '> 0.01: revise the +/- 1% wording'})")


if __name__ == "__main__":
    main()
