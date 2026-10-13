"""
Quantitative justification of the factorized Gaussian uncertainty head.
Implements the two empirical checks described in Section (ii) of the reply
to Reviewer 1.

  Part 1 -- Residual correlation matrix
      rho_ij = Corr((z_i - mu_i), (z_j - mu_j))  on the held-out test set.
      Claim: max |rho_ij| < 0.21.

  Part 2 -- NLL gap to a low-rank Cholesky covariance head (rank = 2)
      Trains a second model with covariance
          Sigma = diag(exp(log_diag)) + V V^T,   V in R^{D x r}
      under identical conditions and reports |Delta NLL| on the test set.
      Claim: |Delta NLL| < 0.04 nats.

Usage:
    python gaussian_head_verification.py            # Part 1 only
    python gaussian_head_verification.py --train    # Part 1 + Part 2 (trains Cholesky)
    python gaussian_head_verification.py --eval     # Part 1 + Part 2 from an existing Cholesky ckpt
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_test1 import (
    TimeSeriesTransformer,
    load_obs_style_trajectories,
    TrajectoryStreamDataset,
    collate_fn,
)


# ---------------- Config ----------------
TEST_FILE = "PPO_obs_and_rewards_test.csv"
TRAIN_FILES = [
    "PPO_obs_and_rewards1.csv",
    "PPO_obs_and_rewards.csv",
    "PPO_obs_and_rewards10.csv",
    "PPO_obs_and_rewards3.csv",
    "PPO_obs_and_rewards2.csv",
    "PPO_obs_and_rewards4.csv",
    "PPO_obs_and_rewards5.csv",
]
FACTORIZED_CKPT = "transformer_weights_gaussian/epoch_120400.pth"
CHOLESKY_DIR = "transformer_weights_cholesky_r2"
CHOLESKY_CKPT = os.path.join(CHOLESKY_DIR, "final.pth")

STATE_DIM = 16
FEATURE_DIM = 6
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3
CHOLESKY_RANK = 2

BATCH_SIZE = 128
WINDOW_MIN = 20
WINDOW_MAX = 50
NUM_EVAL_BATCHES = 200          # ~25,600 samples

CHOLESKY_TRAIN_EPOCHS = 500     # warm-started from factorized backbone
STEPS_PER_EPOCH = 150
CHOLESKY_LR = 5e-6
GRAD_CLIP = 1.0


# ===========================================================
# Low-rank Cholesky covariance head
# ===========================================================
class CholeskyTransformer(nn.Module):
    """Same backbone as TimeSeriesTransformer; head outputs
    [mu, log_diag, V] with V in R^{D x rank}.

    Covariance: Sigma = diag(exp(log_diag)) + V V^T.
    """

    def __init__(self, state_dim=STATE_DIM, feature_dim=FEATURE_DIM,
                 d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
                 dropout=0.0, rank=CHOLESKY_RANK):
        super().__init__()
        self.feature_dim = feature_dim
        self.rank = rank
        self.d_model = d_model

        self.input_proj = nn.Linear(state_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        out_dim = feature_dim + feature_dim + feature_dim * rank
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, x, mask):
        B, T, _ = x.shape
        x = self.input_proj(x)
        position = torch.arange(T, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(T, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x = x + pe.unsqueeze(0)

        x = self.encoder(x, src_key_padding_mask=~mask)
        x = x * mask.unsqueeze(-1).float()
        pooled = x.sum(1) / mask.sum(1, keepdim=True)

        out = self.output_proj(pooled)
        D = self.feature_dim
        mu = out[:, :D]
        log_diag = out[:, D:2 * D]
        V = out[:, 2 * D:].reshape(B, D, self.rank)
        return mu, log_diag, V


def warm_start_cholesky(chol_model: CholeskyTransformer,
                        factorized_sd: dict) -> CholeskyTransformer:
    """Copy backbone weights (input_proj, encoder) from the factorized
    model into the Cholesky model. Head weights stay at random init
    because they have different shape."""
    new_sd = chol_model.state_dict()
    for k, v in factorized_sd.items():
        if "output_proj" in k:
            continue
        if k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k] = v.clone()
    chol_model.load_state_dict(new_sd)
    return chol_model


# ===========================================================
# Full Gaussian NLLs (with 2pi constant) for fair comparison
# ===========================================================
def factorized_full_nll(mu, logvar, target):
    D = mu.shape[-1]
    sq = (target - mu) ** 2
    nll = 0.5 * (sq / torch.exp(logvar)).sum(-1) \
        + 0.5 * logvar.sum(-1) \
        + 0.5 * D * math.log(2.0 * math.pi)
    return nll                                                   # (B,)


def cholesky_full_nll(mu, log_diag, V, target):
    cov = torch.diag_embed(torch.exp(log_diag)) + torch.bmm(V, V.transpose(1, 2))
    jitter = 1e-6 * torch.eye(cov.size(-1), device=cov.device).unsqueeze(0)
    dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov + jitter)
    return -dist.log_prob(target)                                # (B,)


# ===========================================================
# Part 1: residual correlation matrix
# ===========================================================
def part1_residual_correlation(model, loader, device, num_batches):
    model.eval()
    residuals = []
    with torch.no_grad():
        for i, (x, y, m) in enumerate(loader):
            if i >= num_batches:
                break
            x, y, m = x.to(device), y.to(device), m.to(device)
            mu, _ = model(x, m)
            residuals.append((y - mu).cpu().numpy())
    residuals = np.concatenate(residuals, axis=0)
    print(f"[Part 1] Collected {residuals.shape[0]:,} test residuals over "
          f"{residuals.shape[1]} dims.")

    corr = np.corrcoef(residuals, rowvar=False)
    print("\nResidual correlation matrix rho_ij = Corr((z_i - mu_i), (z_j - mu_j)):")
    print("         " + "    ".join(f" z{j+1} " for j in range(6)))
    for i in range(6):
        cells = "  ".join(f"{corr[i, j]:+0.3f}" for j in range(6))
        print(f"   z{i+1}    {cells}")

    off_diag = corr.copy()
    np.fill_diagonal(off_diag, 0.0)
    max_abs = float(np.max(np.abs(off_diag)))
    i_max, j_max = np.unravel_index(np.argmax(np.abs(off_diag)), off_diag.shape)
    print(f"\nMax |rho_ij| (off-diagonal) = {max_abs:.4f}  "
          f"(z{i_max+1} vs z{j_max+1})")
    return corr, max_abs


# ===========================================================
# Part 2: train Cholesky model
# ===========================================================
def train_cholesky(model, loader, device, num_epochs, lr,
                   steps_per_epoch, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data_iter = iter(loader)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total = 0.0
        for _ in range(steps_per_epoch):
            try:
                x, y, m = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y, m = next(data_iter)
            x, y, m = x.to(device), y.to(device), m.to(device)
            mu, log_diag, V = model(x, m)
            nll = cholesky_full_nll(mu, log_diag, V, y).mean()
            optimizer.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += nll.item()
        avg = total / steps_per_epoch
        print(f"[Part 2a] epoch {epoch:4d}/{num_epochs} | NLL = {avg:.4f}")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path)

    torch.save(model.state_dict(), save_path)
    print(f"[Part 2a] saved Cholesky model -> {save_path}")


def evaluate_nll(model, loader, device, num_batches, kind):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, (x, y, m) in enumerate(loader):
            if i >= num_batches:
                break
            x, y, m = x.to(device), y.to(device), m.to(device)
            if kind == "factorized":
                mu, logvar = model(x, m)
                nll = factorized_full_nll(mu, logvar, y)
            else:
                mu, log_diag, V = model(x, m)
                nll = cholesky_full_nll(mu, log_diag, V, y)
            total += nll.sum().item()
            n += nll.numel()
    return total / n


# ===========================================================
# Main
# ===========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="Train Cholesky model, then run NLL comparison.")
    parser.add_argument("--eval", action="store_true",
                        help="Skip training; load CHOLESKY_CKPT and compare NLL.")
    parser.add_argument("--epochs", type=int, default=CHOLESKY_TRAIN_EPOCHS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading test trajectories ...")
    test_trajs = load_obs_style_trajectories(
        TEST_FILE, state_dim=STATE_DIM, feature_dim=FEATURE_DIM
    )
    test_dataset = TrajectoryStreamDataset(
        test_trajs,
        state_dim=STATE_DIM, feature_dim=FEATURE_DIM,
        window_min_len=WINDOW_MIN, window_max_len=WINDOW_MAX,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Factorized model
    fact_model = TimeSeriesTransformer().to(device)
    fact_sd = torch.load(FACTORIZED_CKPT, map_location=device)
    fact_model.load_state_dict(fact_sd)
    print(f"Loaded factorized model: {FACTORIZED_CKPT}")

    # ---------- Part 1 ----------
    print("\n" + "=" * 64)
    print("Part 1: Residual correlation (factorized Gaussian model)")
    print("=" * 64)
    corr, max_abs_rho = part1_residual_correlation(
        fact_model, test_loader, device, NUM_EVAL_BATCHES
    )

    # ---------- Part 2 ----------
    if not (args.train or args.eval):
        print("\n[Part 2] skipped. Pass --train (or --eval if checkpoint exists).")
        return

    chol_model = CholeskyTransformer().to(device)

    if args.train:
        chol_model = warm_start_cholesky(chol_model, fact_sd)
        print("Warm-started Cholesky backbone from factorized checkpoint.")

        print("Loading training trajectories ...")
        train_trajs = load_obs_style_trajectories(
            TRAIN_FILES, state_dim=STATE_DIM, feature_dim=FEATURE_DIM
        )
        train_dataset = TrajectoryStreamDataset(
            train_trajs,
            state_dim=STATE_DIM, feature_dim=FEATURE_DIM,
            window_min_len=3, window_max_len=90,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn
        )

        print("\n" + "=" * 64)
        print(f"Part 2a: Training Cholesky model "
              f"(rank={CHOLESKY_RANK}, epochs={args.epochs})")
        print("=" * 64)
        train_cholesky(
            chol_model, train_loader, device,
            num_epochs=args.epochs, lr=CHOLESKY_LR,
            steps_per_epoch=STEPS_PER_EPOCH, save_path=CHOLESKY_CKPT,
        )
    else:
        if not os.path.exists(CHOLESKY_CKPT):
            raise FileNotFoundError(
                f"--eval was requested but {CHOLESKY_CKPT} does not exist. "
                f"Train first with --train."
            )
        chol_model.load_state_dict(torch.load(CHOLESKY_CKPT, map_location=device))
        print(f"Loaded Cholesky model: {CHOLESKY_CKPT}")

    print("\n" + "=" * 64)
    print("Part 2b: NLL comparison on held-out test set")
    print("=" * 64)
    nll_fact = evaluate_nll(fact_model, test_loader, device, NUM_EVAL_BATCHES, "factorized")
    nll_chol = evaluate_nll(chol_model, test_loader, device, NUM_EVAL_BATCHES, "cholesky")
    delta = nll_chol - nll_fact
    rel = 100.0 * abs(delta) / max(abs(nll_fact), 1e-12)

    print(f"\n  Factorized Gaussian NLL : {nll_fact:+.4f} nats / sample")
    print(f"  Low-rank Cholesky   NLL : {nll_chol:+.4f} nats / sample (rank={CHOLESKY_RANK})")
    print(f"  Delta NLL                : {delta:+.4f}  ( |Delta| = {abs(delta):.4f} )")
    print(f"  Relative change          : {rel:.2f} %")


if __name__ == "__main__":
    main()
