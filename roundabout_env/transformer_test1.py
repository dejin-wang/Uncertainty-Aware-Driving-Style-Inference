import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

# === Step 1:  ===
def load_obs_style_trajectories(file_paths, state_dim=24, feature_dim=6):
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    all_trajectories = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            current_traj, last_episode = [], None
            for row in reader:
                episode = int(row["Episode"])
                if last_episode is None:
                    last_episode = episode

                if episode != last_episode:
                    if current_traj:
                        all_trajectories.append(current_traj)
                    current_traj = []
                    last_episode = episode

                try:
                    obs_flat = eval(row["Obs(flat)"])
                    style = eval(row["Style"])
                    if len(obs_flat) == state_dim and len(style) == feature_dim:
                        current_traj.append((obs_flat, style))
                except:
                    continue
            if current_traj:
                all_trajectories.append(current_traj)

    return all_trajectories


# === Step 2: IterableDataset ===
class TrajectoryStreamDataset(torch.utils.data.IterableDataset):
    def __init__(self, all_trajectories, state_dim=24, feature_dim=6,
                 window_min_len=3, window_max_len=90):
        super().__init__()
        self.trajectories = all_trajectories
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.window_min_len = window_min_len
        self.window_max_len = window_max_len

    def __iter__(self):
        while True:
            traj = random.choice(self.trajectories)
            traj_len = len(traj)
            if traj_len < self.window_min_len:
                continue
            start = random.randint(0, traj_len - self.window_min_len)
            end = random.randint(start + self.window_min_len,
                                 min(start + self.window_max_len, traj_len))
            window = traj[start:end]
            states = [step[0] for step in window]
            target = window[-1][1]
            yield torch.tensor(states, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


# === Step 3: collate ===
def collate_fn(batch):
    sequences, targets = zip(*batch)
    max_len = max(seq.size(0) for seq in sequences)
    padded = torch.stack([F.pad(seq, (0, 0, 0, max_len - len(seq))) for seq in sequences])
    mask = torch.tensor([[1]*len(seq)+[0]*(max_len - len(seq)) for seq in sequences], dtype=torch.bool)
    return padded, torch.stack(targets), mask


# === Step 4: Transformer (Gaussian Head) ===
class TimeSeriesTransformer(nn.Module):
    def __init__(self, state_dim=24, feature_dim=6, d_model=128, nhead=8, num_layers=3, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, 2 * feature_dim)  # Gaussian Head
        self.d_model = d_model

    def forward(self, x, mask):
        B, T, _ = x.shape
        x = self.input_proj(x)


        position = torch.arange(T, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device) * -(np.log(10000.0) / self.d_model))
        pe = torch.zeros(T, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x = x + pe.unsqueeze(0)

        # Transformer
        x = self.encoder(x, src_key_padding_mask=~mask)

        # pooling
        x = x * mask.unsqueeze(-1).float()
        pooled = x.sum(1) / mask.sum(1, keepdim=True)

        out = self.output_proj(pooled)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar


# === Step 5: Gaussian NLL ===
def gaussian_nll_loss(mu, logvar, target):
    var = torch.exp(logvar)
    nll = 0.5 * ((target - mu) ** 2 / var + logvar)
    return nll.mean()


# === Step 6: eval ===
def evaluate_loader(model, loader, device, max_batches=100, print_examples=5):
    model.eval()
    total_nll = 0.0
    mse_per_dim = None
    count = 0
    examples_shown = 0

    with torch.no_grad():
        for i, (x, y, m) in enumerate(loader):
            if i >= max_batches:
                break
            x, y, m = x.to(device), y.to(device), m.to(device)
            mu, logvar = model(x, m)
            loss = gaussian_nll_loss(mu, logvar, y)
            total_nll += loss.item()

            # MSE
            diff = (mu - y) ** 2
            batch_mse = diff.mean(dim=0).cpu().numpy()
            if mse_per_dim is None:
                mse_per_dim = batch_mse
            else:
                mse_per_dim += batch_mse
            count += 1

            if examples_shown < print_examples:
                print(f"\n--- Example Batch {examples_shown+1} ---")
                for j in range(min(3, x.size(0))):
                    target = y[j].cpu().numpy().round(4)
                    pred_mu = mu[j].cpu().numpy().round(4)
                    pred_sigma = torch.exp(0.5 * logvar[j]).cpu().numpy().round(4)
                    seq_len = int(m[j].sum().item())
                    print(f"Sample {j+1}:")
                    print(f"  Input length: {seq_len}")
                    print(f"  Target:      {target}")
                    print(f"  Pred μ:      {pred_mu}")
                    print(f"  Pred σ:      {pred_sigma}")
                    print(f"  MSE per dim: {((target - pred_mu) ** 2).round(6)}")
                examples_shown += 1

    avg_nll = total_nll / max_batches
    avg_mse = mse_per_dim / count

    print("\n🔎 Per-dimension Avg MSE (z1-z6):")
    for i, mse in enumerate(avg_mse, 1):
        print(f"  z{i}: {mse:.6f}")

    return avg_nll, avg_mse.mean()   # ✅


# === Step 7===
if __name__ == "__main__":
    test_file = "PPO_obs_and_rewards500.csv"
    #99950,118350, 145700
    checkpoint = "transformer_weights_gaussian/epoch_95600.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_trajectories = load_obs_style_trajectories(test_file, state_dim=24, feature_dim=6)
    dataset = TrajectoryStreamDataset(all_trajectories, state_dim=24, feature_dim=6,
                                      window_min_len=3, window_max_len=10)
    loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)

    print(f"✅ Loaded {len(all_trajectories)} trajectories for evaluation")

    model = TimeSeriesTransformer().to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✅ Loaded checkpoint: {checkpoint}")

    avg_nll, avg_mse = evaluate_loader(model, loader, device, max_batches=100, print_examples=5)
    print(f"\n🧪 Test Avg NLL: {avg_nll:.6f}")
    print(f"🧪 Test Avg MSE: {avg_mse:.6f}")

