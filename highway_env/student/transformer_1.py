import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import random

# === Step 1:  ===
def load_obs_style_trajectories(file_paths, state_dim=16, feature_dim=6):
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


# === Step 2:  window IterableDataset ===
class TrajectoryStreamDataset(IterableDataset):
    def __init__(self, all_trajectories, state_dim=16, feature_dim=6,
                 window_min_len=3, window_max_len=90):
        super().__init__()
        self.trajectories = all_trajectories
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.window_min_len = window_min_len
        self.window_max_len = window_max_len

    def __iter__(self):
        while True:  # 
            traj = random.choice(self.trajectories)
            traj_len = len(traj)
            if traj_len < self.window_min_len:
                continue

            # window
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
    def __init__(self, state_dim=16, feature_dim=6, d_model=128, nhead=8, num_layers=3, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 2*feature_dim (mean + logvar)
        self.output_proj = nn.Linear(d_model, 2 * feature_dim)
        self.d_model = d_model

    def forward(self, x, mask):
        B, T, _ = x.shape
        x = self.input_proj(x)

        # 
        position = torch.arange(T, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device) * -(np.log(10000.0) / self.d_model))
        pe = torch.zeros(T, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x = x + pe.unsqueeze(0)

        # Transformer Encoder
        x = self.encoder(x, src_key_padding_mask=~mask)

        # mask pooling
        x = x * mask.unsqueeze(-1).float()
        pooled = x.sum(1) / mask.sum(1, keepdim=True)

        out = self.output_proj(pooled)  # [B, 2*feature_dim]
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar


# === Step 5: Gaussian NLL Loss ===
def gaussian_nll_loss(mu, logvar, target):
    # mu, logvar, target: [B, D]
    return 0.5 * torch.mean((target - mu) ** 2 / logvar.exp() + logvar)



import torch
import torch.nn.functional as F

def gaussian_nll_loss_soft(mu, logvar, target,
                           a=[0,0,0,0,0.6,0.75],
                           b=[0.9,0.9,0.9,0.9,0.99,1],
                           k=1.0, lam=1.0):

    # convert a,b to tensors and broadcast
    a = torch.as_tensor(a, dtype=mu.dtype, device=mu.device)
    b = torch.as_tensor(b, dtype=mu.dtype, device=mu.device)

    # ensure broadcastable shape (1, D)
    if a.ndim == 1:
        a = a.unsqueeze(0)
    if b.ndim == 1:
        b = b.unsqueeze(0)

    # --- standard Gaussian NLL ---
    nll = 0.5 * ((target - mu) ** 2 / logvar.exp() + logvar)
    nll = nll.mean()

    # --- soft constraint penalty ---
    sigma = torch.exp(0.5 * logvar)
    lower = mu - k * sigma
    upper = mu + k * sigma

    penalty_lower = F.relu(a - lower)
    penalty_upper = F.relu(upper - b)

    penalty = (penalty_lower + penalty_upper).mean()

    total_loss = nll + lam * penalty
    return total_loss




# === Step 6:  ===
def train_model(model, loader, optimizer, device, steps_per_epoch=100):
    model.train()
    total = 0
    for _ in range(steps_per_epoch):
        x, y, m = next(iter(loader))  # 
        x, y, m = x.to(device), y.to(device), m.to(device)
        optimizer.zero_grad()
        mu, logvar = model(x, m)
        loss = gaussian_nll_loss_soft(mu, logvar, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / steps_per_epoch


# === Step 7:  scheduler ===
def main_train(
    data_file="PPO_obs_and_rewards10.csv",
    save_dir="transformer_weights1",
    num_epochs=200,
    batch_size=64,
    lr=1e-4,
    steps_per_epoch=100,
    device=None
):
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, "-", torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU")

    # 
    all_trajectories = load_obs_style_trajectories(data_file, state_dim=16, feature_dim=6)
    print(f"Loaded {len(all_trajectories)} trajectories")

    dataset = TrajectoryStreamDataset(all_trajectories)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # 
    model = TimeSeriesTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ===  ===
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=2e-6)

    # 
    loss_log = []
    loss_file = os.path.join(save_dir, "loss_log.csv")
    with open(loss_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "lr"])  # 

        for epoch in range(1, num_epochs + 1):
            loss = train_model(model, loader, optimizer, device, steps_per_epoch)
            scheduler.step()   # 

            # 
            current_lr = optimizer.param_groups[0]["lr"]
            loss_log.append((loss, current_lr))

            # 
            writer.writerow([epoch, loss, current_lr])
            f.flush()

            print(f"Epoch {epoch}/{num_epochs} - Loss: {loss:.6f} - LR: {current_lr:.8f}")

            # checkpoint
            if epoch % 50 == 0:
                ckpt_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"✅ Saved checkpoint: {ckpt_path}")

    # 
    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"🎉 Training finished! Final model saved at: {final_path}")

    return loss_log


# === Step 8: ===
if __name__ == "__main__":
    loss_log = main_train(
        data_file=[
            "PPO_obs_and_rewards1.csv",
            "PPO_obs_and_rewards.csv",
            "PPO_obs_and_rewards10.csv",
            "PPO_obs_and_rewards3.csv",
            "PPO_obs_and_rewards2.csv",
            "PPO_obs_and_rewards4.csv",
            "PPO_obs_and_rewards5.csv"
        ],
        save_dir="transformer_weights_gaussian",
        num_epochs=150000,
        batch_size=128,
        lr=1.5e-5,
        steps_per_epoch=150
    )
