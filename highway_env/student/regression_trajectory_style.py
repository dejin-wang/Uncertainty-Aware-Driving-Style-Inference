import os
import csv
import ast
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def load_state_action_trajectories(csv_path, state_dim=16, action_dim=2):
    episodes = []
    current = []
    last_ep = None

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["Episode"])

            if last_ep is None:
                last_ep = ep

            if ep != last_ep:
                if len(current) > 0:
                    episodes.append(current)
                current = []
                last_ep = ep

            try:
                state = np.array(ast.literal_eval(row["Obs(flat)"]), dtype=np.float32)
                action = np.array(ast.literal_eval(row["Action"]), dtype=np.float32)
            except:
                continue

            if len(state) == state_dim and len(action) == action_dim:
                current.append({"state": state, "action": action})

        if len(current) > 0:
            episodes.append(current)

    print(f"Loaded {len(episodes)} episodes from {csv_path}")
    return episodes



# ==============================================
# 2) Dataset → Transformer →
# ==============================================
class StyleActionDataset(torch.utils.data.IterableDataset):
    def __init__(self, episodes, state_dim=16, action_dim=2,
                 min_len=10, max_len=50):
        super().__init__()
        self.episodes = episodes
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.min_len = min_len
        self.max_len = max_len

    def __iter__(self):
        while True:
            traj = random.choice(self.episodes)
            if len(traj) < self.min_len + 1:
                continue

            T = random.randint(self.min_len, min(len(traj), self.max_len))
            window = traj[:T]

            seq_states = np.stack([step["state"] for step in window])
            last_state = seq_states[-1]
            action = window[-1]["action"]

            yield (
                torch.tensor(seq_states, dtype=torch.float32),  # sequence
                torch.tensor(last_state, dtype=torch.float32),   # last state
                torch.tensor(action, dtype=torch.float32),       # target action
            )



# ==============================================
# 3) collate：pad
# ==============================================
def collate_style_action(batch):
    seqs, last_states, actions = zip(*batch)

    max_len = max(seq.shape[0] for seq in seqs)

    padded = torch.stack([
        F.pad(seq, (0, 0, 0, max_len - len(seq)))
        for seq in seqs
    ])

    mask = torch.tensor([
        [1] * len(seq) + [0] * (max_len - len(seq))
        for seq in seqs
    ], dtype=torch.bool)

    return padded, torch.stack(last_states), torch.stack(actions), mask



# ==============================================
# 4) Transformer  (mu, logvar)
# ==============================================
class TimeSeriesTransformer(nn.Module):
    def __init__(self, state_dim=16, feature_dim=6,
                 d_model=128, nhead=8, num_layers=3, dropout=0.0):
        super().__init__()

        self.input_proj = nn.Linear(state_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, 2 * feature_dim)
        self.d_model = d_model

    def forward(self, x, mask):
        B, T, _ = x.shape
        x = self.input_proj(x)

        position = torch.arange(T, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device) * -(np.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(T, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        x = x + pe.unsqueeze(0)

        x = self.encoder(x, src_key_padding_mask=~mask)

        x = x * mask.unsqueeze(-1).float()
        pooled = x.sum(1) / mask.sum(1, keepdim=True)

        out = self.output_proj(pooled)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar



# ==============================================
# 5)  s_t + z → action
# ==============================================
class ActionRegressor(nn.Module):
    def __init__(self, input_dim=22, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),     # Hidden 1
            nn.Linear(256, 256), nn.ReLU(),           # Hidden 2
            nn.Linear(256, 256), nn.ReLU(),           # Hidden 3
            nn.Linear(256, output_dim)                # Output
        )

    def forward(self, x):
        return self.net(x)




# ==============================================
# 6) traiing
# ==============================================
def train_regression_model(
    csv_path,
    style_transformer_checkpoint,
    state_dim=16,
    feature_dim=6,
    action_dim=2,
    batch_size=128,
    lr=1e-4,
    total_steps=5000
):

    # === load episodes ===
    episodes = load_state_action_trajectories(csv_path, state_dim, action_dim)

    dataset = StyleActionDataset(episodes, state_dim, action_dim)
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=collate_style_action)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === load style transformer ===
    transformer = TimeSeriesTransformer(
        state_dim=state_dim, feature_dim=feature_dim
    ).to(device)
    transformer.load_state_dict(torch.load(style_transformer_checkpoint, map_location=device))
    transformer.eval()

    # === regression model ===
    regressor = ActionRegressor(
        input_dim=state_dim + feature_dim,
        output_dim=action_dim
    ).to(device)

    optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # === training loop ===
    for step, (seq, last_state, action, mask) in enumerate(loader):
        if step > total_steps:
            break

        seq, last_state, action, mask = \
            seq.to(device), last_state.to(device), action.to(device), mask.to(device)

        with torch.no_grad():
            mu, logvar = transformer(seq, mask)
            z = mu  # use mean

        reg_in = torch.cat([last_state, z], dim=-1)
        pred_action = regressor(reg_in)

        loss = criterion(pred_action, action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"[Step {step}] Loss = {loss.item():.6f}")

    torch.save(regressor.state_dict(), "regressor_with_style.pth")
    print("Saved regression model → regressor_with_style.pth")



# ==============================================
# 7) main
# ==============================================
if __name__ == "__main__":
    csv_path = "PPO_obs_and_rewards_regression.csv"
    style_checkpoint = "transformer_weights_gaussian/epoch_120400.pth"

    train_regression_model(
        csv_path=csv_path,
        style_transformer_checkpoint=style_checkpoint
    )
