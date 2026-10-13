"""
End-to-end NGSIM driving-style identification.

Stage 1 (re-clustering with reward features)
--------------------------------------------
For every on-ramp vehicle, compute the four features that drive the
reward of `MyHighwayEnv` (problem_env1.py):

    thw_below       = max(0, T0 - time_headway)             [s]
    impulsivity     = |delta(acceleration)| per frame       [m/s^2]
    risk_distance   = max(0, d_safe - space_headway)        [m]
    rule_conformity = max(0, v_min - v) + max(0, v - v_max) [m/s]

Each vehicle is summarised by the per-frame mean of every feature over
its trajectory in the [0, 500] m segment.  Values are converted to
empirical percentiles and clustered with k-means into K=3 groups.

Stage 2 (transformer style inference)
-------------------------------------
For each cluster we build, frame by frame, the 16-dim ego + 3-nearest
observation (swapped to highway-env axes), feed random sub-windows
through the trained Gaussian-head transformer and report the average
style vector (z1-z6, mean +- std over R runs).
"""

import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from IDM_test import (
    TimeSeriesTransformer,
    TrajectoryStreamDataset,
    collate_fn,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_NGSIM = "archive/trajectories-0750am-0805am.txt"
CHECKPOINT = "transformer_weights_gaussian/epoch_120400.pth"
CLUSTERS_OUT_CSV = "ngsim_reward_clusters.csv"
STYLE_OUT_CSV = "ngsim_cluster_transformer_style.csv"

# Reward thresholds (must match problem_env1.py)
T0 = 2.0          # desired time-headway [s]
D_SAFE = 30.0     # safe gap [m]
V_MIN = 20.0      # m/s
V_MAX = 30.0      # m/s

# Clustering
K_CLUSTERS = 3

# Transformer inference
NUM_RUNS = 10
NUM_SAMPLES_PER_RUN = 200
BATCH_SIZE = 128
WINDOW_MIN_LEN = 20
WINDOW_MAX_LEN = 50

FT_TO_M = 0.3048
DT = 0.1
STATE_DIM = 16
FEATURE_DIM = 6
SEGMENT_M = 500.0

COL_NAMES = [
    "Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time",
    "Local_X", "Local_Y", "Global_X", "Global_Y",
    "v_Length", "v_Width", "v_Class", "v_Vel", "v_Acc",
    "Lane_ID", "Preceding", "Following", "Space_Headway", "Time_Headway",
]


# ---------------------------------------------------------------------------
# Step 1: NGSIM loading
# ---------------------------------------------------------------------------
def load_ngsim() -> pd.DataFrame:
    print("Loading NGSIM trajectory file ...")
    df = pd.read_csv(
        INPUT_NGSIM, sep=r"\s+", names=COL_NAMES, engine="c",
        dtype={"Vehicle_ID": np.int32, "Frame_ID": np.int32},
    )
    print(f"  rows = {len(df):,}, vehicles = {df['Vehicle_ID'].nunique():,}")

    df["Local_X_m"] = df["Local_X"] * FT_TO_M
    df["Local_Y_m"] = df["Local_Y"] * FT_TO_M
    df["v_Vel_ms"] = df["v_Vel"] * FT_TO_M
    df["v_Acc_ms2"] = df["v_Acc"] * FT_TO_M
    df["Space_Hw_m"] = df["Space_Headway"] * FT_TO_M

    df = df.sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)

    # Smoothing (paper convention)
    df["v_Vel_ms_s"] = df.groupby("Vehicle_ID")["v_Vel_ms"].transform(
        lambda s: s.ewm(span=30, adjust=False).mean()
    )
    df["v_Acc_ms2_s"] = df.groupby("Vehicle_ID")["v_Acc_ms2"].transform(
        lambda s: s.ewm(span=120, adjust=False).mean()
    )

    # Lateral velocity (for transformer input)
    df["vy_ms"] = df.groupby("Vehicle_ID")["Local_X_m"].diff() / DT
    df["vy_ms"] = df["vy_ms"].fillna(0.0)
    df["vy_ms"] = df.groupby("Vehicle_ID")["vy_ms"].transform(
        lambda s: s.ewm(span=10, adjust=False).mean()
    )

    return df


# ---------------------------------------------------------------------------
# Step 2: Reward-aligned feature computation per vehicle
# ---------------------------------------------------------------------------
def compute_reward_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """Filter to on-ramp drivers, compute the four reward features per
    frame, then aggregate to one row per vehicle (means over [0, 500] m)."""
    # On-ramp filter (first lane == 7)
    first_lane = df_all.groupby("Vehicle_ID")["Lane_ID"].first()
    onramp_ids = first_lane[first_lane == 7].index
    df = df_all[df_all["Vehicle_ID"].isin(onramp_ids)].copy()
    print(f"  on-ramp vehicles = {df['Vehicle_ID'].nunique():,}")

    # Drop vehicles whose lane jumps by more than 1 between consecutive frames
    lane_diff = df.groupby("Vehicle_ID")["Lane_ID"].diff().abs()
    bad_ids = df.loc[lane_diff > 1, "Vehicle_ID"].unique()
    df = df[~df["Vehicle_ID"].isin(bad_ids)]
    print(f"  after lane-jump cleaning = {df['Vehicle_ID'].nunique():,}")

    # Per-frame reward features
    sh = df["Space_Hw_m"].replace(0.0, np.nan)        # treat 0 as no leader
    veh_speed = df["v_Vel_ms_s"].clip(lower=1e-3)
    time_headway = sh / veh_speed

    df["thw_below"] = (T0 - time_headway).clip(lower=0.0).fillna(0.0)
    df["risk_dist"] = (D_SAFE - sh).clip(lower=0.0).fillna(0.0)
    df["rule_conf"] = (
        (V_MIN - df["v_Vel_ms_s"]).clip(lower=0.0)
        + (df["v_Vel_ms_s"] - V_MAX).clip(lower=0.0)
    )
    df["impulsivity"] = (
        df.groupby("Vehicle_ID")["v_Acc_ms2_s"].diff().abs().fillna(0.0)
    )

    # Restrict to 0-500 m segment
    y_min = df["Local_Y_m"].min()
    df["Local_Y_rel"] = df["Local_Y_m"] - y_min
    df_seg = df[df["Local_Y_rel"].between(0.0, SEGMENT_M)].copy()

    # Per-vehicle aggregation
    agg = df_seg.groupby("Vehicle_ID").agg(
        F_thw_below=("thw_below", "mean"),
        F_impulsivity=("impulsivity", "mean"),
        F_risk_distance=("risk_dist", "mean"),
        F_rule_conformity=("rule_conf", "mean"),
    ).reset_index()
    print(f"  feature rows = {len(agg):,}")
    return agg


# ---------------------------------------------------------------------------
# Step 3: Percentile transform + K-means clustering
# ---------------------------------------------------------------------------
def cluster_reward_features(feat: pd.DataFrame, k: int = K_CLUSTERS) -> tuple:
    feat_cols = [
        "F_thw_below", "F_impulsivity",
        "F_risk_distance", "F_rule_conformity",
    ]
    X = feat[feat_cols].rank(method="average", pct=True).values

    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X)
    feat = feat.copy()
    feat["Cluster"] = labels

    sc = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)

    # Label clusters: highest summed-feature mean -> most aggressive
    profile = feat.groupby("Cluster")[feat_cols].mean()
    score = profile.sum(axis=1)
    order = score.sort_values().index.tolist()  # 0 = cautious, k-1 = aggressive
    style_names = {order[0]: "Cautious", order[-1]: "Aggressive"}
    if k == 3:
        style_names[order[1]] = "Moderate"
    feat["Style"] = feat["Cluster"].map(style_names)

    return feat, profile, sc, db, style_names


# ---------------------------------------------------------------------------
# Step 4: Trajectory construction for transformer
# ---------------------------------------------------------------------------
def build_cluster_trajectories(df_all: pd.DataFrame,
                               clusters_df: pd.DataFrame) -> dict:
    """For each cluster vehicle, build a list of 16-dim observations
    (ego + 3 nearest neighbors, highway-env axis convention)."""
    y_min = df_all["Local_Y_m"].min()
    df_all = df_all.assign(Local_Y_rel=df_all["Local_Y_m"] - y_min)

    cluster_map = dict(zip(
        clusters_df["Vehicle_ID"].astype(int),
        clusters_df["Cluster"].astype(int),
    ))

    print("Building per-frame position index ...")
    arr_cols = ["Vehicle_ID", "Local_Y_m", "Local_X_m", "v_Vel_ms", "vy_ms"]
    frame_to_arr = {
        int(fid): grp[arr_cols].to_numpy()
        for fid, grp in df_all.groupby("Frame_ID")
    }
    print(f"  unique frames = {len(frame_to_arr):,}")

    cluster_trajectories = {c: [] for c in range(K_CLUSTERS)}
    cluster_traj_lengths = {c: [] for c in range(K_CLUSTERS)}

    cluster_vids = set(cluster_map.keys())
    df_ego = df_all[
        df_all["Vehicle_ID"].isin(cluster_vids)
        & df_all["Local_Y_rel"].between(0.0, SEGMENT_M)
    ].copy()

    print(f"Building trajectories for {len(cluster_vids)} ego vehicles ...")
    for vid, ego_rows in df_ego.groupby("Vehicle_ID"):
        cid = cluster_map[int(vid)]
        ego_rows = ego_rows.sort_values("Frame_ID")

        frames = ego_rows["Frame_ID"].to_numpy()
        ego_x = ego_rows["Local_Y_m"].to_numpy()
        ego_y = ego_rows["Local_X_m"].to_numpy()
        ego_vx = ego_rows["v_Vel_ms"].to_numpy()
        ego_vy = ego_rows["vy_ms"].to_numpy()

        trajectory = []
        for k in range(len(frames)):
            arr = frame_to_arr.get(int(frames[k]))
            if arr is None:
                continue
            mask = arr[:, 0] != vid
            others = arr[mask]
            if len(others) < 3:
                continue

            dx = others[:, 1] - ego_x[k]
            dy = others[:, 2] - ego_y[k]
            dists = np.hypot(dx, dy)
            idx = np.argpartition(dists, 3)[:3]
            idx = idx[np.argsort(dists[idx])]
            neighbors = others[idx, 1:5]

            obs = np.empty((4, 4), dtype=np.float32)
            obs[0] = [ego_x[k], ego_y[k], ego_vx[k], ego_vy[k]]
            obs[1:4] = neighbors
            trajectory.append(obs.flatten())

        if len(trajectory) >= WINDOW_MIN_LEN:
            cluster_trajectories[cid].append(trajectory)
            cluster_traj_lengths[cid].append(len(trajectory))

    for c in sorted(cluster_trajectories):
        n = len(cluster_trajectories[c])
        if n:
            ml = float(np.mean(cluster_traj_lengths[c]))
            print(f"  cluster {c}: {n} trajectories, mean length = {ml:.1f}")
        else:
            print(f"  cluster {c}: 0 trajectories")

    return cluster_trajectories


# ---------------------------------------------------------------------------
# Step 5: Transformer inference helper
# ---------------------------------------------------------------------------
def evaluate_cluster(model, trajectories, device, num_samples) -> np.ndarray:
    placeholder = np.zeros(FEATURE_DIM, dtype=np.float32)
    trajs = [[(s.astype(np.float32), placeholder) for s in traj]
             for traj in trajectories]

    dataset = TrajectoryStreamDataset(
        trajs,
        state_dim=STATE_DIM, feature_dim=FEATURE_DIM,
        window_min_len=WINDOW_MIN_LEN, window_max_len=WINDOW_MAX_LEN,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model.eval()
    mus = []
    with torch.no_grad():
        for x, _, m in loader:
            x, m = x.to(device), m.to(device)
            mu, _ = model(x, m)
            for i in range(x.size(0)):
                mus.append(mu[i].cpu().numpy())
                if len(mus) >= num_samples:
                    break
            if len(mus) >= num_samples:
                break
    return np.mean(np.stack(mus), axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Stage 1: reward-feature clustering ----
    df = load_ngsim()
    print("Computing reward-aligned per-vehicle features ...")
    feat = compute_reward_features(df)
    print(f"Running K-means with K = {K_CLUSTERS} on reward percentile features ...")
    clusters_df, profile, sc, db, style_names = cluster_reward_features(feat)

    print("\n========== Reward-feature clustering result ==========")
    print(f"Silhouette Coefficient : {sc:.4f}")
    print(f"Davies-Bouldin Index   : {db:.4f}")
    print(f"Cluster sizes          : {clusters_df['Cluster'].value_counts().to_dict()}")
    print("\nCluster profile (mean of raw reward features):")
    print(profile.round(3).to_string())
    print("\nStyle assignment (cluster -> label):")
    for c in sorted(style_names):
        n = int((clusters_df['Cluster'] == c).sum())
        print(f"  Cluster {c}: {style_names[c]}  (n = {n})")

    clusters_df.to_csv(CLUSTERS_OUT_CSV, index=False)
    print(f"\nSaved clusters -> {CLUSTERS_OUT_CSV}")

    # ---- Stage 2: transformer style inference ----
    model = TimeSeriesTransformer().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    print(f"\nLoaded transformer checkpoint: {CHECKPOINT}")

    cluster_trajectories = build_cluster_trajectories(df, clusters_df)

    results = {}
    for c in sorted(cluster_trajectories):
        trajs = cluster_trajectories[c]
        if not trajs:
            continue
        label = style_names.get(c, f"Cluster {c}")
        print(f"\n===== Cluster {c} ({label}) "
              f"{len(trajs)} trajectories | {NUM_RUNS} runs x {NUM_SAMPLES_PER_RUN} samples =====")

        run_means = []
        for run in range(NUM_RUNS):
            mu = evaluate_cluster(model, trajs, device, NUM_SAMPLES_PER_RUN)
            run_means.append(mu)
            print(f"  run {run+1:2d}: mu = {np.round(mu, 4)}")

        run_means = np.stack(run_means)
        results[c] = {
            "label": label,
            "n_traj": len(trajs),
            "mean": run_means.mean(axis=0),
            "std": run_means.std(axis=0),
        }
        print(f"  mean: {np.round(results[c]['mean'], 4)}")
        print(f"  std : {np.round(results[c]['std'],  4)}")

    # ---- Summary ----
    print("\n========== Final summary ==========")
    header = (
        f"{'Cluster':<8} {'Style':<12} {'n':>4} | "
        + "  ".join(f"{'z'+str(i):>10}" for i in range(1, 7))
    )
    print(header)
    print("-" * len(header))
    for c in sorted(results):
        r = results[c]
        vals = "  ".join(
            f"{r['mean'][i]:>6.3f}+/-{r['std'][i]:.3f}" for i in range(6)
        )
        print(f"{c:<8} {r['label']:<12} {r['n_traj']:>4} | {vals}")

    with open(STYLE_OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["Cluster", "Style", "n_traj"]
            + [f"z{i}_mean" for i in range(1, 7)]
            + [f"z{i}_std" for i in range(1, 7)]
        )
        for c in sorted(results):
            r = results[c]
            w.writerow([c, r["label"], r["n_traj"]]
                       + list(r["mean"]) + list(r["std"]))
    print(f"\nSaved style results -> {STYLE_OUT_CSV}")


if __name__ == "__main__":
    main()
