"""
Held-out behavioral-proxy analysis for NGSIM US-101 on-ramp vehicles.

Uses existing Stage-1 clusters in `ngsim_reward_clusters.csv` and computes:
  1. per-vehicle posterior means  mu_bar^(n)  (M=200 windows)
  2. held-out trajectory statistics NOT used in clustering
  3. Spearman rho, bootstrap 95% CI, Benjamini-Hochberg adjusted p-values

Comparisons (paper Table tab:ngsim_heldout):
  mu_bar_1  vs  merge-completion time
  mu_bar_2  vs  lane-change duration
  mu_bar_3  vs  minimum accepted front/rear gap at lane crossing
"""

import csv
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from torch.utils.data import DataLoader

from cluster_style_ngsim import (
    INPUT_NGSIM,
    CHECKPOINT,
    COL_NAMES,
    DT,
    FT_TO_M,
    SEGMENT_M,
    WINDOW_MIN_LEN,
    WINDOW_MAX_LEN,
    load_ngsim,
)
from IDM_test import (
    TimeSeriesTransformer,
    TrajectoryStreamDataset,
    collate_fn,
    evaluate_style_values,
)

CLUSTERS_CSV = "ngsim_reward_clusters.csv"
OUTPUT_CSV = "ngsim_heldout_results.csv"
NUM_SAMPLES = 200
BATCH_SIZE = 128
N_BOOT = 2000
RNG = np.random.default_rng(42)

VY_LC_THRESH = 0.10   # m/s, lateral-motion threshold for LC duration
MAINLINE_LANE = 6     # first mainline lane after on-ramp lane 7


def build_vehicle_trajectory(df_all: pd.DataFrame, vid: int,
                             frame_to_arr: dict) -> list | None:
    y_min = df_all["Local_Y_m"].min()
    ego_rows = df_all[
        (df_all["Vehicle_ID"] == vid)
        & df_all["Local_Y_rel"].between(0.0, SEGMENT_M)
    ].sort_values("Frame_ID")

    if len(ego_rows) < WINDOW_MIN_LEN:
        return None

    trajectory = []
    for _, row in ego_rows.iterrows():
        fid = int(row["Frame_ID"])
        arr = frame_to_arr.get(fid)
        if arr is None:
            continue

        mask = arr[:, 0] != vid
        others = arr[mask]
        if len(others) < 3:
            continue

        ego_x, ego_y = row["Local_Y_m"], row["Local_X_m"]
        ego_vx, ego_vy = row["v_Vel_ms"], row["vy_ms"]

        dx = others[:, 1] - ego_x
        dy = others[:, 2] - ego_y
        dists = np.hypot(dx, dy)
        idx = np.argpartition(dists, 3)[:3]
        idx = idx[np.argsort(dists[idx])]
        neighbors = others[idx, 1:5]

        obs = np.empty((4, 4), dtype=np.float32)
        obs[0] = [ego_x, ego_y, ego_vx, ego_vy]
        obs[1:4] = neighbors
        trajectory.append(obs.flatten())

    return trajectory if len(trajectory) >= WINDOW_MIN_LEN else None


def infer_vehicle_style(model, trajectory, device) -> np.ndarray:
    placeholder = np.zeros(6, dtype=np.float32)
    traj = [(s.astype(np.float32), placeholder) for s in trajectory]
    dataset = TrajectoryStreamDataset(
        [traj],
        state_dim=16,
        feature_dim=6,
        window_min_len=WINDOW_MIN_LEN,
        window_max_len=WINDOW_MAX_LEN,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    return evaluate_style_values(model, loader, device, num_samples=NUM_SAMPLES)


def rear_gap_m(df_frame: pd.DataFrame, ego_row) -> float:
    """Gap to following vehicle at the same frame, if available."""
    fol_id = int(ego_row["Following"])
    if fol_id <= 0:
        return np.nan
    fol = df_frame[df_frame["Vehicle_ID"] == fol_id]
    if fol.empty:
        return np.nan
    fol = fol.iloc[0]
    gap = abs(float(ego_row["Local_Y_m"]) - float(fol["Local_Y_m"]))
    gap -= 0.5 * (float(ego_row["v_Length"]) + float(fol["v_Length"])) * FT_TO_M
    return max(gap, 0.0)


def compute_heldout_metrics(df_seg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vid, g in df_seg.groupby("Vehicle_ID"):
        g = g.sort_values("Frame_ID").reset_index(drop=True)
        t0 = float(g["Global_Time"].iloc[0])

        mainline = g[g["Lane_ID"] <= MAINLINE_LANE]
        if mainline.empty:
            merge_time = np.nan
        else:
            merge_time = (float(mainline["Global_Time"].iloc[0]) - t0) / 1000.0

        lc_durations = []
        min_gaps = []
        lanes = g["Lane_ID"].to_numpy()
        vy = g["vy_ms"].to_numpy()
        sh = g["Space_Hw_m"].replace(0.0, np.nan).to_numpy()

        for i in range(1, len(g)):
            if lanes[i] == lanes[i - 1]:
                continue

            lo = i
            while lo > 0 and abs(vy[lo]) > VY_LC_THRESH:
                lo -= 1
            hi = i
            while hi + 1 < len(g) and abs(vy[hi]) > VY_LC_THRESH:
                hi += 1
            lc_durations.append((hi - lo + 1) * DT)

            row = g.iloc[i]
            front = float(sh[i]) if np.isfinite(sh[i]) else np.nan
            rear = rear_gap_m(g, row)
            gaps = [x for x in (front, rear) if np.isfinite(x)]
            if gaps:
                min_gaps.append(min(gaps))

        rows.append({
            "Vehicle_ID": int(vid),
            "merge_completion_time_s": merge_time,
            "lane_change_duration_s": (
                float(np.mean(lc_durations)) if lc_durations else np.nan
            ),
            "min_accepted_gap_m": (
                float(np.min(min_gaps)) if min_gaps else np.nan
            ),
            "n_lane_changes": len(lc_durations),
        })
    return pd.DataFrame(rows)


def bootstrap_spearman_ci(x, y, n_boot=N_BOOT, rng=RNG):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    rho, p = spearmanr(x, y)

    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b], _ = spearmanr(x[idx], y[idx])

    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(rho), float(p), float(lo), float(hi)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clusters = pd.read_csv(CLUSTERS_CSV)
    vehicle_ids = clusters["Vehicle_ID"].astype(int).tolist()
    print(f"Loaded {len(vehicle_ids)} clustered vehicles from {CLUSTERS_CSV}")

    df = load_ngsim()
    y_min = df["Local_Y_m"].min()
    df = df.assign(Local_Y_rel=df["Local_Y_m"] - y_min)
    df_seg = df[
        df["Vehicle_ID"].isin(vehicle_ids)
        & df["Local_Y_rel"].between(0.0, SEGMENT_M)
    ].copy()

    print("Computing held-out trajectory statistics ...")
    heldout = compute_heldout_metrics(df_seg)

    arr_cols = ["Vehicle_ID", "Local_Y_m", "Local_X_m", "v_Vel_ms", "vy_ms"]
    frame_to_arr = {
        int(fid): grp[arr_cols].to_numpy()
        for fid, grp in df.groupby("Frame_ID")
    }

    model = TimeSeriesTransformer().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    print(f"Loaded checkpoint: {CHECKPOINT}")

    style_rows = []
    print("Running per-vehicle transformer inference ...")
    for k, vid in enumerate(vehicle_ids, 1):
        traj = build_vehicle_trajectory(df, vid, frame_to_arr)
        if traj is None:
            print(f"  [{k:3d}/{len(vehicle_ids)}] vehicle {vid}: skipped (short traj)")
            continue
        mu = infer_vehicle_style(model, traj, device)
        style_rows.append({
            "Vehicle_ID": vid,
            **{f"mu_z{i+1}": float(mu[i]) for i in range(6)},
        })
        if k % 20 == 0 or k == len(vehicle_ids):
            print(f"  [{k:3d}/{len(vehicle_ids)}] done")

    styles = pd.DataFrame(style_rows)
    merged = styles.merge(heldout, on="Vehicle_ID", how="inner")
    merged = merged.merge(clusters[["Vehicle_ID", "Style"]], on="Vehicle_ID", how="left")
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved per-vehicle table -> {OUTPUT_CSV}")

    comparisons = [
        ("mu_z1", "merge_completion_time_s",
         r"$\bar{\mu}_1$ vs merge-completion time"),
        ("mu_z2", "lane_change_duration_s",
         r"$\bar{\mu}_2$ vs lane-change duration"),
        ("mu_z3", "min_accepted_gap_m",
         r"$\bar{\mu}_3$ vs minimum accepted gap"),
    ]

    print("\n========== Held-out Spearman correlations ==========")
    raw_p = []
    results = []
    for xcol, ycol, label in comparisons:
        sub = merged[[xcol, ycol]].dropna()
        rho, p, lo, hi = bootstrap_spearman_ci(
            sub[xcol].to_numpy(), sub[ycol].to_numpy()
        )
        raw_p.append(p)
        results.append((label, rho, lo, hi, p, len(sub)))
        print(f"{label}")
        print(f"  n = {len(sub)}")
        print(f"  Spearman rho = {rho:+.3f}")
        print(f"  95% CI       = [{lo:+.3f}, {hi:+.3f}]")
        print(f"  raw p        = {p:.4g}")

    _, p_adj, _, _ = multipletests(raw_p, method="fdr_bh")
    print("\nBenjamini-Hochberg adjusted p-values:")
    for (label, rho, lo, hi, p, n), padj in zip(results, p_adj):
        print(f"  {label}: p_adj = {padj:.4g}")


if __name__ == "__main__":
    main()
