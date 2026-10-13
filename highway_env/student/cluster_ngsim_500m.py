"""
Cluster NGSIM US-101 on-ramp drivers (0-500 m segment) into K=3 clusters
using the four reward-aligned features of MyHighwayEnv (problem_env1.py).

Features per vehicle (computed from data where Local_Y_rel in [0, 500] m):

    F_thw_below       = mean over frames of max(0, T0 - time_headway)
    F_impulsivity     = mean over frames of |delta(acceleration)|
    F_risk_distance   = mean over frames of max(0, d_safe - space_headway)
    F_rule_conformity = mean over frames of
                            max(0, v_min - v) + max(0, v - v_max)

Constants:
    T0 = 2.0 s,  d_safe = 30 m,  v_min = 20 m/s,  v_max = 30 m/s
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
COL_NAMES = [
    "Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time",
    "Local_X", "Local_Y", "Global_X", "Global_Y",
    "v_Length", "v_Width", "v_Class", "v_Vel", "v_Acc",
    "Lane_ID", "Preceding", "Following", "Space_Headway", "Time_Headway",
]

INPUT_PATH = "archive/trajectories-0750am-0805am.txt"
OUTPUT_CSV = "ngsim_500m_clusters.csv"

# Reward thresholds (must match problem_env1.py)
T0 = 2.0          # desired time-headway [s]
D_SAFE = 30.0     # safe gap [m]
V_MIN = 20.0      # m/s
V_MAX = 30.0      # m/s

FT_TO_M = 0.3048
SEGMENT_M = 500.0
K_CLUSTERS = 3


# ----------------------------------------------------------------------
# 1. Load NGSIM data
# ----------------------------------------------------------------------
print("Loading NGSIM trajectory file ...")
df = pd.read_csv(
    INPUT_PATH,
    sep=r"\s+",
    names=COL_NAMES,
    engine="c",
    dtype={
        "Vehicle_ID": np.int32, "Frame_ID": np.int32, "Lane_ID": np.int8,
    },
)
print(f"  rows = {len(df):,}, vehicles = {df['Vehicle_ID'].nunique():,}")


# ----------------------------------------------------------------------
# 2. Unit conversion
# ----------------------------------------------------------------------
df["Local_Y_m"] = df["Local_Y"] * FT_TO_M
df["v_Vel_ms"] = df["v_Vel"] * FT_TO_M
df["v_Acc_ms2"] = df["v_Acc"] * FT_TO_M
df["Space_Hw_m"] = df["Space_Headway"] * FT_TO_M


# ----------------------------------------------------------------------
# 3. EMA smoothing (paper convention: span=30 for v, span=120 for a)
# ----------------------------------------------------------------------
df = df.sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)
print("Smoothing velocity (span=30) and acceleration (span=120) ...")
df["v_Vel_ms_s"] = df.groupby("Vehicle_ID")["v_Vel_ms"].transform(
    lambda s: s.ewm(span=30, adjust=False).mean()
)
df["v_Acc_ms2_s"] = df.groupby("Vehicle_ID")["v_Acc_ms2"].transform(
    lambda s: s.ewm(span=120, adjust=False).mean()
)


# ----------------------------------------------------------------------
# 4. On-ramp drivers only (first observed lane == 7)
# ----------------------------------------------------------------------
first_lane = df.groupby("Vehicle_ID")["Lane_ID"].first()
onramp_ids = first_lane[first_lane == 7].index
df = df[df["Vehicle_ID"].isin(onramp_ids)].copy()
print(f"  on-ramp vehicles = {df['Vehicle_ID'].nunique():,}")


# ----------------------------------------------------------------------
# 5. Drop vehicles whose lane jumps by more than 1 in 0.1 s
# ----------------------------------------------------------------------
lane_diff = df.groupby("Vehicle_ID")["Lane_ID"].diff().abs()
bad_ids = df.loc[lane_diff > 1, "Vehicle_ID"].unique()
df = df[~df["Vehicle_ID"].isin(bad_ids)]
print(f"  after lane-jump cleaning = {df['Vehicle_ID'].nunique():,}")


# ----------------------------------------------------------------------
# 6. Per-frame reward features
# ----------------------------------------------------------------------
print("Computing per-frame reward features ...")

# Space headway: NGSIM stores 0 when no preceding vehicle -> treat as NaN
sh = df["Space_Hw_m"].replace(0.0, np.nan)
v_safe = df["v_Vel_ms_s"].clip(lower=1e-3)
time_headway = sh / v_safe

df["thw_below"] = (T0 - time_headway).clip(lower=0.0).fillna(0.0)
df["risk_dist"] = (D_SAFE - sh).clip(lower=0.0).fillna(0.0)
df["rule_conf"] = (
    (V_MIN - df["v_Vel_ms_s"]).clip(lower=0.0)
    + (df["v_Vel_ms_s"] - V_MAX).clip(lower=0.0)
)
df["impulsivity"] = (
    df.groupby("Vehicle_ID")["v_Acc_ms2_s"].diff().abs().fillna(0.0)
)


# ----------------------------------------------------------------------
# 7. Restrict to the 0-500 m segment
# ----------------------------------------------------------------------
y_min = df["Local_Y_m"].min()
df["Local_Y_rel"] = df["Local_Y_m"] - y_min
df_seg = df[df["Local_Y_rel"].between(0.0, SEGMENT_M)].copy()
print(f"  Local_Y raw range : [{df['Local_Y_m'].min():.1f}, {df['Local_Y_m'].max():.1f}] m")
print(f"  using relative Local_Y in [0, {SEGMENT_M:.0f}] m")


# ----------------------------------------------------------------------
# 8. Per-vehicle aggregation (mean of every reward feature)
# ----------------------------------------------------------------------
print("Aggregating per vehicle ...")
feat = df_seg.groupby("Vehicle_ID").agg(
    F_thw_below=("thw_below", "mean"),
    F_impulsivity=("impulsivity", "mean"),
    F_risk_distance=("risk_dist", "mean"),
    F_rule_conformity=("rule_conf", "mean"),
).reset_index()
feat = feat.dropna()
print(f"  feature rows = {len(feat):,}")


# ----------------------------------------------------------------------
# 9. Percentile transform (paper-style normalisation)
# ----------------------------------------------------------------------
FEAT_COLS = [
    "F_thw_below", "F_impulsivity",
    "F_risk_distance", "F_rule_conformity",
]
X = feat[FEAT_COLS].rank(method="average", pct=True).values


# ----------------------------------------------------------------------
# 10. K-means with K = 3
# ----------------------------------------------------------------------
print(f"Running K-means with K = {K_CLUSTERS} ...")
km = KMeans(n_clusters=K_CLUSTERS, n_init=20, random_state=42)
labels = km.fit_predict(X)
feat["Cluster"] = labels

sc = silhouette_score(X, labels)
db = davies_bouldin_score(X, labels)


# ----------------------------------------------------------------------
# 11. Cluster profile + driving-style label
# ----------------------------------------------------------------------
profile = feat.groupby("Cluster")[FEAT_COLS].mean().round(4)
sizes = feat["Cluster"].value_counts().sort_index()

# Higher mean reward-violation features -> more aggressive
score = profile.sum(axis=1)
order = score.sort_values().index.tolist()  # 0 = cautious, last = aggressive
style_names = {order[0]: "Cautious", order[-1]: "Aggressive"}
if K_CLUSTERS == 3:
    style_names[order[1]] = "Moderate"
feat["Style"] = feat["Cluster"].map(style_names)


# ----------------------------------------------------------------------
# 12. Report
# ----------------------------------------------------------------------
print("\n========== Clustering result (reward features, K = 3, p = 500 m) ==========")
print(f"Silhouette Coefficient : {sc:.4f}")
print(f"Davies-Bouldin Index   : {db:.4f}")
print(f"Cluster sizes          : {sizes.to_dict()}")

print("\nCluster profile (mean of raw reward features):")
print(profile.to_string())

print("\nStyle assignment (cluster -> label):")
for c in sorted(style_names):
    print(f"  Cluster {c}: {style_names[c]}  (n = {int(sizes[c])})")

feat.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved -> {OUTPUT_CSV}")
