"""
Phase-switched IDM+MOBIL rollouts on highway-env, evaluated step-by-step
by the trained style-inference Transformer.

This script addresses the two concerns raised by Reviewer 1:

  (R1.1) Sensitivity under abrupt environmental change.
         Each rollout is split into two phases: the controller's IDM
         parameters switch at a fixed step (SWITCH_STEP). The plot
         marks the switch with a black vertical line; the transformer
         should react to the discontinuous change in observed
         behaviour.

  (R1.2) Step-by-step semantic persistence.
         For every step t >= WINDOW_MIN_LEN we slide a window through
         the trajectory and obtain (mu, sigma). The six style traces
         are plotted along with phase-coloured shading so the reader
         can verify that each trait remains semantically aligned with
         the active style during each phase, and that the temporal
         response to the switch is smooth rather than chaotic.

Two scenarios are reported in the same figure:

    A  Aggressive  ->  Cautious   (red)
    B  Cautious    ->  Aggressive (blue)

Outputs:
    scenario_style_evolution.png       (4x2 panel figure)
    scenario_style_predictions.csv     (per-step mu, sigma, phase)
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

from problem_env1 import MyHighwayEnv
from IDM_mobile import (
    IDMParams,
    idm_acceleration,
    find_lead_vehicle,
    mobil_decision,
)
from transformer_test1 import TimeSeriesTransformer


# ---------- Config ----------
CHECKPOINT = "transformer_weights_gaussian/epoch_120400.pth"
SAVE_FIG = "scenario_style_evolution.png"
SAVE_CSV = "scenario_style_predictions.csv"
SAVE_DYN_CSV = "scenario_dynamics.csv"

MAX_STEPS = 90
WINDOW_MIN_LEN = 20
WINDOW_MAX_LEN = 50
SWITCH_STEP = 45                          # phase change inside each rollout
ENV_VEHICLES = 15                         # denser traffic -> more lane-change opportunities

# Style parameter sets
# (AGGR was tuned down from v0=30, T=0.25, s0=1 because those
#  parameters tend to crash within the first few steps, leaving no
#  trajectory long enough to apply the WINDOW_MIN_LEN window.)
AGGR_PARAMS = IDMParams(v0=28.0, T=0.40, a_max=4.0, b=4.0, s0=2.0)
CAUT_PARAMS = IDMParams(v0=18.0, T=2.00, a_max=1.0, b=1.0, s0=4.0)

# MOBIL overrides used per phase.  The default a_thr=0.1 is too
# conservative for our short rollouts; the aggressive phase uses a much
# smaller threshold so lane changes actually fire when there is any
# benefit, while the cautious phase keeps the original threshold so the
# ego does not change lane out of style.
AGGR_MOBIL = dict(a_thr=0.02, politeness=0.1, b_safe=2.0)
CAUT_MOBIL = dict(a_thr=0.50, politeness=0.5, b_safe=2.0)

SCENARIOS = {
    "Aggressive -> Cautious": {
        "phases": [
            ("Aggressive", AGGR_PARAMS, AGGR_MOBIL),
            ("Cautious",    CAUT_PARAMS, CAUT_MOBIL),
        ],
        # Sweep these candidate env seeds in order; first one that
        # produces >= MIN_LC_IN_AGGR lane changes during the aggressive
        # phase AND survives at least MIN_TRAJ_LENGTH frames is selected.
        "seed_candidates": list(range(40)),
    },
    "Cautious -> Aggressive": {
        "phases": [
            ("Cautious",    CAUT_PARAMS, CAUT_MOBIL),
            ("Aggressive", AGGR_PARAMS, AGGR_MOBIL),
        ],
        "seed_candidates": [13, 4, 21, 8, 1, 29, 6, 33, 12, 18, 26, 47]
                            + list(range(50, 90)),
    },
}
MIN_LC_IN_AGGR = 1     # require at least one lane change during aggressive phase
MIN_TRAJ_LENGTH = 50   # require the trajectory to survive long enough for the window

Z_NAMES = [
    "z1 (Aggressiveness)",
    "z2 (Impulsivity)",
    "z3 (Risk Tolerance)",
    "z4 (Rule Conformity)",
    "z5 (Prospectiveness)",
    "z6 (Expertness)",
]

SCEN_COLORS = {
    "Aggressive -> Cautious": "tab:red",
    "Cautious -> Aggressive": "tab:blue",
}

# Light pastel shading used to indicate the active phase in the top
# dynamics panels (we apply it to the first listed scenario only to
# avoid double-shading conflicts in the joint plot).
PHASE_BG = {
    "Aggressive": "#fdecec",
    "Cautious":   "#ecf2fb",
}


# ---------- Custom IDM + MOBIL policy with tunable MOBIL threshold ----------
def custom_idm_mobil_policy(obs: dict,
                            idm_params: IDMParams,
                            a_thr: float = 0.1,
                            politeness: float = 0.2,
                            b_safe: float = 2.0) -> np.ndarray:
    """Same structure as IDM_mobile.idm_policy but exposes the MOBIL
    knobs so the scenario can dial in more aggressive lane-change
    behaviour."""
    M = np.asarray(obs["obs"], dtype=np.float32)
    ego_x, ego_y, vx_ego, vy_ego = M[0]
    v_ego = float(np.hypot(vx_ego, vy_ego))
    ego_lane = int(ego_y // 4.0)

    mobil = mobil_decision(M, ego_lane, idm_params,
                           politeness=politeness, a_thr=a_thr, b_safe=b_safe)
    if mobil["left"]:
        steering = -1.0
    elif mobil["right"]:
        steering = 1.0
    else:
        steering = 0.0

    lead_idx, gap = find_lead_vehicle(M)
    if lead_idx is None:
        v_lead, gap = v_ego, 1e6
    else:
        vx_l, vy_l = M[lead_idx, 2], M[lead_idx, 3]
        v_lead = float(np.hypot(vx_l, vy_l))
    a_long = idm_acceleration(v_ego, v_lead, gap, idm_params)
    return np.array([a_long, steering], dtype=np.float32)


# ---------- Phase-switched rollout ----------
def run_scenario_phased(env: MyHighwayEnv,
                        phases: list,
                        switch_step: int,
                        max_steps: int,
                        seed: int):
    """Roll out using phases[0] for the first `switch_step` env steps
    and phases[1] thereafter.  Each phase entry is a 3-tuple
    (name, IDMParams, MOBIL_kwargs).
    """
    try:
        obs, _ = env.reset(seed=seed)
    except TypeError:
        obs, _ = env.reset()

    seq = [np.asarray(obs["obs"], dtype=np.float32).flatten()]
    phase_record = []

    step = 0
    done = False
    cur_idx = 0
    cur_name, cur_params, cur_mobil = phases[cur_idx]
    phase_start = 0

    while not done and step < max_steps:
        if step >= switch_step and cur_idx < len(phases) - 1:
            phase_record.append({"name": cur_name,
                                 "start": phase_start,
                                 "end": step})
            cur_idx += 1
            cur_name, cur_params, cur_mobil = phases[cur_idx]
            phase_start = step

        action = custom_idm_mobil_policy(obs, cur_params, **cur_mobil)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, _, terminated, truncated, _ = env.step(action)
        seq.append(np.asarray(obs["obs"], dtype=np.float32).flatten())
        step += 1
        done = terminated or truncated

    phase_record.append({"name": cur_name, "start": phase_start, "end": step})
    return np.stack(seq, axis=0), phase_record


def find_seed_with_lane_change(env, phases, switch_step, max_steps,
                               candidates, min_lc, min_T):
    """Sweep candidate seeds; return the first whose rollout produces
    at least `min_lc` lane changes inside the AGGRESSIVE phase
    AND survives at least `min_T` frames."""
    best = None  # fallback: longest trajectory seen so far
    best_len = 0
    for s in candidates:
        obs_seq, phase_record = run_scenario_phased(
            env, phases, switch_step, max_steps, s
        )
        dyn = extract_dynamics(obs_seq)
        aggr = next((p for p in phase_record if p["name"] == "Aggressive"), None)
        aggr_lc = []
        if aggr is not None:
            aggr_lc = [int(t) for t in dyn["lc_steps"]
                       if aggr["start"] <= t < aggr["end"]]
        T = len(obs_seq)
        print(f"    seed = {s:3d}  T = {T:3d}  "
              f"aggressive-phase lane changes = {len(aggr_lc)} at {aggr_lc}")
        if T >= min_T and len(aggr_lc) >= min_lc:
            return s, obs_seq, phase_record
        if T > best_len:
            best_len = T
            best = (s, obs_seq, phase_record)
    print("    (warning) no seed met both criteria; using longest trajectory.")
    return best


# ---------- Sliding-window transformer inference ----------
def rolling_predict(model, obs_seq, win_min, win_max, device):
    T = obs_seq.shape[0]
    if T < win_min:
        return np.array([]), np.zeros((0, 6)), np.zeros((0, 6))
    ts, mus, sigmas = [], [], []
    model.eval()
    with torch.no_grad():
        for t in range(win_min, T + 1):
            start = max(0, t - win_max)
            window = obs_seq[start:t]
            x = torch.tensor(window, dtype=torch.float32,
                             device=device).unsqueeze(0)
            mask = torch.ones(1, window.shape[0], dtype=torch.bool, device=device)
            mu, logvar = model(x, mask)
            sigma = torch.exp(0.5 * logvar)
            ts.append(t)
            mus.append(mu[0].cpu().numpy())
            sigmas.append(sigma[0].cpu().numpy())
    return np.array(ts), np.stack(mus), np.stack(sigmas)


# ---------- Dynamics extraction ----------
def extract_dynamics(obs_seq: np.ndarray):
    T = obs_seq.shape[0]
    ego = obs_seq[:, :4]
    speed = np.hypot(ego[:, 2], ego[:, 3])
    lane = np.floor(ego[:, 1] / 4.0).astype(int)
    lane_change_steps = np.where(np.diff(lane) != 0)[0] + 1
    return {
        "steps": np.arange(T),
        "speed": speed,
        "lane": lane,
        "lc_steps": lane_change_steps,
    }


# ---------- Plot ----------
def plot_style_evolution(results: dict,
                         dynamics: dict,
                         phase_records: dict,
                         save_path: str):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    fig, axes = plt.subplots(4, 2, figsize=(14, 12.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})

    # The phase background is taken from one canonical scenario.  Use
    # the first one that actually produced predictions (some scenarios
    # may be skipped if all candidate seeds crashed early).
    canonical_scen = next((s for s in SCENARIOS if s in phase_records), None)
    if canonical_scen is None:
        print("No scenarios available to plot.")
        return
    canonical_phases = phase_records[canonical_scen]

    def add_phase_shading(ax):
        for ph in canonical_phases:
            ax.axvspan(ph["start"], ph["end"],
                       color=PHASE_BG.get(ph["name"], "#ffffff"),
                       alpha=0.6, zorder=0)
        # Black switch line(s)
        for ph in canonical_phases[:-1]:
            ax.axvline(x=ph["end"], color="black",
                       linestyle="-", linewidth=1.6, alpha=0.9, zorder=1)

    # ----- Row 0: dynamics -----
    ax_speed, ax_lane = axes[0, 0], axes[0, 1]
    for label, dyn in dynamics.items():
        color = SCEN_COLORS.get(label, "gray")
        ax_speed.plot(dyn["steps"], dyn["speed"], color=color,
                      linewidth=2.0, label=label, zorder=4)
        ax_lane.step(dyn["steps"], dyn["lane"], color=color,
                     linewidth=2.0, where="post", label=label, zorder=4)
        T_end = dyn["steps"][-1]
        ax_speed.scatter([T_end], [dyn["speed"][-1]], color=color,
                         marker="x", s=70, zorder=6)
        ax_lane.scatter([T_end], [dyn["lane"][-1]], color=color,
                        marker="x", s=70, zorder=6)

    for ax in (ax_speed, ax_lane):
        add_phase_shading(ax)
        ax.grid(True, linestyle="--", alpha=0.4)
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.set_title("Ego speed")
    ax_lane.set_ylabel("Lane index")
    ax_lane.set_title("Ego lane (step plot)")

    # ----- Rows 1-3: 6 style panels -----
    for i in range(6):
        r = 1 + i // 2
        c = i % 2
        ax = axes[r, c]
        add_phase_shading(ax)

        for label, (ts, mus, sigmas) in results.items():
            color = SCEN_COLORS.get(label, "gray")
            mu = np.clip(mus[:, i], 0.0, 1.0)
            sigma = sigmas[:, i]
            lo = np.clip(mu - sigma, 0.0, 1.0)
            hi = np.clip(mu + sigma, 0.0, 1.0)
            ax.fill_between(ts, lo, hi,
                            color=color, alpha=0.18, linewidth=0, zorder=2)
            ax.plot(ts, mu, color=color, linewidth=2.0,
                    label=label, zorder=3)
            for lc in dynamics[label]["lc_steps"]:
                ax.axvline(x=lc, color=color, linestyle=":",
                           linewidth=1.1, alpha=0.5, zorder=2)

        ax.set_title(Z_NAMES[i])
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.4)

    for ax in axes[-1, :]:
        ax.set_xlabel("Step")

    # ----- Top legend -----
    handles, labels = ax_speed.get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], color="black", linestyle="-", linewidth=1.6,
                   label=f"Phase switch (step {SWITCH_STEP})"),
        plt.Line2D([], [], color="black", linestyle=":", linewidth=1.5,
                   label="Lane change"),
        plt.Line2D([], [], color="black", marker="x", linestyle="",
                   markersize=9, label="Trajectory end"),
    ]
    # Phase-background swatches
    handles += [
        plt.Rectangle((0, 0), 1, 1, color=PHASE_BG["Aggressive"],
                      label=f"{canonical_scen.split(' ')[0]} phase (canonical)"),
        plt.Rectangle((0, 0), 1, 1, color=PHASE_BG["Cautious"],
                      label=f"{canonical_scen.split(' ')[-1]} phase (canonical)"),
    ]
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels,
               loc="upper center",
               ncol=4,
               frameon=False,
               bbox_to_anchor=(0.5, 1.005),
               fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure -> {save_path}")


# ---------- Main ----------
def main():
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TimeSeriesTransformer().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print(f"Loaded transformer checkpoint: {CHECKPOINT}")

    env = MyHighwayEnv(config={})
    # Increase traffic density so MOBIL has a chance to fire during the
    # short aggressive phase.  The observation still feeds the 4 nearest
    # neighbours into the transformer.
    env.config["vehicles_count"] = ENV_VEHICLES
    env.config["initial_spacing"] = 1.5
    env.env.unwrapped.configure(env.config)
    print(f"Env configured with vehicles_count = {env.config['vehicles_count']}")

    results = {}
    dynamics = {}
    phase_records = {}
    csv_rows = []

    for label, cfg in SCENARIOS.items():
        print(f"\n=== Scenario: {label} ===  (searching for a seed with lane changes)")
        seed_used, obs_seq, phase_record = find_seed_with_lane_change(
            env, cfg["phases"], SWITCH_STEP, MAX_STEPS,
            cfg["seed_candidates"], MIN_LC_IN_AGGR, MIN_TRAJ_LENGTH,
        )
        print(f"  selected seed = {seed_used}")
        print(f"  trajectory length = {len(obs_seq)} frames")
        for ph in phase_record:
            print(f"    phase {ph['name']:>11}: steps [{ph['start']:>3}, {ph['end']:>3})")

        ts, mus, sigmas = rolling_predict(
            model, obs_seq, WINDOW_MIN_LEN, WINDOW_MAX_LEN, device
        )
        if len(ts) == 0:
            print("  skip: trajectory too short for window"); continue

        results[label] = (ts, mus, sigmas)
        dynamics[label] = extract_dynamics(obs_seq)
        phase_records[label] = phase_record
        print(f"  rolling predictions = {len(ts)}")
        print(f"  lane changes        = {len(dynamics[label]['lc_steps'])} "
              f"at steps {dynamics[label]['lc_steps'].tolist()}")
        print(f"  final mu    = {np.round(mus[-1], 3)}")
        print(f"  final sigma = {np.round(sigmas[-1], 3)}")

        # Per-step CSV with phase tag
        def phase_at(step):
            for ph in phase_record:
                if ph["start"] <= step < ph["end"]:
                    return ph["name"]
            return phase_record[-1]["name"]

        for k, t in enumerate(ts):
            row = ([label, int(t), phase_at(int(t))]
                   + list(mus[k]) + list(sigmas[k]))
            csv_rows.append(row)

    # ---- Save CSV (style predictions) ----
    with open(SAVE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["scenario", "step", "phase"]
            + [f"mu_z{i+1}" for i in range(6)]
            + [f"sigma_z{i+1}" for i in range(6)]
        )
        w.writerows(csv_rows)
    print(f"\nSaved per-step predictions -> {SAVE_CSV}")

    # ---- Save CSV (per-step ego dynamics) ----
    dyn_rows = []
    for label, dyn in dynamics.items():
        lc_set = set(int(s) for s in dyn["lc_steps"])
        for t in range(len(dyn["steps"])):
            dyn_rows.append([
                label,
                int(dyn["steps"][t]),
                float(dyn["speed"][t]),
                int(dyn["lane"][t]),
                int(t in lc_set),
            ])
    with open(SAVE_DYN_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "step", "speed", "lane", "lane_change"])
        w.writerows(dyn_rows)
    print(f"Saved per-step dynamics    -> {SAVE_DYN_CSV}")

    plot_style_evolution(results, dynamics, phase_records, SAVE_FIG)


if __name__ == "__main__":
    main()
