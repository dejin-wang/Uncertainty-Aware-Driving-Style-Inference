import csv
import numpy as np

from problem_env1 import MyHighwayEnv


# ============================================================
# IDM Parameter Class
# ============================================================
class IDMParams:
    def __init__(
        self,
        v0=25.0,
        T=1.5,
        a_max=1.0,
        b=2.0,
        s0=2.0,
        delta=4.0
    ):
        self.v0 = v0
        self.T = T
        self.a_max = a_max
        self.b = b
        self.s0 = s0
        self.delta = delta


# ============================================================
# IDM Longitudinal Acceleration
# ============================================================
def idm_acceleration(v_ego: float, v_lead: float, gap: float, params: IDMParams) -> float:
    v_ego = float(v_ego)
    v_lead = float(v_lead)
    gap = float(gap)

    dv = v_ego - v_lead

    s_star = params.s0 + max(
        0.0,
        v_ego * params.T + v_ego * dv / (2.0 * np.sqrt(params.a_max * params.b))
    )

    accel_free = (v_ego / params.v0) ** params.delta
    accel_int = (s_star / max(gap, 1e-3)) ** 2

    return float(params.a_max * (1.0 - accel_free - accel_int))


# ============================================================
# Find Lead Vehicle in SAME Lane
# ============================================================
def find_lead_vehicle(kinematics: np.ndarray, lane_width: float = 4.0):
    M = np.asarray(kinematics, dtype=np.float32)
    ego_x, ego_y = float(M[0, 0]), float(M[0, 1])

    best_idx, best_dx = None, np.inf

    for i in range(1, M.shape[0]):
        x_i, y_i = float(M[i, 0]), float(M[i, 1])
        dx = x_i - ego_x

        if dx <= 0:
            continue
        if abs(y_i - ego_y) > lane_width / 2.0:
            continue

        if dx < best_dx:
            best_dx = dx
            best_idx = i

    if best_idx is None:
        return None, np.inf
    return best_idx, best_dx


# ============================================================
# Find Lead and Follower in TARGET lane
# ============================================================
def find_lead_and_follow(M, ego_idx, target_lane, lane_width=4.0):
    """
    Returns: lead_idx, lead_gap, follow_idx, follow_gap
    """
    ego_x, ego_y = M[ego_idx, 0], M[ego_idx, 1]
    N = M.shape[0]

    lane_center = target_lane * lane_width + lane_width / 2

    lead_idx, lead_dx = None, np.inf
    follow_idx, follow_dx = None, np.inf

    for i in range(N):
        if i == ego_idx:
            continue

        x_i, y_i = M[i, 0], M[i, 1]

        # Must be inside target lane
        if abs(y_i - lane_center) > lane_width / 2:
            continue

        dx = x_i - ego_x

        if dx > 0:  # lead
            if dx < lead_dx:
                lead_dx = dx
                lead_idx = i
        else:  # follower
            if -dx < follow_dx:
                follow_dx = -dx
                follow_idx = i

    return lead_idx, lead_dx, follow_idx, follow_dx


# ============================================================
# MOBIL Lane Change Decision
# ============================================================
def mobil_decision(M, ego_lane, idm_params, politeness=0.2, a_thr=0.1, b_safe=2.0):
    decisions = {"left": False, "right": False}

    ego_x, ego_y, vx_ego, vy_ego = M[0]
    v_ego = float(np.hypot(vx_ego, vy_ego))

    # Current lane acceleration
    lead_idx_old, gap_old = find_lead_vehicle(M)
    if lead_idx_old is None:
        v_lead_old, gap_old = v_ego, 1e6
    else:
        vx_l, vy_l = M[lead_idx_old, 2], M[lead_idx_old, 3]
        v_lead_old = float(np.hypot(vx_l, vy_l))

    a_old = idm_acceleration(v_ego, v_lead_old, gap_old, idm_params)

    # Candidate lanes
    candidate_lanes = []
    if ego_lane > 0:
        candidate_lanes.append(("left", ego_lane - 1))
    if ego_lane < 2:
        candidate_lanes.append(("right", ego_lane + 1))

    for side, target_lane in candidate_lanes:

        lead_idx_new, gap_new, follow_idx, gap_follow = find_lead_and_follow(
            M, 0, target_lane
        )

        # New lane ego acceleration
        if lead_idx_new is None:
            v_lead_new, gap_new = v_ego, 1e6
        else:
            vx_l, vy_l = M[lead_idx_new, 2], M[lead_idx_new, 3]
            v_lead_new = float(np.hypot(vx_l, vy_l))

        a_new = idm_acceleration(v_ego, v_lead_new, gap_new, idm_params)

        # Follower safety
        if follow_idx is not None:
            vx_f, vy_f = M[follow_idx, 2], M[follow_idx, 3]
            v_follow = float(np.hypot(vx_f, vy_f))

            a_follow_new = idm_acceleration(
                v_follow, v_ego, max(gap_follow, 1e-3), idm_params
            )

            if a_follow_new < -b_safe:
                continue
        else:
            a_follow_new = 0.0

        mobil_gain = (a_new - a_old) + politeness * (a_follow_new)

        if mobil_gain > a_thr:
            decisions[side] = True

    return decisions


# ============================================================
# Final IDM + MOBIL Policy
# ============================================================
def idm_policy(obs: dict, idm_params: IDMParams) -> np.ndarray:
    M = np.asarray(obs["obs"], dtype=np.float32)

    ego_x, ego_y, vx_ego, vy_ego = M[0]
    v_ego = float(np.hypot(vx_ego, vy_ego))

    # Lane index
    ego_lane = int(ego_y // 4.0)

    # MOBIL decision
    mobil = mobil_decision(M, ego_lane, idm_params)

    if mobil["left"]:
        steering = -1.0
    elif mobil["right"]:
        steering = 1.0
    else:
        steering = 0.0

    # IDM longitudinal
    lead_idx, gap = find_lead_vehicle(M)
    if lead_idx is None:
        v_lead, gap = v_ego, 1e6
    else:
        vx_l, vy_l = M[lead_idx, 2], M[lead_idx, 3]
        v_lead = float(np.hypot(vx_l, vy_l))

    a_long = idm_acceleration(v_ego, v_lead, gap, idm_params)

    return np.array([a_long, steering], dtype=np.float32)


# ============================================================
# Rollout
# ============================================================
def run_idm_rollout(
    num_episodes: int = 20,
    csv_path: str = "IDM_MOBIL_obs_and_rewards.csv"
):
    env = MyHighwayEnv(config={})
    idm_params = IDMParams()

    total_rewards = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Step", "Reward", "Obs(flat)", "Style"])

        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            step = 0
            ep_reward = 0.0

            while not done:
                action = idm_policy(obs, idm_params)
                action = np.clip(action, env.action_space.low, env.action_space.high)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_reward += float(reward)

                writer.writerow([
                    ep,
                    step,
                    float(reward),
                    obs["obs"].flatten().tolist(),
                    obs["style"].tolist(),
                ])

                step += 1

            total_rewards.append(ep_reward)
            print(f"Episode {ep+1}: total reward = {ep_reward:.3f}")

    print("=======================================")
    print(f"Average reward: {float(np.mean(total_rewards)):.3f}")
    print(f"Max reward:     {float(np.max(total_rewards)):.3f}")
    print(f"Min reward:     {float(np.min(total_rewards)):.3f}")
    print("CSV saved to:", csv_path)


if __name__ == "__main__":
    run_idm_rollout()
