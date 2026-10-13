import csv
import numpy as np

from problem_env1 import MyHighwayEnv


class IDMParams:
    def __init__(
        self,
        v0=25.0,       # desired speed [m/s]
        T=1.5,         # desired time headway [s]
        a_max=1.0,     # maximum acceleration [m/s^2]
        b=2.0,         # comfortable deceleration [m/s^2]
        s0=2.0,        # minimum gap [m]
        delta=4.0      # acceleration exponent
    ):
        self.v0 = v0
        self.T = T
        self.a_max = a_max
        self.b = b
        self.s0 = s0
        self.delta = delta


def idm_acceleration(v_ego: float, v_lead: float, gap: float, params: IDMParams) -> float:
    """
    Scalar IDM longitudinal acceleration.
    v_ego, v_lead, gap are all floats in meters / m/s.
    """
    v_ego = float(v_ego)
    v_lead = float(v_lead)
    gap = float(gap)

    dv = v_ego - v_lead  # positive if ego is faster

    s_star = params.s0 + np.maximum(
        0.0,
        v_ego * params.T + v_ego * dv / (2.0 * np.sqrt(params.a_max * params.b))
    )

    accel_free = (v_ego / params.v0) ** params.delta
    accel_int = (s_star / np.maximum(gap, 1e-3)) ** 2

    a = params.a_max * (1.0 - accel_free - accel_int)
    return float(a)


def find_lead_vehicle(kinematics: np.ndarray, lane_width: float = 4.0):
    """
    Find the closest lead vehicle in the same lane.

    kinematics: shape (N, 4), rows: [x, y, vx, vy]
    Returns:
        lead_idx: int or None
        gap: float, distance in x from ego to lead (meters)
    """
    M = np.asarray(kinematics, dtype=np.float32)
    ego_x, ego_y = float(M[0, 0]), float(M[0, 1])

    best_idx = None
    best_dx = np.inf

    for i in range(1, M.shape[0]):
        x_i = float(M[i, 0])
        y_i = float(M[i, 1])
        dx = x_i - ego_x

        # Must be in front
        if dx <= 0.0:
            continue

        # Roughly same lane in y
        if abs(y_i - ego_y) > lane_width / 2.0:
            continue

        if dx < best_dx:
            best_dx = dx
            best_idx = i

    if best_idx is None:
        return None, np.inf
    else:
        return best_idx, best_dx


def idm_policy(obs: dict, idm_params: IDMParams) -> np.ndarray:
    """
    IDM-based policy for MyHighwayEnv.

    obs is a dict:
        {
            "obs": (4, 4) array of [x, y, vx, vy] for ego and other vehicles,
            "style": (6,) style vector (unused here but kept for logging)
        }

    Returns:
        action: np.array([acc, steering], dtype=float32)
    """
    M = np.asarray(obs["obs"], dtype=np.float32)

    # Ego state
    ego_x, ego_y, vx_ego, vy_ego = [float(v) for v in M[0]]
    v_ego = float(np.hypot(vx_ego, vy_ego))

    # Find lead vehicle in the same lane
    lead_idx, gap = find_lead_vehicle(M)

    if lead_idx is None:
        # No car ahead in the same lane: free-road behavior
        v_lead = v_ego
        gap = 1e6
    else:
        _, _, vx_lead, vy_lead = [float(v) for v in M[lead_idx]]
        v_lead = float(np.hypot(vx_lead, vy_lead))

    # IDM longitudinal acceleration
    a_long = idm_acceleration(v_ego, v_lead, gap, idm_params)

    # Lateral control: keep lane, no steering by default
    steering = 0.0

    return np.array([a_long, steering], dtype=np.float32)


def run_idm_rollout(
    num_episodes: int = 100,
    csv_path: str = "IDM_obs_and_rewards.csv"
):
    """
    Run IDM-controlled rollouts on MyHighwayEnv and save data to CSV.
    """
    # Instantiate your custom environment
    env = MyHighwayEnv(config={})

    # IDM parameters (you can tune v0, T, etc. if needed)
    idm_params = IDMParams(
        v0=25.0,
        T=1.5,
        a_max=1.0,
        b=2.0,
        s0=2.0,
        delta=4.0,
    )

    total_rewards = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Step", "Reward", "Obs(flat)", "Style"])

        for ep in range(num_episodes):
            obs, info = env.reset()
            print(obs)
            done = False
            step = 0
            ep_reward = 0.0

            while not done:
                # Compute IDM action
                action = idm_policy(obs, idm_params)

                # Ensure action is within the environment's action bounds
                action = np.clip(action, env.action_space.low, env.action_space.high)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_reward += float(reward)

                writer.writerow([
                    ep,
                    step,
                    float(reward),
                    np.asarray(obs["obs"], dtype=np.float32).flatten().tolist(),
                    np.asarray(obs["style"], dtype=np.float32).tolist(),
                ])

                step += 1

            total_rewards.append(ep_reward)
            print(f"Episode {ep + 1}: total reward = {ep_reward:.3f}")

    total_rewards = np.asarray(total_rewards, dtype=np.float32)
    print("========================================")
    print(f"Average reward: {float(np.mean(total_rewards)):.3f}")
    print(f"Max reward:     {float(np.max(total_rewards)):.3f}")
    print(f"Min reward:     {float(np.min(total_rewards)):.3f}")
    print("CSV saved to:", csv_path)


if __name__ == "__main__":
    run_idm_rollout()
