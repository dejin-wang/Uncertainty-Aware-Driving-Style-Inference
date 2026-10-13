import numpy as np
from ray.rllib.env import EnvContext
import torch
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
import random
import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Dict, Box

from highway_env import utils


class MyHighwayEnv(gym.Env):
    def __init__(self, config):
        super(MyHighwayEnv, self).__init__()

        self.config = {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": 4,
                "see_behind": False,
            },
            "action": {
                "type": "ContinuousAction",
                "target_speeds": np.linspace(0, 32, 9),
            },
            "lanes_count": 3,
            "vehicles_count": 6,
            "duration": 30,  # [s]
            "initial_spacing": 2,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 3,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 1200,  # [px]
            "screen_height": 300,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": False

        }

        self.env = gym.make("highway-v0", render_mode="rgb_array")
        self.env.unwrapped.configure(self.config)

        from gymnasium.spaces import Box
        low = np.array([-6.0, -0.2], dtype=np.float32)
        high = np.array([6.0, 0.2], dtype=np.float32)

        self.action_space = Box(low=low, high=high, dtype=np.float32)
        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,4), dtype=np.float32)

        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32)
        self.observation_space = Dict({
            "obs": Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32),
            "style": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })


        self.prev_lane = self.env.unwrapped.vehicle.lane_index
        self.a_prev = 0

        self.range_1 = [0, 0.9]
        self.range_2 = [0, 0.9]
        self.range_3 = [0, 0.9]
        self.range_4 = [0, 0.9]
        self.range_5 = [0.6, 0.99]
        self.range_6 = [0.75, 1]



        self.z1 = np.random.uniform(*self.range_1)  # aggressiveness
        self.z2 = np.random.uniform(*self.range_2)  # impulsivity
        self.z3 = np.random.uniform(*self.range_3)  # risk
        self.z4 = np.random.uniform(*self.range_4)  # rule obedience
        self.z5 = np.random.uniform(*self.range_5)  # discount factor
        self.z6 = np.random.uniform(*self.range_6)  # another parameter

        self.cash_num = 0



    def step(self, action):

        # frame = self.env.render()
        obs, _, base_done, truncated, info = self.env.step(action)
        vehicle = self.env.unwrapped.vehicle
        custom_done = vehicle.crashed or not vehicle.on_road

        done = base_done or custom_done
        reward = self.redefined_reward(action, vehicle)
        # return obs, reward, done, truncated, info
        return {"obs": obs, "style": np.array([self.z1, self.z2, self.z3, self.z4, self.z5, self.z6],
                                              dtype=np.float32)}, reward, done, truncated, info

    def step_store(self, action):

        frame = self.env.render()
        obs, _, base_done, truncated, info = self.env.step(action)
        vehicle = self.env.unwrapped.vehicle
        custom_done = vehicle.crashed or not vehicle.on_road

        if custom_done:
            self.cash_num = self.cash_num+ 1
        # print(self.cash_num)

        done = base_done or custom_done
        reward = self.redefined_reward(action, vehicle)
        feature = self.features(vehicle, action, T0=2, d_safe =30, v_min=20.0, v_max=30.0)
        return obs, reward, done, truncated,  feature




    def redefined_reward(self, action, vehicle) -> float:

        headway_reward = self.compute_headway_penalty(vehicle)

        velocity_reward = self.compute_rule_conformity_penalty(vehicle)
        crash_reward = -5.0 if (vehicle.crashed or not vehicle.on_road) else 2

        impulsivity_reward = self.compute_impulsivity_penalty(action)
        risk_penalty = self.compute_risk_penalty(vehicle)
        lane_change_penalty = self.compute_lane_change_penalty(vehicle,action)


        reward = velocity_reward + crash_reward + headway_reward + impulsivity_reward + risk_penalty + lane_change_penalty

        # print(f"[Reward Breakdown] "
        #       f"Velocity: {velocity_reward:.2f}, "
        #       f"Crash: {crash_reward:.2f}, "
        #       f"Headway: {headway_reward:.2f}, "
        #       f"Impulsivity: {impulsivity_reward:.2f}, "
        #       f"Risk: {risk_penalty:.2f}, "
        #       f"Lane Change: {lane_change_penalty:.2f}, "
        #       f"Total: {reward:.2f}")

        return reward



    def compute_headway_penalty(self, vehicle, T0=2.0, R1=2):
        T_des = (1 - self.z1) * T0

        neighbors = vehicle.road.neighbour_vehicles(vehicle)

        if not neighbors or neighbors[0] is None:
            # print("no preceding vehicle")
            return 0.0

        front_vehicle = neighbors[0]

        delta_x = front_vehicle.position[0] - vehicle.position[0]
        time_headway = delta_x / max(vehicle.speed, 1e-5)


        if time_headway <= T_des:

            penalty = -R1 * (T_des - time_headway)
        else:
            penalty = 0.0

        return penalty


    def compute_impulsivity_penalty(self, action, R2=2):

        a_t = np.sqrt(action[0] ** 2 + 1*action[1] ** 2)
        penalty = -R2 * (1 - self.z2) * abs(a_t - self.a_prev)
        self.a_prev = a_t

        return penalty



    def compute_risk_penalty(self, vehicle,
                             R3: float = 0.4, d_safe: float = 30.0) -> float:
        neighbors = vehicle.road.neighbour_vehicles(vehicle)
        if not neighbors or neighbors[0] is None:
            return 0.0

        front_vehicle = neighbors[0]
        d_t = front_vehicle.position[0] - vehicle.position[0]


        if d_t < d_safe:
            return -R3 * (1.0 - self.z3) * (d_safe - d_t)
        return 0.0



    def compute_rule_conformity_penalty(self, vehicle, v_min=20, v_max=30.0, R4=5.0):
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        if forward_speed < v_min:
            penalty = -self.z4 * R4 * abs(v_min - forward_speed)
        elif forward_speed > v_max:
            penalty = -self.z4 * R4 * abs(forward_speed - v_max)
        else:
            penalty = 0.0

        return penalty


    def compute_lane_change_penalty(self, vehicle, action):
        current_lane = vehicle.lane_index[2] if vehicle.lane_index else None
        penalty = 0.0

        if current_lane is not None and self.prev_lane is not None:
            if current_lane != self.prev_lane:
                penalty = -0.5

        # penalty = penalty -0.2* abs(vehicle.lane.start[1]-vehicle.position[1])
        penalty = penalty -5* abs(action[1])

        self.prev_lane = current_lane
        return penalty

    def features(self, vehicle, action, T0=2, d_safe=30, v_min=20.0, v_max=30.0):
        T_des =T0

        neighbors = vehicle.road.neighbour_vehicles(vehicle)

        if not neighbors or neighbors[0] is None:
            timeheadway_below = 0.0
            d_t = d_safe
        else:
            front_vehicle = neighbors[0]
            delta_x = front_vehicle.position[0] - vehicle.position[0]
            time_headway = delta_x / max(vehicle.speed, 1e-5)
            d_t = delta_x

            if time_headway <= T_des:
                timeheadway_below = T_des - time_headway
            else:
                timeheadway_below = 0.0

        a_t = np.sqrt(action[0] ** 2 + 0.5 * action[1] ** 2)
        impulsivity = abs(a_t - self.a_prev)

        if d_t < d_safe:
            risk_distance = d_safe - d_t
        else:
            risk_distance = 0.0

        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        if forward_speed < v_min:
            rule_conformity = abs(v_min - forward_speed)
        elif forward_speed > v_max:
            rule_conformity = abs(forward_speed - v_max)
        else:
            rule_conformity = 0.0

        # Option 1: return only behavior features
        return [timeheadway_below, impulsivity, risk_distance, rule_conformity]

        # Option 2: return 6 style features + 4 behavior features (recommended for analysis)
        # return [self.z1, self.z2, self.z3, self.z4, self.z5, self.z6]

    def reset(self, **kwargs):
        self.z1 = np.random.uniform(*self.range_1)  # aggressiveness
        self.z2 = np.random.uniform(*self.range_2)  # impulsivity
        self.z3 = np.random.uniform(*self.range_3)  # risk
        self.z4 = np.random.uniform(*self.range_4)  # rule obedience
        self.z5 = np.random.uniform(*self.range_5)  # discount factor
        self.z6 = np.random.uniform(*self.range_6)  # another parameter

        # self.z1 = 0.0  # aggressiveness
        # self.z2 = 0.0 # impulsivity
        # self.z3 = 0.3  # risk
        # self.z4 = 1.0  # rule obedience
        # self.z5 = 0.9  # discount factor
        # self.z6 =1  # another parameter



        # print(f"z1={self.z1:.4f}, z2={self.z2:.4f}, z3={self.z3:.4f}, "
        #       f"z4={self.z4:.4f}, z5={self.z5:.4f}, z6={self.z6:.4f}")

        plt.close('all')
        obs, info = self.env.reset(**kwargs)
        self.a_prev = 0

        return {"obs": obs, "style": np.array([self.z1, self.z2, self.z3, self.z4, self.z5, self.z6],
                                              dtype=np.float32)}, info






