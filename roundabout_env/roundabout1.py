import numpy as np
import torch
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
import random
import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
from gymnasium.spaces import Dict, Box



class MyRoundaboutEnv(gym.Env):
    def __init__(self, config=None):
        super(MyRoundaboutEnv, self).__init__()


        self.config = {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": 4,
                "see_behind": False,
            },
            "action": {
                "type": "ContinuousAction",
                "target_speeds": np.linspace(0, 32, 9),
            },
            "vehicles_count": 6,
            "incoming_vehicle_destination": None,
            "duration": 25,  # [s]
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 3,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 800,
            "screen_height": 800,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": False
        }

        # roundabout
        self.env = gym.make("roundabout-v0", render_mode="rgb_array")

        self.env.unwrapped.configure(self.config)
        self.env.unwrapped._rewards = lambda action: {}
        self.env.unwrapped._reward = lambda action: 0.0

        #  [-6,6] × [-0.2,0.2]
        low = np.array([-6.0, -0.9], dtype=np.float32)
        high = np.array([6.0, 0.9], dtype=np.float32)
        self.action_space = Box(low=low, high=high, dtype=np.float32)


        self.observation_space = Dict({
            "obs": Box(low=-np.inf, high=np.inf, shape=(4, 6), dtype=np.float32),
            "style": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })


        self.prev_lane = self.env.unwrapped.vehicle.lane_index
        self.a_prev = 0
        self.cash_num = 0

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
        self.passed_mid = False
        self.a_prev1 = 0



    def step(self, action):

        frame = self.env.render()
        obs, _, base_done, truncated, info = self.env.step(action)
        vehicle = self.env.unwrapped.vehicle
        custom_done = vehicle.crashed or not vehicle.on_road
        target_pos = [60, 2.3]
        ego_pos = np.array(vehicle.position)
        dist_to_exit = np.linalg.norm(target_pos - ego_pos)
        reached_exit = dist_to_exit < 2.0  #
        if reached_exit:
            info["reached_exit"] = True
        else:
            info["reached_exit"] = False

        # vehicle.position = [29.5, 5.45491503]

        done = base_done or custom_done or reached_exit

        reward = self.redefined_reward(action, vehicle)
        obs_dict = {
            "obs": obs,
            "style": np.array([self.z1, self.z2, self.z3, self.z4, self.z5, self.z6],
                              dtype=np.float32),
        }

        return obs_dict, reward, done, truncated, info



    def redefined_reward(self, action, vehicle) -> float:
        crash_reward = -5.0 if (vehicle.crashed or not vehicle.on_road) else 1.0
        risk_penalty = self.compute_risk_penalty(vehicle)
        velocity_penalty = self.compute_rule_conformity_penalty(vehicle)

        ego_pos = np.array(vehicle.position)


        target_exit = "exs"
        road = self.env.unwrapped.road
        lane = road.network.get_lane(("ex", target_exit, 0))


        mid_pos = lane.position(0, 0)
        mid_pos = [29.5, 5.45491503]

        dist_to_mid = np.linalg.norm(mid_pos - ego_pos)
        mid_reward = 0.0

        if not self.passed_mid and dist_to_mid < 2:
            mid_reward += 30.0
            self.passed_mid = True
            # print(self.passed_mid)

        target_pos = [60, 2.3]
        dist_to_exit = np.linalg.norm(target_pos - ego_pos)
        approach_reward = .0 * np.exp(-0.05 * dist_to_exit)
        exit_reward = 0.0

        if dist_to_exit < 2:
            exit_reward = 30.0

        headway_reward = self.compute_headway_penalty(vehicle)

        impulsivity_reward = self.compute_impulsivity_penalty(action)


        reward = crash_reward + risk_penalty + velocity_penalty + headway_reward +impulsivity_reward + approach_reward + exit_reward + mid_reward
        return reward

    def compute_headway_penalty(self, vehicle, T0=2.0, R1=2.0):
        T_des = max(0.2, (1 - self.z1) * T0)  # 
        neighbors = vehicle.road.neighbour_vehicles(vehicle)
        if not neighbors or neighbors[0] is None:
            return 0.0
        front_vehicle = neighbors[0]

        delta_x = float(front_vehicle.position[0] - vehicle.position[0])

        if delta_x <= 0.0:
            return 0.0

        speed = float(max(vehicle.speed, 0.5))  # 
        time_headway = delta_x / speed
        time_headway = float(np.clip(time_headway, 0.0, 10.0))  # 

        if time_headway < T_des:
            return -R1 * (T_des - time_headway)
        return 0.0

    def compute_impulsivity_penalty(self, action, R2=0.2):

        a_t = np.sqrt(action[0] ** 2 + 1*action[1] ** 2)
        penalty = -R2 * (1 - self.z2) * abs(a_t - self.a_prev)
        self.a_prev = a_t

        return penalty


    def compute_rule_conformity_penalty(self, vehicle, v_min=2, v_max=10.0, R4=2.0, reward_scale=2):
        forward_speed = vehicle.speed

        if forward_speed < v_min:
            penalty = -self.z4 * R4 * abs(v_min - forward_speed)
        elif forward_speed > v_max:
            penalty = -self.z4 * R4 * abs(forward_speed - v_max)
        else:
            norm_speed = (forward_speed - v_min) / (v_max - v_min)
            penalty = reward_scale * norm_speed
        return penalty

    def compute_risk_penalty(self, vehicle,
                             R3: float = 0.4, d_safe: float = 20.0) -> float:
        neighbors = vehicle.road.neighbour_vehicles(vehicle)
        if not neighbors or neighbors[0] is None:
            return 0.0

        front_vehicle = neighbors[0]
        d_t = front_vehicle.position[0] - vehicle.position[0]

        if d_t < d_safe:
            return -R3 * (1.0 - self.z3) * (d_safe - d_t)
        return 0.0

    def features(self, vehicle, action, T0=2, d_safe=30, v_min=2, v_max=10):
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
        impulsivity = abs(a_t - self.a_prev1)
        self.a_prev1 = a_t

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



    def reset(self, **kwargs):
        self.z1 = np.random.uniform(*self.range_1)  # aggressiveness
        self.z2 = np.random.uniform(*self.range_2)  # impulsivity
        self.z3 = np.random.uniform(*self.range_3)  # risk
        self.z4 = np.random.uniform(*self.range_4)  # rule obedience
        self.z5 = np.random.uniform(*self.range_5)  # discount factor
        self.z6 = np.random.uniform(*self.range_6)  # another parameter

        # self.z1 = 0.1  # aggressiveness
        # self.z2 = 0.1 # impulsivity
        # self.z3 = 0.4  # risk
        # self.z4 = 0.9  # rule obedience
        # self.z5 = 0.99  # discount factor
        # self.z6 =1  # another parameter

        plt.close('all')
        self.passed_mid = False
        self.config["other_vehicles_type"] = "highway_env.vehicle.behavior.IDMVehicle"
        self.env.unwrapped.configure(self.config)
        obs, info = self.env.reset(**kwargs)
        ego = self.env.unwrapped.vehicle
        for v in self.env.unwrapped.road.vehicles:
            if v is not ego:
                if isinstance(v, highway_env.vehicle.behavior.IDMVehicle):
                    v.desired_speed = np.random.uniform(5.0, 8.0)  # 
                    v.speed = np.random.uniform(4.0, 8.0)  # 
        self.a_prev = 0
        self.a_prev1 = 0
        return {
            "obs": obs,
            "style": np.array([self.z1, self.z2, self.z3, self.z4, self.z5, self.z6], dtype=np.float32),
        }, info


