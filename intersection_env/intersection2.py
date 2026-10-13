import numpy as np
import torch
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
import random
import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
from gymnasium.spaces import Dict, Box
import  time
from shapely.geometry import LineString, Polygon
from shapely.geometry import Point

class MyIntersectionEnv(gym.Env):
    def __init__(self, config=None):
        super(MyIntersectionEnv, self).__init__()


        self.config = {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": 3,
                "see_behind": False,
            },
            "action": {
                "type": "ContinuousAction",
                "target_speeds": np.linspace(0, 32, 9),
            },
            "vehicles_count": 3,
            "spawn_probability": 0.0,
            "initial_vehicle_count": 3,
            "incoming_vehicle_destination": None,
            "duration": 20,  # [s]
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 6,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 800,
            "screen_height": 800,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": False
        }


        self.env = gym.make("intersection-v0", render_mode="rgb_array")

        self.env.unwrapped.configure(self.config)
        self.env.unwrapped._rewards = lambda action: {}
        self.env.unwrapped._reward = lambda action: 0.0


        low = np.array([-6.0, -0.6], dtype=np.float32)
        high = np.array([6.0, 0.6], dtype=np.float32)
        self.action_space = Box(low=low, high=high, dtype=np.float32)


        self.observation_space = Dict({
            "obs": Box(low=-np.inf, high=np.inf, shape=(3, 6), dtype=np.float32),
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

        net = self.env.unwrapped.road.network
        origin = "o0"
        destination = "o1"
        route_nodes = net.shortest_path(origin, destination)
        self.points = []
        for i in range(len(route_nodes) - 1):
            lane = net.get_lane((route_nodes[i], route_nodes[i + 1], 0))
            s_values = np.linspace(0, lane.length, 20)
            for s in s_values:
                x, y = lane.position(s, 0)
                self.points.append((x, y))
        self.points = np.array(self.points)
        center_line = LineString(self.points)
        self.road_polygon = center_line.buffer(3, cap_style=2, join_style=2)
        self.passed_exit_zone = False


    def is_off_road(self, vehicle_pos):
        point = Point(vehicle_pos[0], vehicle_pos[1])
        return not self.road_polygon.buffer(1e-6).contains(point)

    def step(self, action):
        frame = self.env.render()
        obs, _, base_done, truncated, info = self.env.step(action)
        vehicle = self.env.unwrapped.vehicle
        custom_done = vehicle.crashed or not vehicle.on_road
        target_pos = [-110, -2]
        ego_pos = np.array(vehicle.position)
        dist_to_exit = np.linalg.norm(target_pos - ego_pos)
        reached_exit = dist_to_exit < 2.0  #

        off_reward_done = self.is_off_road(ego_pos)

        if reached_exit:
            info["reached_exit"] = True
        else:
            info["reached_exit"] = False

        done = base_done or custom_done or reached_exit or off_reward_done



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
        off_reward = -5.0 if self.is_off_road(ego_pos) else 1.0

        time_headway_reward = self.compute_headway_penalty(vehicle)

        impulsivity_reward = self.compute_impulsivity_penalty(action)
        reward = crash_reward + risk_penalty + velocity_penalty +impulsivity_reward + time_headway_reward + off_reward
        return reward

    def compute_headway_penalty(self, vehicle, T0=2.0, R1=2.0, d_safe=15.0, alpha=0.3):
        T_des = max(0.2, (1 - self.z1) * T0)
        ego_pos = np.array(vehicle.position)
        ego_heading = vehicle.heading
        total_penalty = 0.0

        for other in vehicle.road.vehicles:
            if other is vehicle:
                continue

            rel_pos = np.array(other.position) - ego_pos
            dist = np.linalg.norm(rel_pos)
            if dist > d_safe:
                continue


            rel_angle = np.arctan2(rel_pos[1], rel_pos[0]) - ego_heading

            front_factor = np.cos(rel_angle)
            if front_factor > 0:
                front_factor = max(alpha, front_factor)
            else:
                front_factor = 0.0

            speed = float(max(vehicle.speed, 0.5))
            time_headway = dist / speed

            if time_headway < T_des:
                total_penalty += -R1 * (T_des - time_headway) * front_factor

        return total_penalty

    def compute_impulsivity_penalty(self, action, R2=0.2):

        a_t = np.sqrt(action[0] ** 2 + 1*action[1] ** 2)
        penalty = -R2 * (1 - self.z2) * abs(a_t - self.a_prev)
        self.a_prev = a_t

        return penalty


    def compute_rule_conformity_penalty(self, vehicle, v_min=1, v_max=15.0, R4=2, reward_scale=2):
        forward_speed = vehicle.speed

        if forward_speed < v_min:
            penalty = -self.z4 * R4 * abs(v_min - forward_speed)
        elif forward_speed > v_max:
            penalty = -self.z4 * R4 * abs(forward_speed - v_max)
        else:
            norm_speed = (forward_speed - v_min) / (v_max - v_min)
            penalty = reward_scale * norm_speed
        return penalty

    def compute_risk_penalty(self, vehicle, R3=0.2, d_safe=20.0, alpha=0.5):
        ego_pos = np.array(vehicle.position)
        ego_heading = vehicle.heading
        total_penalty = 0.0

        for other in vehicle.road.vehicles:
            if other is vehicle:
                continue

            dist = np.linalg.norm(np.array(other.position) - ego_pos)
            if dist < d_safe:
                dtheta = abs(ego_heading - other.heading)
                rel_dir = abs(np.sin(dtheta)) + alpha * (np.cos(dtheta) ** 2)
                rel_dir = max(rel_dir, 0.1)
                total_penalty += -R3 * (1 - self.z3) * (d_safe - dist) * rel_dir

        return total_penalty

    def features(self, vehicle, action, T0=2.0, R1=2.0, d_safe=20.0, v_min=1.0, v_max=15.0):
        ego_pos = np.array(vehicle.position)
        ego_heading = vehicle.heading

        # -------- (1) time headway  headway risk --------
        # T_des = max(0.2, (1 - self.z1) * T0)
        T_des = T0
        timeheadway_below = 0.0

        for other in vehicle.road.vehicles:
            if other is vehicle:
                continue

            rel_pos = np.array(other.position) - ego_pos
            dist = np.linalg.norm(rel_pos)
            if dist > d_safe:
                continue

            rel_angle = np.arctan2(rel_pos[1], rel_pos[0]) - ego_heading
            front_factor = np.cos(rel_angle)
            if front_factor > 0:
                front_factor = max(0.3, front_factor)  # 
            else:
                front_factor = 0.0

            speed = float(max(vehicle.speed, 0.5))
            time_headway = dist / speed

            if time_headway < T_des:

                timeheadway_below += (T_des - time_headway) * front_factor

        # -------- (2) impulsivity --------
        a_t = np.sqrt(action[0] ** 2 + 1 * action[1] ** 2)
        impulsivity = abs(a_t - self.a_prev1)
        self.a_prev1 = a_t

        # -------- (3) risk distance（ compute_risk_penalty）--------
        risk_distance = 0.0
        for other in vehicle.road.vehicles:
            if other is vehicle:
                continue

            dist = np.linalg.norm(np.array(other.position) - ego_pos)
            if dist < d_safe:
                dtheta = abs(ego_heading - other.heading)
                rel_dir = abs(np.sin(dtheta)) + 0.5 * (np.cos(dtheta) ** 2)
                rel_dir = max(rel_dir, 0.1)
                risk_distance += (d_safe - dist) * rel_dir

        # -------- (4) rule conformity --------
        forward_speed = vehicle.speed
        if forward_speed < v_min:
            rule_conformity = abs(v_min - forward_speed)
        elif forward_speed > v_max:
            rule_conformity = abs(forward_speed - v_max)
        else:
            rule_conformity = 0.0

        return [timeheadway_below, impulsivity, risk_distance, rule_conformity]

        # Option 2: return 6 style features + 4 behavior features (recommended for analysis)
        # return [self.z1, self.z2, self.z3, self.z4, self.z5, self.z6]





    def reset(self, **kwargs):
        self.env.unwrapped.configure(self.config)
        self.env.reset()

        self.z1 = np.random.uniform(*self.range_1)  # aggressiveness
        self.z2 = np.random.uniform(*self.range_2)  # impulsivity
        self.z3 = np.random.uniform(*self.range_3)  # risk
        self.z4 = np.random.uniform(*self.range_4)  # rule obedience
        self.z5 = np.random.uniform(*self.range_5)  # discount factor
        self.z6 = np.random.uniform(*self.range_6)  # another parameter


        self.passed_exit_zone = False



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


