# register_env.py
from ray.tune.registry import register_env
from problem_env1 import MyHighwayEnv

def register_custom_env():
    register_env("CustomEnv", lambda config: MyHighwayEnv(config))




