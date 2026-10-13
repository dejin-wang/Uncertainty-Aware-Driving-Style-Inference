# register_env.py
from ray.tune.registry import register_env
from roundabout1 import MyRoundaboutEnv

def register_custom_env():
    register_env("CustomEnv", lambda config: MyRoundaboutEnv(config))




