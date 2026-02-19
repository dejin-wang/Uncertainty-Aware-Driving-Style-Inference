
from ray.tune.registry import register_env
from intersection2 import MyIntersectionEnv

def register_custom_env():
    register_env("CustomEnv", lambda config: MyIntersectionEnv(config))




