# register_env.py

from ray.tune.registry import register_env
from gymnasium.envs.registration import register
from problem_env1 import MyHighwayEnv


def register_custom_env():

    register_env("CustomEnv", lambda config: MyHighwayEnv(config))


    def gym_env_creator(**kwargs):

        return MyHighwayEnv(config=kwargs)


    try:
        register(
            id="CustomEnv",
            entry_point=gym_env_creator,
        )
    except Exception as e:
        if "already registered" not in str(e):
            raise e
