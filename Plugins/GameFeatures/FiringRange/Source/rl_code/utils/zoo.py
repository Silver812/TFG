from rl_algorithm.envs.continuous_world import ContinuousWorldEnv
from rl_algorithm.envs.centering_world import CenteringWorldEnv
from rl_algorithm.envs.centering_world_ext import CenteringWorldExtEnv

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env
import stable_baselines3 as sb3


def test_world1():
    # sb3.get_system_info()
    env = gym.make("ContinuousWorld-v0")
    # check_env(env.unwrapped, skip_render_check=True)
    sb3_check_env(env)
    # env.reset()

def test_world2():
    # sb3.get_system_info()
    env = gym.make("CenteringWorld-v0")
    check_env(env.unwrapped, skip_render_check=True)
    # sb3_check_env(env)
    # env.reset()

def test_world3():
    # sb3.get_system_info()
    env = gym.make("CenteringWorld-v1")
    check_env(env.unwrapped, skip_render_check=True)
    # sb3_check_env(env)
    # env.reset()

if __name__ == "__main__":
    # test_world1()
    # test_world2()
    test_world3()