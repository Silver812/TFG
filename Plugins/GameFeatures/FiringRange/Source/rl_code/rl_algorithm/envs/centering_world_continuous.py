import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CenteringWorldEnv(gym.Env):
    metadata = {"render_modes": [None]}

    def __init__(self, size_x=50, size_y=50, render_mode=None):
        self._least_distance: np.inf
        self._size_x = size_x
        self._size_y = size_y
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self._target_location = np.array([0, 0], dtype=np.float32)
        self._timestep_counter = 0

    # Normalizes observation values to be in the range [-1, 1]
    def _normalize_observation(self):
        return self._target_location / np.array([self._size_x // 2, self._size_y // 2], dtype=np.float32)

    # Distance between the target location and the center of the grid
    def _get_distance(self):
        return np.linalg.norm(np.array([0, 0], dtype=np.float32) - self._target_location)

    def _get_info(self):
        return {"distance": self._get_distance()}

    # Transform [-1, 1] action values to [-size_x/2, size_x/2] and [-size_y/2, size_y/2]
    def _action_to_direction(self, action):
        return action * np.array([self._size_x // 2, self._size_y // 2], dtype=np.float32)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._target_location = np.array(
            [self.np_random.uniform(-self._size_x // 2, self._size_x // 2), self.np_random.uniform(-self._size_y // 2, self._size_y // 2)],
            dtype=np.float32,
        )

        self._timestep_counter = 0
        self._least_distance = self._get_distance()
        observation = self._normalize_observation()
        info = self._get_info()

        return observation, info

    def _get_reward(self, terminated=False):

        distance = self._get_distance()
        reward = 0.0

        # If the distance to the target is less than 1, the agent has reached the target
        if terminated:
            # print("Agent has reached the target")
            reward += 1.0

        # If the distance to the target is less than the least distance, the agent is getting closer to the target
        elif distance < self._least_distance:
            self._least_distance = distance
            # print("Agent is getting closer to the target")
            reward += 0.01

        # If the distance to the target is greater than the least distance, the agent is getting farther from the target
        else:
            reward -= 0.01

        return reward

    def step(self, action):
        # Transform [-1, 1] action values to [-size_x/2, size_x/2] and [-size_y/2, size_y/2]
        direction = self._action_to_direction(action)

        # Make sure we don't leave the grid
        self._target_location = np.clip(
            self._target_location + direction,
            -np.array([self._size_x // 2, self._size_y // 2], dtype=np.float32),
            np.array([self._size_x // 2, self._size_y // 2], dtype=np.float32),
        )

        # An episode is done if the the target location without decimals is equal to [0, 0]
        terminated = np.array_equal(self._target_location.astype(int), np.array([0, 0], dtype=int))

        reward = self._get_reward(terminated)
        observation = self._normalize_observation()
        info = self._get_info()

        truncated = False

        if self._timestep_counter >= 50000:
            truncated = True
            print("Episode truncated")
        else:
            self._timestep_counter += 1

        return observation, reward, terminated, truncated, info

    def close(self):
        print("Closing the environment")


if __name__ == "__main__":
    
    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    # from stable_baselines3 import PPO
    from sbx import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from rl_zoo3 import linear_schedule
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback
    import matplotlib.pyplot as plt
    from stable_baselines3.common import results_plotter
    from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
    from torch import nn as nn
    from gymnasium.utils.env_checker import check_env
    from rl_algorithm.envs.continuous_world import ContinuousWorldEnv
    from rl_algorithm.envs.centering_world import CenteringWorldEnv
    import sys

    # env = gym.make("ContinuousWorld-v0", 50)
    # check_env(env.unwrapped, skip_render_check=True)

    # Set the seeds for reproducibility
    seed_value = 42
    world_size_x = 100
    world_size_y = 100

    # Set the seed for numpy
    np.random.seed(seed_value)

    # Function that creates a new environment
    # def make_env():
    #     def _init():
    #         return ContinuousWorldEnv(world_size)

    #     return _init

    def make_env(world_size_x: int, world_size_y: int, rank: int = 0, seed: int = 0):
        """
        Utility function for multiprocessed env.

        :param world_size: the size of the environment
        :param num_env: the number of environments you wish to have in subprocesses
        :param seed: the inital seed for RNG
        :param rank: index of the subprocess
        """

        def _init():
            env = gym.make(CenteringWorldEnv(world_size_x, world_size_y), render_mode=None)
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init

    # num_envs = 16

    # # List of environment creation functions
    # vec_env = [make_env() for _ in range(num_envs)]

    num_cpu = 16  # Number of processes to use
    n_training_envs = 4
    timesteps = 1e6
    log_dir = "./tmp/"
    eval_log_dir = "./eval_logs/"

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    env = make_vec_env(
        "CenteringWorld-v0",
        n_envs=num_cpu,
        seed=seed_value,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"size_x": world_size_x, "size_y": world_size_y, "render_mode": None},
    )

    # Vectorized environment
    # env = DummyVecEnv(vec_env)

    os.makedirs(log_dir, exist_ok=True)

    # env = VecMonitor(env, log_dir) # Not necessary, because make_vec_env already does this

    # Automatically normalize the input features and reward
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1.0)

    # # Separate evaluation env, with different parameters passed via env_kwargs
    # # Eval environments can be vectorized to speed up evaluation.
    # eval_env = make_vec_env("CenteringWorld-v0", n_envs=n_training_envs, seed=seed_value, vec_env_cls=DummyVecEnv, env_kwargs={"size": 50, "render_mode": None})
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=1.0)

    # # Create callback that evaluates agent for 5 episodes every 500 training environment steps.
    # # When using multiple training environments, agent will be evaluated every
    # # eval_freq calls to train_env.step(), thus it will be evaluated every
    # # (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=eval_log_dir,
    #     log_path=eval_log_dir,
    #     eval_freq=max(5000 // n_training_envs, 1),
    #     n_eval_episodes=5,
    #     deterministic=True,
    #     render=False,
    # )

    # Train the agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=512,
        n_steps=1024,
        gamma=0.9999,
        learning_rate=linear_schedule(0.0005490571741643338),
        ent_coef=0.0008758449893201579,
        clip_range=linear_schedule(0.3),
        n_epochs=20,
        gae_lambda=0.92,
        max_grad_norm=0.7,
        vf_coef=0.23534158625221846,
        seed=seed_value,
        policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
    )

    print("\nTraining in process...\n")
    # model.learn(total_timesteps=int(timesteps), progress_bar=True, callback=eval_callback)
    model.learn(total_timesteps=int(timesteps), progress_bar=True)

    # model.save(log_dir + "ppo_centering_world")
    # stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    # env.save(stats_path)

    # plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO Centering World")
    # plt.show()

    print("\nEvaluation in process...\n")

    # Load the saved statistics
    # vec_env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
    # vec_env = VecNormalize.load(stats_path, vec_env)
    # Load the agent
    # model = PPO.load(log_dir + "ppo_halfcheetah", env=vec_env)

    # # Do not update them at test time
    # env.training = False
    # # reward normalization is not needed at test time
    # env.norm_reward = False

    # Make a single environment
    env = gym.make("CenteringWorld-v0", size_x=world_size_x, size_y=world_size_y)
    env.reset(seed=seed_value)

    # More detailed evaluation:
    total_reward = 0.0
    episode_rewards = []
    terminated_counter = 0
    total_eval_episodes = 10

    observation, info = env.reset()

    while terminated_counter < total_eval_episodes:
        action, _ = model.predict(observation, deterministic=True)
        # print("Action: ", action)
        observation, reward, terminated, truncated, info = env.step(action)
        # print("obs=", observation, "reward=", reward, "terminated=", terminated, "truncated=", truncated, "info=", info)
        total_reward = reward

        if terminated or truncated:
            print("That was episode: " + str(terminated_counter + 1))
            observation, info = env.reset()
            episode_rewards.append(total_reward)
            total_reward = 0.0
            terminated_counter += 1

    print("\nEpisode rewards:", episode_rewards, "\n")
