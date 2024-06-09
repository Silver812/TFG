from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class ContinuousWorldEnv(gym.Env):
    metadata = {"render_modes": [None]}

    def __init__(self, size=50, render_mode=None):
        self.least_distance: np.inf
        self.last_action: 0
        self.size = size  # The size of the square grid

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64),
                "target": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64),
            }
        )
        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)
        self.timestep_counter = 0

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0], dtype=int),
            Actions.UP.value: np.array([0, 1], dtype=int),
            Actions.LEFT.value: np.array([-1, 0], dtype=int),
            Actions.DOWN.value: np.array([0, -1], dtype=int),
        }

    # Normalizes observation values to be in the range [-1, 1]
    def _normalize_observation(self, observation):
        normalized_observation = {}

        for key, value in observation.items():
            normalized_observation[key] = np.array([((v + self.size / 2) / self.size) * 2 - 1 for v in value], dtype=np.float64)

        return normalized_observation

    def _get_distance(self):
        return np.linalg.norm(self._agent_location - self._target_location, ord=1)

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": self._get_distance()}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random from -size/2 to size/2
        self._agent_location = self.np_random.integers(-self.size // 2, self.size // 2, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(-self.size // 2, self.size // 2, size=2, dtype=int)

        self.last_action = 0
        self.timestep_counter = 0
        self.least_distance = self._get_distance()
        # observation = self._get_obs()
        observation = self._normalize_observation(self._get_obs())
        info = self._get_info()

        return observation, info

    def _get_reward(self, action):

        distance = self._get_distance()
        reward = 0.0

        # If the distance to the target is less than 1, the agent has reached the target
        if distance < 1:
            print("Agent has reached the target")
            reward += 1.0

        # If the distance to the target is less than the least distance, the agent is getting closer to the target
        elif distance < self.least_distance:
            self.least_distance = distance
            # print("Agent is getting closer to the target")
            reward += 0.01

        # If the distance to the target is greater than the least distance, the agent is getting farther from the target
        else:
            reward -= 0.01

        if action != self.last_action:
            reward += 0.01
        else:
            reward -= 0.01

        return reward

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, -self.size // 2, (self.size - 1) // 2)
        # An episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = self._get_reward(action)
        self.last_action = action
        # observation = self._get_obs()
        observation = self._normalize_observation(self._get_obs())
        info = self._get_info()

        truncated = False

        if self.timestep_counter >= 50000:
            truncated = True
            print("Episode truncated")
        else:
            self.timestep_counter += 1

        return observation, reward, terminated, truncated, info

    def close(self):
        print("Closing the environment")


if __name__ == "__main__":
    import os

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    from stable_baselines3 import PPO
    from rl_zoo3 import linear_schedule
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    import matplotlib.pyplot as plt
    from stable_baselines3.common import results_plotter
    from stable_baselines3.common.results_plotter import plot_results
    from torch import nn as nn
    from rl_algorithm.envs.continuous_world import ContinuousWorldEnv

    # Set the seeds for reproducibility
    seed_value = 42
    world_size = 50

    # Set the seed for numpy
    np.random.seed(seed_value)

    def make_env(world_size: int, rank: int = 0, seed: int = 0):
        """
        Utility function for multiprocessed env.

        :param world_size: the size of the environment
        :param num_env: the number of environments you wish to have in subprocesses
        :param seed: the inital seed for RNG
        :param rank: index of the subprocess
        """

        def _init():
            env = gym.make(ContinuousWorldEnv(world_size), render_mode=None)
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init

    num_cpu = 16  # Number of processes to use
    n_training_envs = 4
    timesteps = 3e5
    log_dir = "./tmp/"
    eval_log_dir = "./eval_logs/"

    env = make_vec_env("ContinuousWorld-v0", n_envs=num_cpu, seed=seed_value, vec_env_cls=DummyVecEnv, env_kwargs={"size": 50, "render_mode": None})
    os.makedirs(log_dir, exist_ok=True)

    # Automatically normalize the input features and reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1.0)

    # Separate evaluation env, with different parameters passed via env_kwargs
    eval_env = make_vec_env("ContinuousWorld-v0", n_envs=n_training_envs, seed=seed_value, vec_env_cls=DummyVecEnv, env_kwargs={"size": 50, "render_mode": None})
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=1.0)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir,
        eval_freq=max(5000 // n_training_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Train the agent
    model = PPO(
        "MultiInputPolicy",
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
        policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64]), "activation_fn": nn.ReLU},
    )

    print("\nTraining in process...\n")
    
    model.learn(total_timesteps=int(timesteps), progress_bar=True)

    model.save(log_dir + "ppo_continuous_world")
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env.save(stats_path)

    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO Continuous World")
    plt.show()

    print("\nEvaluation in process...\n")

    # Do not update them at test time
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False

    # Disable the vectorized environment
    env = ContinuousWorldEnv(world_size)

    # More detailed evaluation:
    total_reward = 0.0
    episode_rewards = []
    terminated_counter = 0
    total_eval_episodes = 10

    observation, info = env.reset()

    while terminated_counter < total_eval_episodes:
        action, _ = model.predict(observation, deterministic=True)
        action = int(action)
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