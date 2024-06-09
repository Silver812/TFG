import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import IntEnum


class Actions(IntEnum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    SHOOT = 4


class CenteringWorldEnv(gym.Env):
    metadata = {"render_modes": [None]}

    def __init__(self, size_x=1000, size_y=1000, max_timesteps=50000, distance_percentage=0.03, render_mode=None):
        # Size of the grid
        self._size_x = size_x
        self._size_y = size_y

        # The least the target has been from the center of the grid
        self._least_distance = np.inf

        # Location of the current target
        self._target_location = np.array([0, 0], dtype=np.float32)

        # Maximum number of timesteps before truncating the episode
        self._max_timesteps = max_timesteps
        self._timestep_counter = 0

        # Percentage of the grid that is covered by the target on each step.
        self._distance_percentage = np.float32(distance_percentage)
        self._distance_x = np.float32(self._size_x * self._distance_percentage)
        self._distance_y = np.float32(self._size_y * self._distance_percentage)

        # Minimum distance for the target to be considered reached based on euclidean distance -5% of itself to try to manage aprox. errors
        self._min_distance = np.linalg.norm(np.array([self._distance_x, self._distance_y], dtype=np.float32)) * 0.95

        # Definitions of the action and observation spaces
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(Actions))

        self._action_to_direction = {
            Actions.RIGHT: np.array([self._distance_x, 0], dtype=np.float32),
            Actions.UP: np.array([0, self._distance_y], dtype=np.float32),
            Actions.LEFT: np.array([-self._distance_x, 0], dtype=np.float32),
            Actions.DOWN: np.array([0, -self._distance_y], dtype=np.float32),
        }

        self._center_reached_previously = False
        self._in_center = False
        self._center_unreached = False

    # Normalizes observation values to be in the range [-1, 1]
    def _normalize_observation(self):
        return self._target_location / np.array([self._size_x / 2, self._size_y / 2], dtype=np.float32)

    # Distance between the target location and the center of the grid
    def _get_distance(self):
        return np.linalg.norm(self._target_location)

    def _has_reached_center(self):
        current_distance = self._get_distance()
        return current_distance < (self._min_distance)

    def _get_info(self):
        return {"distance": self._get_distance()}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # ------------------------------------------------------------------------------------------------------------------------------

        # EXTRA TASK ONLY FOR THE TRAINING: randomize size of the grid and distance_percentage to make the agent more robust

        self._size_x = self.np_random.integers(200, 8000, dtype=int)
        self._size_y = self.np_random.integers(200, 8000, dtype=int)
        self._distance_percentage = np.float32(self.np_random.uniform(0.01, 0.05))

        # Percentage of the grid that is covered by the target on each step.
        self._distance_x = np.float32(self._size_x * self._distance_percentage)
        self._distance_y = np.float32(self._size_y * self._distance_percentage)

        # Minimum distance for the target to be considered reached based on euclidean distance -5% of itself to try to manage aprox. errors
        self._min_distance = np.linalg.norm(np.array([self._distance_x, self._distance_y], dtype=np.float32)) * 0.95

        self._action_to_direction = {
            Actions.RIGHT: np.array([self._distance_x, 0], dtype=np.float32),
            Actions.UP: np.array([0, self._distance_y], dtype=np.float32),
            Actions.LEFT: np.array([-self._distance_x, 0], dtype=np.float32),
            Actions.DOWN: np.array([0, -self._distance_y], dtype=np.float32),
        }

        # ------------------------------------------------------------------------------------------------------------------------------

        # Choose the target location uniformly at random
        self._target_location = np.array(
            [self.np_random.uniform(-self._size_x / 2, self._size_x / 2), self.np_random.uniform(-self._size_y / 2, self._size_y / 2)],
            dtype=np.float32,
        )

        # print(f"Target location: {self._target_location}")

        self._center_reached_previously = False
        self._in_center = self._has_reached_center()
        self._center_unreached = False
        self._timestep_counter = 0
        self._least_distance = self._get_distance()
        observation = self._normalize_observation()
        info = self._get_info()

        return observation, info

    def _get_reward(self, action):

        distance = self._get_distance()
        reward = 0.0
        terminated = False

        if action == Actions.SHOOT:
            if self._in_center:
                reward = 1.0
                terminated = True
                # print("Target reached")
            else:
                reward = 0.0

        elif self._center_unreached:
            # In the last step the agent was in the center, but now it is not
            reward = -0.5
            self._center_unreached = False
            # print("Agent left the center")

        elif self._in_center and not self._center_reached_previously:
            reward = 0.5
            self._center_reached_previously = True
            # print("Agent reached the center")

        elif distance < self._least_distance:
            # The agent is getting closer to the target
            self._least_distance = distance
            reward = 0.1
            # print("Agent is getting closer to the target")

        else:
            # The agent is getting away from the target
            reward = -0.1

        return reward, terminated

    def step(self, action):

        if action != Actions.SHOOT:
            # Transform discrete actions into vectors for the grid
            direction = self._action_to_direction[action]

            # Make sure we don't leave the grid
            self._target_location = np.clip(
                self._target_location + direction,
                -np.array([self._size_x / 2, self._size_y / 2], dtype=np.float32),
                np.array([self._size_x / 2, self._size_y / 2], dtype=np.float32),
            )

            was_in_center = self._in_center
            self._in_center = self._has_reached_center()

            # If the agent was in the center in the previous step but now it is not
            if was_in_center and not self._in_center:
                self._center_unreached = True

        reward, terminated = self._get_reward(action)
        observation = self._normalize_observation()
        info = self._get_info()

        truncated = False

        if self._timestep_counter >= self._max_timesteps:
            truncated = True
            print("Episode truncated")
        else:
            self._timestep_counter += 1

        return observation, reward, terminated, truncated, info

    def close(self):
        print("Closing the environment")


"""
# ?Optimize
python -m rl_zoo3.train --algo ppo --env CenteringWorld-v0 -n 50000 -optimize --n-trials 15 --n-jobs 8 --sampler tpe --pruner median
python -m rl_zoo3.train --algo a2c --env CenteringWorld-v0 -n 50000 -optimize --n-trials 250 --n-jobs 8 --sampler tpe --pruner median
python -m rl_zoo3.train --algo ppo --env CenteringWorld-v1 -n 50000 -optimize --n-trials 15 --n-jobs 8 --sampler tpe --pruner median

# ?Train
python -m rl_zoo3.train --algo ppo --env CenteringWorld-v0 --n-timesteps 1000000 --progress --eval-freq 250000  --seed 42
python -m rl_zoo3.train --algo a2c --env CenteringWorld-v0 --n-timesteps 25000000 --progress --eval-freq 250000 --seed 42
python -m rl_zoo3.train --algo ppo --env CenteringWorld-v1 --n-timesteps 1000000 --progress --eval-freq 250000 --seed 42

python train_sbx.py --algo ppo --env CenteringWorld-v0 --n-timesteps 1000000 --progress --eval-freq 250000

# ?Continue training
python -m rl_zoo3.train --algo ppo --env CenteringWorld-v0 --n-timesteps 1000000 --progress --eval-freq 250000 -i logs\ppo\CenteringWorld-v0_1\CenteringWorld-v0.zip

# ?Evaluate
python -m rl_zoo3.enjoy --algo ppo --env CenteringWorld-v0 --no-render --n-timesteps 100000 --folder logs --exp-id 1
python -m rl_zoo3.enjoy --algo a2c --env CenteringWorld-v0 --no-render --n-timesteps 100000 --folder logs --exp-id 1
python -m rl_zoo3.enjoy --algo ppo --env CenteringWorld-v1 --no-render --n-timesteps 100000 --folder logs --exp-id 1

python enjoy_sbx.py --algo ppo --env CenteringWorld-v0 --no-render --n-timesteps 1000 --folder logs

# ?Training plots
python -m rl_zoo3.plots.plot_train --algo ppo --env CenteringWorld-v0 -f logs -y reward -x steps -w 2000 --figsize 12 7
python -m rl_zoo3.plots.plot_train --algo a2c --env CenteringWorld-v0 -f logs -y reward -x steps --figsize 12 7 --max 3000000
python -m rl_zoo3.plots.plot_train --algo ppo --env CenteringWorld-v1 -f logs -y reward -x steps -w 2000 --figsize 12 7

python -m rl_zoo3.plots.plot_train --algo ppo --env CenteringWorld-v0 -f logs -y length -x steps -w 2000 --figsize 12 7

# ?Training plot PPO + PPO extended
python -m rl_zoo3.plots.plot_train --algo ppo_exp --env CenteringWorld-v0 -f logs -y norm -x steps -w 2000 --figsize 12 7

# ?Other training plots: compare different runs syncing the x-axis
python -m rl_zoo3.plots.all_plots --algo ppo --env CenteringWorld-v0 -f logs -k results
python -m rl_zoo3.plots.all_plots --algo ppo --env CenteringWorld-v0 -f logs -k timesteps
python -m rl_zoo3.plots.all_plots --algo ppo --env CenteringWorld-v0 -f logs -k ep_lengths

# ?Create package (from the root folder)
pip install -e .
"""
