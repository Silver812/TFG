###################
# Library imports #
###################

from flask import Flask  # Communication between UE5 and Python
import socketio  # Socket.IO client and server library for real-time web applications
import json  # JSON encoder and decoder for Python
import numpy as np  # Library for working with arrays and matrices
import sys  # System-specific parameters and functions
import os  # Miscellaneous operating system interfaces
import gymnasium as gym  # Gymnasium is a toolkit for developing and comparing reinforcement learning algorithms
from gymnasium import spaces  # Spaces module defines the observation and action spaces used by the environment
from enum import IntEnum
import threading
import gevent
from stable_baselines3 import PPO, A2C

#############
# Variables #
#############

TIMEOUT_SECONDS = 3  # Timeout in seconds for the check_obs function

ue_observations = None
ue_reward = None
ue_terminated = None
ue_truncated = None
ue_update_time = None

# Socket.IO server with Flask web framework
sio = socketio.Server(logger=False, async_mode="gevent")
app = Flask(__name__)
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

# Global variable to signal the server to stop
stop_server = False

# Global threading event
data_received_event = threading.Event()

round_counter = 0

###############
# Environment #
###############


class Actions(IntEnum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    SHOOT = 4
    RIGHT_UP = 5
    RIGHT_DOWN = 6
    LEFT_UP = 7
    LEFT_DOWN = 8


class CenteringWorldEnv(gym.Env):
    metadata = {"render_modes": [None]}

    def __init__(self, size_x=50, size_y=50, max_timesteps=50000, distance_percentage=0.03, update_time=0.001, render_mode=None):
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

        # Minimum distance for the target to be considered reached based on distance_percentage and the size of the grid
        # Calculate the minimum distance for the target to be considered reached
        # self._min_distance = np.hypot(self._distance_x, self._distance_y)
        self._min_distance = np.linalg.norm(np.array([self._distance_x, self._distance_y], dtype=np.float32))

        # Definitions of the action and observation spaces
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(Actions))

        self._center_reached_previously = False
        self._in_center = False
        self._center_unreached = False
        self._update_time = update_time
        self._shoot_counter = 0
        self._current_distance = np.inf
        self._error_counter = 0

    def _normalize_observation(self):
        # Range [-1 ,1]
        return self._target_location / np.array([self._size_x / 2, self._size_y / 2], dtype=np.float32)

    def _get_fixed_obs(self, observation):
        # Convert the target location from [0, size_x] and [0, size_y] to [-size_x/2, size_x/2] and [-size_y/2, size_y/2]
        return observation - np.array([self._size_x / 2, self._size_y / 2], dtype=np.float32)

    # Distance between the target location and the center of the grid
    def _get_distance(self):
        return np.linalg.norm(self._target_location)

    def _has_reached_center(self):
        self._current_distance = self._get_distance()
        return self._current_distance < (self._min_distance)

    def _get_info(self):
        return {"distance": self._get_distance()}

    def _check_obs(self):
        global data_received_event
        global stop_server

        for_range = int(TIMEOUT_SECONDS * (1 / self._update_time))

        # Wait for the event to be set, with a timeout
        for _ in range(for_range):  # Multiply by 10 for a resolution of 0.1 seconds
            if stop_server:
                break
            if data_received_event.is_set():
                # If the event was set, clear it for the next use
                data_received_event.clear()
                return True
            gevent.sleep(float(self._update_time))  # Yield control to other greenlets

        return False

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        global ue_observations

        # ---------------------------------------------------------------------- #
        # Initialize the agent with a random action
        sio.emit("receive_reset_data", "Next target")
        # ---------------------------------------------------------------------- #

        if self._check_obs():
            self._current_distance = np.inf
            self._target_location = self._get_fixed_obs(ue_observations)
            self._center_reached_previously = False
            self._in_center = self._has_reached_center()
            self._center_unreached = False
            self._timestep_counter = 0
            self._least_distance = self._get_distance()
            observation = self._normalize_observation()
            info = self._get_info()

            return observation, info
        else:
            print("Timeout reached")
            return np.array([0, 0], dtype=np.float32), {}

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

        # ---------------------------------------------------------------------- #
        if action == Actions.SHOOT:
            self._shoot_counter += 1

            # Change the position to try to break the agent out of the infinite shooting loop
            if self._shoot_counter > 2:
                action = list(Actions)[self.np_random.integers(0, 3, dtype=int)]
                self._shoot_counter = 0
                self._error_counter += 1

            # Send the action to UE
            sio.emit("send_action", action)
            # Wait for the shot to be registered
            gevent.sleep(float(0.4))
        else:
            # If action is SHOOT, wait for the shot to be fired
            sio.emit("send_action", action)
            self._shoot_counter = 0
        # ---------------------------------------------------------------------- #

        if self._check_obs():
            self._target_location = self._get_fixed_obs(ue_observations)

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
        else:
            global round_counter
            print("Enf of round: ", round_counter)
            print("Number of errors this round: ", self._error_counter)
            return np.array([0, 0], dtype=np.float32), 0.0, True, False, {}

    def close(self):
        print("Closing the environment")


###################
# SocketIO Events #
###################


@sio.event
def connect(*_):
    print("Connected To Unreal Engine")


@sio.event
def disconnect(sid):
    try:
        global stop_server
        print("\nDisconnected From Unreal Engine, Exiting RLEngine")
        stop_server = True
        sio.disconnect(sid)

    except TypeError as e:
        print(f"An error occurred: {e}")


# Observation received from UE
@sio.on("send_obs")
def send_observation(_, observation_data):
    global ue_observations
    global ue_reward
    global ue_terminated
    global ue_truncated
    global data_received_event

    json_input = json.loads(observation_data)
    ue_observations = np.array(json_input["observation"].split(","), dtype=np.float32)
    ue_reward = json_input["reward"]
    ue_terminated = json_input["bTerminated"]
    ue_truncated = json_input["bTruncated"]

    # Set the event to indicate that data has been received
    data_received_event.set()


@sio.on("send_message")
def send_message(_, message):
    json_input = json.loads(message)
    ue_message = json_input["message"]
    print(ue_message)


# UE wants to connect to the server
@sio.on("launch_rl_engine")
def receive(sid, initialization_data):
    global ue_terminated
    global ue_truncated
    global stop_server
    global round_counter

    round_counter += 1
    stop_server = False
    json_input = json.loads(initialization_data)
    observation = json_input["observation"]
    reward = json_input["reward"]
    terminated = json_input["bTerminated"]
    truncated = json_input["bTruncated"]
    info = json_input["info"]
    size_x = json_input["sizeX"]
    size_y = json_input["sizeY"]
    max_timesteps = json_input["maxTimesteps"]
    distance_percentage = json_input["distancePercentage"]
    eval_episodes = json_input["evalEpisodes"]
    update_time = json_input["updateTime"]
    rl_algorithm = json_input["rLAlgorithm"]
    print("Selected algorithm: ", rl_algorithm)

    env = CenteringWorldEnv(size_x, size_y, max_timesteps, distance_percentage, update_time)

    if rl_algorithm == "ppo":
        model_path = "rl_algorithm\envs\logs\ppo\CenteringWorld-v0_1\CenteringWorld-v0.zip"
        model = PPO.load(model_path)

    elif rl_algorithm == "ppo_ext":
        model_path = "rl_algorithm\envs\logs\ppo\CenteringWorld-v1_2\CenteringWorld-v1.zip"
        model = PPO.load(model_path)

    elif rl_algorithm == "a2c":
        model_path = "rl_algorithm\envs\logs\\a2c\CenteringWorld-v0_3\CenteringWorld-v0.zip"
        model = A2C.load(model_path)

    else:
        print("Unknown RL algorithm")
        sys.exit("\nExiting RLEngine\n")

    log_messages = "Loading the Agent for Prediction"
    print(log_messages)
    sio.emit("send_log", log_messages)

    observation, info = env.reset(seed=123, options={})
    # print("Evaluation Episodes: ", eval_episodes, "\n")

    total_reward = 0.0
    episode_rewards = []
    terminated_counter = 0

    # while terminated_counter < eval_episodes:
    while not stop_server:
        action, _ = model.predict(observation, deterministic=True)
        int_action = int(action)
        # print("Action: ", int_action)
        observation, reward, terminated, truncated, info = env.step(int_action)
        # print("obs=", observation, "reward=", reward, "terminated=", terminated)
        total_reward += reward

        if terminated or truncated:
            # print("That was episode: %s\n" % (terminated_counter + 1))
            observation, info = env.reset()
            episode_rewards.append(round(total_reward, 2))
            total_reward = 0.0
            terminated_counter += 1

    env.close()

    # Disabled for convenience
    print("\nEpisode rewards:", episode_rewards, "\n")

    log_messages = "Evaluation Complete"
    sio.emit("send_log", log_messages)
    # sys.exit("\nExiting RLEngine\n")


def start_server():
    """
    This sets up the server connection, with UE acting as the client in a socketIO relationship
    """
    global stop_server
    try:
        if sio.async_mode == "gevent":
            from gevent import pywsgi
            from geventwebsocket.handler import WebSocketHandler

            print("\nRLEngine running, waiting for Unreal Engine to connect\n")
            server = pywsgi.WSGIServer(("localhost", 3000), app, log=None, handler_class=WebSocketHandler)

            try:
                server.serve_forever(stop_timeout=1)
            except KeyboardInterrupt:
                print("\nReceived interrupt, stopping server.\n")
            finally:
                server.stop()
        else:
            print("\nUnknown async_mode: %s\n", sio.async_mode)
    except Exception as e:
        print("\nError starting server: %s\n", e)


if __name__ == "__main__":
    start_server()
