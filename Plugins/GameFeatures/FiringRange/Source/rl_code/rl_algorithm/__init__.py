from gymnasium.envs.registration import register

register(
    id="ContinuousWorld-v0",
    entry_point="rl_algorithm.envs:ContinuousWorldEnv",
)

register(
    id="CenteringWorld-v0",
    entry_point="rl_algorithm.envs:CenteringWorldEnv",
)

register(
    id="CenteringWorld-v1",
    entry_point="rl_algorithm.envs:CenteringWorldExtEnv",
)