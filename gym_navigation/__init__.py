from gym.envs.registration import register


register(
    id='Navigation10x10-v0',
    entry_point='gym_navigation.envs:Nav_Env',
    kwargs={'env_size': 10}
)
