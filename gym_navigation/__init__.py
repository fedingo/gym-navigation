from gym.envs.registration import register


register(
    id='Navigation10x10-v0',
    entry_point='gym_navigation.envs:Nav_Env',
    timestep_limit=50,
    kwargs={'env_size': 10}
)
