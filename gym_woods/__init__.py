from gym.envs.registration import register

register(
    id='Woods1-v0',
    entry_point='gym_woods.envs:Woods1',
    max_episode_steps=500,
    nondeterministic=False
)

register(
    id='Woods2-v0',
    entry_point='gym_woods.envs:Woods2',
    max_episode_steps=500,
    nondeterministic=False
)

register(
    id='Woods14-v0',
    entry_point='gym_woods.envs:Woods14',
    max_episode_steps=500,
    nondeterministic=False
)

