from gym.envs.registration import register

from .grid import Grid

max_episode_steps = 2000

register(
    id='grid-5-v0',
    entry_point='gym_grid:Grid',
    max_episode_steps=max_episode_steps,
    kwargs={'size': 5}
)

register(
    id='grid-10-v0',
    entry_point='gym_grid:Grid',
    max_episode_steps=max_episode_steps,
    kwargs={'size': 10}
)

register(
    id='grid-20-v0',
    entry_point='gym_grid:Grid',
    max_episode_steps=max_episode_steps,
    kwargs={'size': 20}
)

register(
    id='grid-40-v0',
    entry_point='gym_grid:Grid',
    max_episode_steps=max_episode_steps,
    kwargs={'size': 40}
)

register(
    id='grid-100-v0',
    entry_point='gym_grid:Grid',
    max_episode_steps=max_episode_steps,
    kwargs={'size': 100}
)
