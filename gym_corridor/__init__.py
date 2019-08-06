from gym.envs.registration import register

from .corridor import Corridor

max_episode_steps = 200

register(
    id='corridor-20-v0',
    entry_point='gym_corridor:Corridor',
    max_episode_steps=max_episode_steps,
    kwargs={'size': 20}
)

register(
    id='corridor-40-v0',
    entry_point='gym_corridor:Corridor',
    max_episode_steps=max_episode_steps,
    kwargs={'size': 40}
)

register(
    id='corridor-100-v0',
    entry_point='gym_corridor:Corridor',
    max_episode_steps=max_episode_steps,
    kwargs={'size': 100}
)
