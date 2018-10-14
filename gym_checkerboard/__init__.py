from gym.envs.registration import register

from .checkerboard import Checkerboard

checkerboard_name = "checkerboard"
max_episode_steps = 1

register(
    id=f'{checkerboard_name}-2D-2div-v0',
    entry_point='gym_checkerboard:Checkerboard',
    max_episode_steps=max_episode_steps,
    kwargs={'dim': 2, 'ndiv': 2}
)

register(
    id=f'{checkerboard_name}-2D-3div-v0',
    entry_point='gym_checkerboard:Checkerboard',
    max_episode_steps=max_episode_steps,
    kwargs={'dim': 2, 'ndiv': 3}
)

register(
    id=f'{checkerboard_name}-2D-4div-v0',
    entry_point='gym_checkerboard:Checkerboard',
    max_episode_steps=max_episode_steps,
    kwargs={'dim': 2, 'ndiv': 4}
)

register(
    id=f'{checkerboard_name}-2D-5div-v0',
    entry_point='gym_checkerboard:Checkerboard',
    max_episode_steps=max_episode_steps,
    kwargs={'dim': 2, 'ndiv': 5}
)
