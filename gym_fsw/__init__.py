from gym.envs.registration import register

from .fsw import FiniteStateWorld

register(
    id='fsw-5-v0',
    entry_point='gym_fsw:FiniteStateWorld',
    kwargs={'size': 5}
)
