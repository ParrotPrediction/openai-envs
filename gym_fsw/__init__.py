from gym.envs.registration import register

from .fsw import FiniteStateWorld  # noqa: F401

register(
    id='fsw-5-v0',
    entry_point='gym_fsw:FiniteStateWorld',
    kwargs={'size': 5}
)

register(
    id='fsw-10-v0',
    entry_point='gym_fsw:FiniteStateWorld',
    kwargs={'size': 10}
)

register(
    id='fsw-20-v0',
    entry_point='gym_fsw:FiniteStateWorld',
    kwargs={'size': 20}
)

register(
    id='fsw-40-v0',
    entry_point='gym_fsw:FiniteStateWorld',
    kwargs={'size': 40}
)
