from gym.envs.registration import register

from gym_handeye.handeye import HandEye
from gym_handeye.handeye import HandEyeSimulator

register(
    id='HandEye3-v0',
    entry_point='gym_handeye:HandEye',
    max_episode_steps=500,
    kwargs={'grid_size': 3, 'note_in_hand': False, 'test_only_changes': 0}
)

register(
    id='HandEye5-v0',
    entry_point='gym_handeye:HandEye',
    max_episode_steps=500,
    kwargs={'grid_size': 5, 'note_in_hand': False, 'test_only_changes': 0}
)

register(
    id='HandEye7-v0',
    entry_point='gym_handeye:HandEye',
    max_episode_steps=500,
    kwargs={'grid_size': 7, 'note_in_hand': False, 'test_only_changes': 0}
)

register(
    id='HandEye9-v0',
    entry_point='gym_handeye:HandEye',
    max_episode_steps=500,
    kwargs={'grid_size': 9, 'note_in_hand': False, 'test_only_changes': 0}
)