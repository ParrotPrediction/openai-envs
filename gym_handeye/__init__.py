from gym.envs.registration import register

from gym_handeye.handeye import HandEye
from gym_handeye.handeye import HandEyeSimulator

register(
    id='HandEye3-v0',
    entry_point='gym_handeye:HandEye',
    max_episode_steps=500,
    kwargs={'grid_size': 3, 'note_in_hand': True, 'test_only_changes': 0}
)
