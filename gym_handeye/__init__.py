from gym.envs.registration import register

from .handeye import HandEye


register(
    id='HandEye3-v0',
    entry_point='gym_handeye:HandEye',
    max_episode_steps=50,
    kwargs={'grid_size' : 3, 'note_in_hand' : True, 'test_only_changes' : False}
)
