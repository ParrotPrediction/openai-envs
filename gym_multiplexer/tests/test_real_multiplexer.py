import logging
import random
import sys
import pytest

import gym

# noinspection PyUnresolvedReferences
import gym_multiplexer  # noqa: F401

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


class TestRealMultiplexer:

    @pytest.mark.parametrize("_env_name, _obs_space", [
        ('real-multiplexer-3bit-v0', 4),
        ('real-multiplexer-6bit-v0', 7),
        ('real-multiplexer-11bit-v0', 12),
        ('real-multiplexer-20bit-v0', 21),
    ])
    def test_should_initialize_real_mpx(self, _env_name, _obs_space):
        # when
        mp = gym.make(_env_name)

        # then
        assert mp is not None
        assert (_obs_space,) == mp.observation_space.shape
        assert mp.action_space.n == 2

    def test_should_return_observation_when_reset(self):
        # given
        mp = gym.make('real-multiplexer-6bit-v0')

        # when
        state = mp.reset()

        # then
        assert state is not None
        assert state[-1] == 0.0
        assert 7 == len(state)
        assert type(state) is list
        for attrib in state:
            assert type(attrib) is float

    def test_should_execute_step(self):
        # given
        mp = gym.make('real-multiplexer-6bit-v0')
        mp.reset()
        action = self._random_action()

        # when
        state, reward, done, _ = mp.step(action)

        # then
        assert state is not None
        assert type(state[-1]) is float
        assert state[-1] in [0., 1.]
        assert type(state) is list
        assert reward in [0, 1000]
        assert done is True

    def test_execute_multiple_steps_and_keep_constant_perception_length(self):
        # given
        mp = gym.make('real-multiplexer-6bit-v0')
        steps = 100

        # when & then
        for _ in range(0, steps):
            p0 = mp.reset()
            assert 7 == len(p0)

            action = self._random_action()
            p1, reward, done, _ = mp.step(action)
            assert 7 == len(p1)

    @staticmethod
    def _random_action():
        return random.sample([0, 1], 1)[0]
