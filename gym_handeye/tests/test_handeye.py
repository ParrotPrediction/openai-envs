import logging
import random
import sys

import gym

# noinspection PyUnresolvedReferences
import gym_handeye  # noqa: F401

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


class TestHandEye:
    def test_initialize(self):
        # given, when
        he = gym.make('HandEye3-v0')

        # then
        assert he is not None
        assert 10 == he.observation_space.n
        assert 6 == he.action_space.n

    def test_return_observation_when_reset(self):
        # given
        he = gym.make('HandEye3-v0')

        # when
        state = he.reset()

        # then
        assert state is not None
        assert 10 == len(state)
        assert tuple == type(state)
        for i, obs in enumerate(state):
            if i < 9:
                assert obs in ['w', 'b', 'g']
            else:
                assert obs in ['0', '1', '2']

    def test_execute_step(self):
        # given
        he = gym.make('HandEye3-v0')
        he.reset()

        # when
        action = self._random_action()
        state, reward, done, _ = he.step(action)

        # then
        assert state is not None
        assert tuple == type(state)
        assert done is False
        for i, obs in enumerate(state):
            if i < 9:
                assert obs in ['w', 'b', 'g']
            else:
                assert obs in ['0', '1', '2']

    def test_execute_multiple_steps_and_keep_constant_perception_length(self):
        # given
        he = gym.make('HandEye3-v0')
        steps = 100

        for _ in range(0, steps):
            # when
            start = he.reset()

            # then
            assert 10 == len(start)

            # when
            action = self._random_action()
            end, reward, done, _ = he.step(action)

            # then
            assert 10 == len(end)

    def test_get_all_possible_transitions(self):
        # given
        he = gym.make('HandEye3-v0')

        # when
        transitions = he.env.get_all_possible_transitions()

        # then
        assert 258 == len(transitions)
        # 258 is a number from article for grid_size = 3

    @staticmethod
    def _random_action():
        return random.choice(list(range(6)))
