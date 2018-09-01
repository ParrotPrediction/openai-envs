import logging
import random
import sys
import unittest

import gym

# noinspection PyUnresolvedReferences
import gym_handeye

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

class TestHandEye(unittest.TestCase):
    def test_initialize(self):
        # given, when
        he = gym.make('HandEye3-v0')

        self.assertNotEqual(he, None)
        self.assertEqual(10, he.observation_space.n)
        self.assertEqual(6, he.action_space.n)

    def test_return_observation_when_reset(self):
        # given
        he = gym.make('HandEye3-v0')

        # when
        state = he.reset()

        # then
        self.assertNotEqual(state, None)
        self.assertEqual(10, len(state))
        self.assertEqual(tuple, type(state))
        for i, obs in enumerate(state):
            if i < 9:
                self.assertIn(obs, ['w', 'b', 'g'])
            else:
                self.assertIn(obs, ['0', '1', '2'])

    def test_execute_step(self):
        # given
        he = gym.make('HandEye3-v0')
        he.reset()

        # when
        action = self._random_action()
        state, reward, done, _ = he.step(action)

        # then
        self.assertNotEqual(state, None)
        self.assertEqual(tuple, type(state))
        self.assertIn(reward, [0, 1000])
        self.assertFalse(done)
        for i, obs in enumerate(state):
            if i < 9:
                self.assertIn(obs, ['w', 'b', 'g'])
            else:
                self.assertIn(obs, ['0', '1', '2'])

    def test_execute_multiple_steps_and_keep_constant_perception_length(self):
        # given
        he = gym.make('HandEye3-v0')
        steps = 100

        for _ in range(0, steps):
            # when
            p0 = he.reset()

            # then
            self.assertEqual(10, len(p0))

            # when
            action = self._random_action()
            p1, reward, done, _ = he.step(action)

            # then
            self.assertEqual(10, len(p1))

    @staticmethod
    def _random_action():
        return random.choice(list(range(6)))
