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
        he = gym.make('HandEye3-v0')

        self.assertNotEqual(he, None)
        self.assertEqual(10, he.observation_space.n)
        self.assertEqual(6, he.action_space.n)

    def test_return_observation_when_reset(self):
        he = gym.make('HandEye3-v0')

        state = he.reset()

        self.assertNotEqual(state, None)
        self.assertEqual(10, len(state))
        self.assertEqual(list, type(state))

        for i, obs in enumerate(state):
            if i < 9:
                self.assertIn(obs,['w', 'b', 'g', '0', '1', '2'])
            else:
                self.assertIn(obs, ['0', '1', '2'])

    def test_should_render_state(self):
        he = gym.make('HandEye3-v0')
        he.reset()

        state = he.render()

        self.assertNotEqual(state, None)
        self.assertEqual(10, len(state))
        self.assertEqual(list, type(state))
        for i, obs in enumerate(state):
            if i < 9:
                self.assertIn(obs,['w', 'b', 'g', '0', '1', '2'])
            else:
                self.assertIn(obs, ['0', '1', '2'])

    def test_execute_step(self):
        he = gym.make('HandEye3-v0')
        he.reset()
        action = self._random_action()

        state, reward, done, _ = he.step(action)

        self.assertNotEqual(state, None)
        self.assertEqual(list, type(state))
        self.assertIn(reward, [0, 1000])
        self.assertFalse(done)
        for i, obs in enumerate(state):
            if i < 9:
                self.assertIn(obs,['w', 'b', 'g', '0', '1', '2'])
            else:
                self.assertIn(obs, ['0', '1', '2'])

    def test_execute_multiple_steps_and_keep_constant_perception_length(self):
        he = gym.make('HandEye3-v0')
        steps = 100

        for _ in range(0, steps):
            p0 = he.reset()
            self.assertEqual(10, len(p0))

            action = self._random_action()
            p1, reward, done, _ = he.step(action)
            self.assertEqual(10, len(p1))

    @staticmethod
    def _random_action():
        return random.choice(list(range(6)))


if __name__ == '__main__':
    unittest.main()
