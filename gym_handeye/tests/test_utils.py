import logging
import random
import sys
import unittest

import gym

# noinspection PyUnresolvedReferences
import gym_handeye

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


class TestHandEye(unittest.TestCase):
    def test_get_all_possible_transitions(self):
        # given
        he = gym.make('HandEye3-v0')

        # when
        transitions = he.env.get_all_possible_transitions()

        # then
        self.assertEqual(258, len(transitions))
        # 258 is a number from article for grid_size = 3
