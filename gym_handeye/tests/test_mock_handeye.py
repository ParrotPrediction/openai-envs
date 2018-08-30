import logging
import random
import sys
import unittest

# noinspection PyUnresolvedReferences
import gym_handeye
from gym_handeye import MockHandEye

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


class TestMockHandEye(unittest.TestCase):
    def test_parse_observe(self):
        start = ["g","w","w","w","w","w","w","w","b","0"]
        mock = MockHandEye(3, True, False)
        mock.parse_observation(start)
        end = mock.observe()
        self.assertEqual(start,end)

    def test_block_in_hand(self):
        start = ["w", "w", "w", "w", "w", "w", "w", "w", "b", "2"]
        mock = MockHandEye(3, True, False)
        mock.parse_observation(start)
        end = mock.observe()
        self.assertEqual(start, end)

    def test_block_not_in_hand(self):
        start = ["w", "w", "w", "w", "w", "w", "w", "w", "g", "1"]
        mock = MockHandEye(3, True, False)
        mock.parse_observation(start)
        end = mock.observe()
        self.assertEqual(start, end)

    def test_move_north_not_in_hand(self):
        start = ["w", "w", "w", "w", "w", "w", "w", "w", "g", "1"]
        mock = MockHandEye(3, True, False)
        mock.parse_observation(start)
        mock.take_action(0) # move north
        end = mock.observe()
        self.assertEqual(["w", "w", "w", "w", "w", "g", "w", "w", "b", "0"], end)

    def test_move_north_in_hand(self):
        start = ["w", "w", "w", "w", "w", "w", "w", "w", "b", "2"]
        mock = MockHandEye(3, True, False)
        mock.parse_observation(start)
        mock.take_action(0) # move north
        end = mock.observe()
        self.assertEqual(["w", "w", "w", "w", "w", "b", "w", "w", "w", "2"], end)

    @staticmethod
    def _random_action():
        return random.choice(list(range(6)))


if __name__ == '__main__':
    unittest.main()
