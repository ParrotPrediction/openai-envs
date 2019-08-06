import logging
import sys

import gym
import numpy as np

# noinspection PyUnresolvedReferences
import gym_grid
from gym_grid.grid import MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


class TestGrid:

    def test_should_initialize(self):
        # when
        grid = gym.make('grid-20-v0')

        # then
        assert grid is not None
        assert 2 == grid.observation_space.n
        assert 4 == grid.action_space.n

    def test_should_handle_hitting_boundaries(self):
        # given
        grid = gym.make('grid-20-v0')

        # handle hitting upper bound
        np.random.seed(42)
        grid.reset()  # (x=7, y=20)
        state, _, _, _ = grid.step(MOVE_UP)
        assert state == ("7", "20")

        # handle hitting right bound
        np.random.seed(27)
        grid.reset()  # (x=20, y=9)
        state, _, _, _ = grid.step(MOVE_RIGHT)
        assert state == ("20", "9")

        # handle hitting lower bound
        np.random.seed(50)
        grid.reset()  # (x=17, y=1)
        state, _, _, _ = grid.step(MOVE_DOWN)
        assert state == ("17", "1")

        # handle hitting left bound
        np.random.seed(48)
        grid.reset()  # (x=1, y=20)
        state, _, _, _ = grid.step(MOVE_LEFT)
        assert state == ("1", "20")

    def test_should_get_reward(self):
        # given
        grid = gym.make('grid-20-v0')
        reward = 0
        done = False

        # when
        grid.reset()
        for _ in range(0, 20):
            grid.step(MOVE_RIGHT)
        while not done:
            obs, reward, done, _ = grid.step(MOVE_UP)

        # then
        assert obs == ('20', '20')
        assert reward == 1000
        assert done is True
