import logging
import sys

import gym
import numpy as np

# noinspection PyUnresolvedReferences
import gym_grid  # noqa: F401
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
        while not reward == 1000:
            obs, reward, done, _ = grid.step(MOVE_UP)

        # then
        assert obs == ('20', '20')
        assert reward == 1000
        assert done is True

    def test_should_move_in_each_direction(self):
        # given
        grid = gym.make('grid-20-v0')
        np.random.seed(54)
        grid.reset()  # (16, 6)

        # when & then
        state, _, _, _ = grid.step(MOVE_LEFT)
        assert state == ("15", "6")

        state, _, _, _ = grid.step(MOVE_UP)
        assert state == ("15", "7")

        state, _, _, _ = grid.step(MOVE_RIGHT)
        assert state == ("16", "7")

        state, _, _, _ = grid.step(MOVE_DOWN)
        assert state == ("16", "6")

    def test_should_reach_reward(self):
        # given
        grid = gym.make('grid-20-v0')

        # from left
        np.random.seed(128)
        grid.reset()  # (19, 20)
        state, reward, done, _ = grid.step(MOVE_RIGHT)
        assert state == ("20", "20")
        assert reward == 1000
        assert done is True

        # from bottom
        np.random.seed(342)
        grid.reset()  # (20, 19)
        state, reward, done, _ = grid.step(MOVE_UP)
        assert state == ("20", "20")
        assert reward == 1000
        assert done is True

    def test_should_get_all_states_and_actions(self):
        # given
        grid = gym.make('grid-5-v0')
        grid.reset()

        # when
        mapping = grid.env._state_action()

        # then
        assert len(mapping) == 25
