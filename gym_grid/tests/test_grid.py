import logging
import sys

import gym
import numpy as np

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
        grid.reset()  # (x=6, y=19)
        state, _, _, _ = grid.step(MOVE_UP)
        assert state == ("6", "19")

        # handle hitting right bound
        np.random.seed(27)
        grid.reset()  # (x=19, y=8)
        state, _, _, _ = grid.step(MOVE_RIGHT)
        assert state == ("19", "8")

        # handle hitting lower bound
        np.random.seed(50)
        grid.reset()  # (x=16, y=0)
        state, _, _, _ = grid.step(MOVE_DOWN)
        assert state == ("16", "0")

        # handle hitting left bound
        np.random.seed(48)
        grid.reset()  # (x=0, y=19)
        state, _, _, _ = grid.step(MOVE_LEFT)
        assert state == ("0", "19")

    def test_should_get_reward(self):
        # given
        grid = gym.make('grid-20-v0')
        reward = 0
        done = False

        # when
        grid.reset()
        for _ in range(20):
            grid.step(MOVE_RIGHT)
        while not reward == 1000:
            obs, reward, done, _ = grid.step(MOVE_UP)

        # then
        assert obs == ('19', '19')
        assert reward == 1000
        assert done is True

    def test_should_move_in_each_direction(self):
        # given
        grid = gym.make('grid-20-v0')
        np.random.seed(54)
        grid.reset()  # (15, 5)

        # when & then
        state, _, _, _ = grid.step(MOVE_LEFT)
        assert state == ("14", "5")

        state, _, _, _ = grid.step(MOVE_UP)
        assert state == ("14", "6")

        state, _, _, _ = grid.step(MOVE_RIGHT)
        assert state == ("15", "6")

        state, _, _, _ = grid.step(MOVE_DOWN)
        assert state == ("15", "5")

    def test_should_reach_reward(self):
        # given
        grid = gym.make('grid-20-v0')

        # from left
        np.random.seed(128)
        grid.reset()  # (18, 19)
        state, reward, done, _ = grid.step(MOVE_RIGHT)
        assert state == ("19", "19")
        assert reward == 1000
        assert done is True

        # from bottom
        np.random.seed(342)
        grid.reset()  # (19, 18)
        state, reward, done, _ = grid.step(MOVE_UP)
        assert state == ("19", "19")
        assert reward == 1000
        assert done is True

    def test_should_get_all_states_and_actions(self):
        # given
        grid = gym.make('grid-5-v0')

        # when
        mapping = grid.env._state_action()

        # then
        assert len(mapping) == 25

    def test_should_get_all_transitions(self):
        # given
        grid = gym.make('grid-5-v0')
        grid.reset()

        # when
        transitions = grid.env.get_transitions()

        # then
        assert len(transitions) == 78
