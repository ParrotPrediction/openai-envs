import numpy as np
import gym
import pytest

import gym_maze  # noqa: F401
from gym_maze.common import MAZE_PATH, MAZE_ANIMAT


class TestRotatingMaze:
    @pytest.mark.parametrize("_env_name, _count", [
        ('Maze228-v0', 19),
        ('Maze252-v0', 21),
        # ('Maze288-v0', 25),  # TODO investigate who is wrong
        # ('Maze324-v0', 26),
    ])
    def test_should_have_proper_terminal_states(self, _env_name, _count):
        # given
        maze = gym.make(_env_name)

        # when
        matrix = maze.env.maze.matrix
        cords = np.where(matrix == MAZE_PATH)
        assert len(cords) == 2

        # then
        assert len(cords[0]) == _count
        assert len(cords[1]) == _count

    def test_should_reset_the_environment(self):
        # given
        env = gym.make('Maze228-v0')

        assert env.env.maze.matrix is not None
        assert np.sum(np.where(env.env.maze.matrix == MAZE_ANIMAT, 1, 0)) == 0

        # when & then
        env.reset()
        assert np.sum(np.where(env.env.maze.matrix == MAZE_ANIMAT, 1, 0)) == 1

        env.reset()
        assert np.sum(np.where(env.env.maze.matrix == MAZE_ANIMAT, 1, 0)) == 1

    @pytest.mark.parametrize("_env_name, _count", [
        ('Maze228-v0', 228)
    ])
    def test_should_calculate_transitions(self, _env_name, _count):
        # given
        env = gym.make(_env_name)

        # when
        transitions = env.env.get_all_possible_transitions()

        # then
        assert len(transitions) == _count

