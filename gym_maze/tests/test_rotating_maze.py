import numpy as np
import gym
import pytest

import gym_maze  # noqa: F401
from gym_maze.common import MAZE_PATH


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
        matrix = maze.env.maze.matrix

        # when
        cords = np.where(matrix == MAZE_PATH)
        assert len(cords) == 2

        # then
        assert len(cords[0]) == _count
        assert len(cords[1]) == _count
