from typing import Tuple, List

import numpy as np

from gym_maze.common import MAZE_WALL, MAZE_PATH, MAZE_ANIMAT, MAZE_REWARD
from gym_maze.internal.abstract_maze import AbstractMaze


class RotatingMazeImpl(AbstractMaze):
    """
    Instead of keeping agent direction the matrix environment is rotated
    """
    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix)
        self.found_reward = False

    def perception(self, cords: Tuple[int, int] = None) -> List:
        cells = super().perception(cords)
        last_bit = str(MAZE_REWARD) if self.found_reward else str(MAZE_PATH)
        return cells + [last_bit]

    def turn_left(self) -> None:
        self.matrix = np.rot90(self.matrix, k=3)

    def turn_right(self) -> None:
        self.matrix = np.rot90(self.matrix)

    def step_ahead(self):
        x, y = self.agent_position
        next_state = (x-1, y)

        assert self.found_reward is False

        if self.matrix[next_state] == MAZE_REWARD:
            # found reward and step on it
            self.matrix[x, y] = MAZE_PATH
            self.matrix[next_state] = MAZE_ANIMAT

            self.found_reward = True

        elif self.matrix[next_state] != MAZE_WALL:
            # perform step ahead
            self.matrix[x, y] = MAZE_PATH
            self.matrix[next_state] = MAZE_ANIMAT

        else:
            # wall - do nothing
            assert self.matrix[next_state] == MAZE_WALL

    def is_done(self) -> bool:
        return self.found_reward
