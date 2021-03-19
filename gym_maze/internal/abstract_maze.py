import random
from typing import Tuple, List

import numpy as np

from gym_maze.common import MAZE_PATH, MAZE_ANIMAT, MAZE_WALL
from gym_maze.common.maze_utils import get_possible_insertion_coordinates, \
    get_animat_xy, adjacent_cell_values


class AbstractMaze:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    @property
    def agent_position(self) -> Tuple[int, int]:
        return get_animat_xy(self.matrix)

    def is_done(self) -> bool:
        raise NotImplementedError

    def perception(self, cords: Tuple[int, int] = None) -> List:
        if cords is None:
            cords = self.agent_position

        return list(map(str, adjacent_cell_values(self.matrix, *cords)))

    def insert_agent(self, cords: Tuple[int, int] = None) -> None:
        if cords is not None:
            assert self.matrix[cords] == MAZE_PATH
            self.matrix[cords] = MAZE_ANIMAT
        else:
            possible_cords = get_possible_insertion_coordinates(self.matrix)
            starting_position = random.choice(possible_cords)
            self.matrix[starting_position] = MAZE_ANIMAT

    def is_wall(self, cords: Tuple[int, int] = None):
        if cords is None:
            cords = self.agent_position

        return bool(self.matrix[cords] == MAZE_WALL)

    def is_path(self, cords: Tuple[int, int] = None):
        if cords is None:
            cords = self.agent_position

        return bool(self.matrix[cords] == MAZE_PATH)
