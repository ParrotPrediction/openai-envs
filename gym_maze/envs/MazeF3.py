from gym_maze.envs import AbstractMaze

import numpy as np


class MazeF3(AbstractMaze):
    def __init__(self):
        super().__init__(np.asarray([
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 9, 1],
            [1, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]))
