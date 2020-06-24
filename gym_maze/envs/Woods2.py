from gym_maze.envs import AbstractMaze

import numpy as np


class Woods1(AbstractMaze):
    def __init__(self):

        # ...... x30


        super().__init__(np.asarray([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 9, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]))
