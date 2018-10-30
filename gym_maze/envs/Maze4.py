from gym_maze.envs import AbstractMaze

import numpy as np


class Maze4(AbstractMaze):
    def __init__(self):
        super().__init__(np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 9, 1],
            [1, 1, 0, 0, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]))
        self.search = True

    def get_goal_state(self):
        # return '1', '1', '1', '1', '0', '0', '0', '1'
        if self.search:
            self.search = False
            return '0', '9', '0', '1', '0', '0', '1', '0'
        else:
            self.search = True
            return None
