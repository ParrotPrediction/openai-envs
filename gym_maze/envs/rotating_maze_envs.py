import numpy as np

from gym_maze import RotatingMaze


class Maze228(RotatingMaze):
    # 19 non-terminal cells and 19 x 4 x 3 = 228 transitions
    def __init__(self):
        super().__init__(np.asarray([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 9, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ]))


class Maze252(RotatingMaze):
    # 21 non-terminal cells and 21 x 4 x 3 = 252 transitions
    def __init__(self):
        super().__init__(np.asarray([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 9, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]))


class Maze288(RotatingMaze):
    # 25 non-terminal cells and 25 x 4 x 3 = 288 transitions
    def __init__(self):
        super().__init__(np.asarray([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 9, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]))


class Maze324(RotatingMaze):
    # 26 non-terminal cells and 26 x 4 x 3 = 324 transitions
    def __init__(self):
        super().__init__(np.asarray([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 9, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]))
