from gym_woods.envs import AbstractWoods

import numpy as np


class Woods14(AbstractWoods):
    # TODO: needs fix according to literature
    def __init__(self):
        super().__init__(np.asarray([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
            [1, 9, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]))
