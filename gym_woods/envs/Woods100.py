import numpy as np

from gym_woods.envs import AbstractWoods


class Woods100(AbstractWoods):
    def __init__(self):
        super().__init__(np.asarray([
            list('OOOOOOOOO'),
            list('O...F...O'),
            list('OOOOOOOOO'),
        ]))
