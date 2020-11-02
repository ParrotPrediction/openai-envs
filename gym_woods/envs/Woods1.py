import numpy as np

from gym_woods.envs import AbstractWoods


class Woods1(AbstractWoods):
    def __init__(self):
        super().__init__(np.asarray([
            list('.....'),
            list('.OOF.'),
            list('.OOO.'),
            list('.OOO.'),
            list('.....'),
        ]))
