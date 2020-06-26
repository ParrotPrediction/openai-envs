import numpy as np

from gym_woods.envs import AbstractWoods


class Woods1(AbstractWoods):
    def __init__(self):
        super().__init__(np.asarray([
            list('.......................................................'),
            list('.OOF..OOF..OOF..OOF..OOF..OOF..OOF..OOF..OOF..OOF..OOF.'),
            list('.OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO.'),
            list('.OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO.'),
            list('.......................................................'),
            list('.......................................................'),
            list('.OOF..OOF..OOF..OOF..OOF..OOF..OOF..OOF..OOF..OOF..OOF.'),
            list('.OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO.'),
            list('.OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO.'),
            list('.......................................................'),
            list('.......................................................'),
            list('.OOF..OOF..OOF..OOF..OOF..OOF..OOF..OOF..OOF..OOF..OOF.'),
            list('.OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO.'),
            list('.OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO..OOO.'),
            list('.......................................................'),
        ]))
