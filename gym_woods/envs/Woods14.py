import numpy as np

from gym_woods.envs import AbstractWoods


class Woods14(AbstractWoods):
    def __init__(self):
        super().__init__(np.asarray([
            list('OOOOOOOOOOOOOO'),
            list('OO...OOOO.OO.O'),
            list('O.OOO.OO.O.O.O'),
            list('O.OOO.O.OOO.OO'),
            list('OFOOO.OO.OOOOO'),
            list('OOOOOO..OOOOOO'),
            list('OOOOOOOOOOOOOO')
        ]))
