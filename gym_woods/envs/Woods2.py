import numpy as np

from gym_woods.envs import AbstractWoods


class Woods2(AbstractWoods):
    def __init__(self):
        super().__init__(np.asarray([
            list('..............................'),
            list('.QQF..QQF..OQF..QQG..OQG..OQF.'),
            list('.OOO..QOO..OQO..OOQ..QQO..QQQ.'),
            list('.OOQ..OQQ..OQQ..QQO..OOO..QQO.'),
            list('..............................'),
            list('..............................'),
            list('.QOF..QOG..QOF..OOF..OOG..QOG.'),
            list('.QQO..QOO..OOO..OQO..QQO..QOO.'),
            list('.QQQ..OOO..OQO..QOQ..QOQ..OQO.'),
            list('.QQQ..OOO..OQO..QOQ..QOQ..OQO.'),
            list('..............................'),
            list('..............................'),
            list('.QOG..QOF..OOG..OQF..OOG..OOF.'),
            list('.OOQ..OQQ..QQO..OQQ..QQO..OQQ.'),
            list('.QQO..OOO..OQO..OOQ..OQQ..QQQ.'),
            list('..............................')
        ]))
