import logging

import gym
import numpy as np
from gym.spaces import Discrete, Box

from gym_checkerboard.checkerboard_simulator import CheckerboardSimulator

logger = logging.getLogger(__name__)

class Checkerboard(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    REWARD = 1

    def __init__(self, dim: int, ndiv: int):
        logger.debug("Initializing environment")
        self._dim = dim
        self._board = CheckerboardSimulator(dim, ndiv)
        self._state = None
        self._validation_bit = 0

        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=1,
                                     shape=(dim + 1,),
                                     dtype=np.float32)

    def reset(self):
        logger.debug("Resetting environment")
        self._state = self._generate_state()
        self._validation_bit = 0
        return self._observation

    def step(self, action):
        logger.debug(f"Performing action {action}")
        reward = 0

        if action == self._true_color:
            self._validation_bit = 1
            reward = self.REWARD

        return self._observation, reward, True, None

    def render(self, mode='human'):
        if mode == 'human':
            print(self._observation)
        elif mode == 'ansi':
            return self._observation
        else:
            super(Checkerboard, self).render(mode=mode)

    def _generate_state(self):
        return np.random.rand(self._dim)

    @property
    def _observation(self) -> list:
        observation = list(self._state)
        observation.append(self._validation_bit)
        return observation

    @property
    def _true_color(self) -> int:
        return self._board.get_color(*self._state)