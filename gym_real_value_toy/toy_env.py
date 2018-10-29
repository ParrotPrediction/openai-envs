from typing import List, Iterable

import gym
import numpy as np
from gym.spaces import Discrete, Box


class ToyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    REWARD = 1

    def _correct_answer(self) -> bool:
        raise NotImplementedError

    def __init__(self) -> None:
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=1,
                                     shape=(2 + 1,),
                                     dtype=np.float32)

        self._state: List = []
        self._validation_bit = 0

    def reset(self):
        self._state = self._generate_state()
        self._validation_bit = 0
        return self._observation

    def step(self, action):
        reward = 0

        if action == self._correct_answer:
            self._validation_bit = 1
            reward = self.REWARD

        return self._observation, reward, True, None

    def render(self, mode='human'):
        if mode == 'human':
            print(self._observation)
        elif mode == 'ansi':
            return self._observation
        else:
            super(ToyEnv, self).render(mode=mode)

    @property
    def _observation(self) -> list:
        observation = list(self._state)
        observation.append(self._validation_bit)
        return observation

    @staticmethod
    def _generate_state() -> Iterable[float]:
        return np.random.random(2).tolist()
