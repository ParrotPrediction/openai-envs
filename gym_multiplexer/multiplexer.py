from typing import List

import gym

from .utils import get_correct_answer


class Multiplexer(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    REWARD = 1000

    def _generate_state(self): raise NotImplementedError

    def _internal_state(self): raise NotImplementedError

    def __init__(self, control_bits=3) -> None:
        self.control_bits = control_bits

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
            super(Multiplexer, self).render(mode=mode)

    @property
    def _observation(self) -> list:
        observation = list(self._state)
        observation.append(self._validation_bit)
        return list(map(float, observation))

    @property
    def _correct_answer(self):
        return get_correct_answer(list(self._internal_state()),
                                  self.control_bits)

    @property
    def _observation_string_length(self):
        return self.control_bits + pow(2, self.control_bits) + 1
