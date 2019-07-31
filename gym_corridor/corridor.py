from random import randint

import gym
from gym.spaces import Discrete


# 0 - move left
# 1 - move right
class Corridor(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    REWARD = 1000

    def __init__(self, size=20):
        self._size = size
        self._position = None

        self.observation_space = Discrete(1)
        self.action_space = Discrete(2)

    def reset(self):
        self._position = randint(1, self._size - 1)
        return str(self._position)

    def step(self, action):
        if action == 0:
            self._position -= 1
        elif action == 1:
            self._position += 1
        else:
            raise ValueError("Illegal action passed")

        if self._position == self._size:
            return str(self._position), self.REWARD, True, None

        if self._position == 0:
            self._position = 1

        return str(self._position), 0, False, None

    def render(self, mode='human'):
        if mode == 'human':
            print(self._visualize())
        elif mode == 'ansi':
            return self._visualize()
        else:
            raise ValueError('Unknown visualisation mode')

    def _visualize(self):
        corridor = ["" for _ in range(0, self._size - 1)]
        corridor[self._position - 1] = "X"
        corridor[self._size - 2] = "$"
        return "[" + ".".join(corridor) + "]"
