import gym
import numpy as np
from gym.spaces import Discrete

MOVE_LEFT = 0
MOVE_RIGHT = 1
MOVE_UP = 3
MOVE_DOWN = 4

# Food located in [n, n]
# Observation x,y in [1, n]


class Grid(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    REWARD = 1000

    def __init__(self, size=20):
        self._size = size
        self._pos_x = None
        self._pos_y = None

        self.observation_space = Discrete(2)
        self.action_space = Discrete(4)

    @property
    def _state(self):
        return str(self._pos_x), str(self._pos_y)

    def reset(self):
        (self._pos_x, self._pos_y) = np.random.randint(
            1, self._size + 1, size=2)

        if self._pos_x == self._size and self._pos_y == self._size:
            self.reset()

        return self._state

    def step(self, action):
        if action == MOVE_LEFT:
            self._pos_x -= 1
        elif action == MOVE_RIGHT:
            self._pos_x += 1
        elif action == MOVE_UP:
            self._pos_y += 1
        elif action == MOVE_DOWN:
            self._pos_y -= 1
        else:
            raise ValueError("Illegal action passed")

        # Handle reaching final state
        if self._pos_x == self._size and self._pos_y == self._size:
            return self._state, self.REWARD, True, None

        # Handle leaving grid
        if self._pos_x == 0:
            self._pos_x = 1
        elif self._pos_x == 21:
            self._pos_x = 20

        if self._pos_y == 0:
            self._pos_y = 1
        elif self._pos_y == 21:
            self._pos_y = 20

        # Return default observation
        return self._state, 0, False, None

    def render(self, mode='human'):
        if mode == 'human':
            print(self._visualize())
        elif mode == 'ansi':
            return self._visualize()
        else:
            raise ValueError('Unknown visualisation mode')

    def _visualize(self):
        print("")
        print(self._state)
        for y in reversed(range(0, self._size + 1)):
            for x in range(0, self._size + 1):
                if x == 0 and y == 0:
                    print(f"{'':^3}", end='')
                elif x == 0:
                    print(f"{y:>3}", end='')
                elif y == 0:
                    print(f"{x:^3}", end='')
                elif x == self._pos_x and y == self._pos_y:
                    print(f"{'X':^3}", end='')
                elif x == self._size and y == self._size:
                    print(f"{'$':^3}", end='')
                else:
                    print(f"{'_':^3}", end='')
            print("")
