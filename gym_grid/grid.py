import gym
import numpy as np
from gym.spaces import Discrete

MOVE_LEFT = 0
MOVE_RIGHT = 1
MOVE_UP = 2
MOVE_DOWN = 3

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

        self.reset()

    @property
    def _state(self):
        return str(self._pos_x), str(self._pos_y)

    @property
    def _in_reward(self):
        return self._pos_x == self._size and self._pos_y == self._size

    def reset(self):
        (self._pos_x, self._pos_y) = np.random.randint(
            1, self._size + 1, size=2)

        if self._in_reward:
            self.reset()

        return self._state

    def step(self, action):
        if action == MOVE_LEFT:
            if self._pos_x - 1 >= 1:
                self._pos_x -= 1
        elif action == MOVE_RIGHT:
            if self._pos_x + 1 <= self._size:
                self._pos_x += 1
        elif action == MOVE_UP:
            if self._pos_y + 1 <= self._size:
                self._pos_y += 1
        elif action == MOVE_DOWN:
            if self._pos_y - 1 >= 1:
                self._pos_y -= 1
        else:
            raise ValueError("Illegal action passed")

        # Handle reaching final state
        if self._in_reward:
            return self._state, self.REWARD, True, None

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

    def _state_action(self):
        """
        Return states and possible actions in each of them
        """

        # Assign all actions for all states
        mapping = {}
        for x in range(1, self._size + 1):
            for y in range(1, self._size + 1):
                mapping[(x, y)] = [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN]

        # Remove actions from certain states
        top_row = dict(filter(lambda i: i[0][0] == self._size, mapping.items()))
        bottom_row = dict(filter(lambda i: i[0][0] == 1, mapping.items()))
        left_col = dict(filter(lambda i: i[0][1] == 1, mapping.items()))
        right_col = dict(filter(lambda i: i[0][1] == self._size, mapping.items()))

        for actions in top_row.values():
            actions.remove(MOVE_UP)

        for actions in bottom_row.values():
            actions.remove(MOVE_DOWN)

        for actions in left_col.values():
            actions.remove(MOVE_LEFT)

        for actions in right_col.values():
            actions.remove(MOVE_RIGHT)

        # No actions possible when found reward
        mapping[(self._size, self._size)] = []

        # Cast (int, int) key to (str, str)
        mapping = {(str(k[0]), str(k[1])): v for k, v in mapping.items()}

        return mapping
