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
        self._transitions = self._calculate_transitions()

        self.reset()

    @property
    def _state(self):
        return str(self._pos_x), str(self._pos_y)

    @property
    def _in_reward(self):
        return self._pos_x == self._size - 1 and self._pos_y == self._size - 1

    def reset(self):
        (self._pos_x, self._pos_y) = np.random.randint(self._size, size=2)

        if self._in_reward:
            self.reset()

        return self._state

    def step(self, action):
        if action == MOVE_LEFT:
            if self._pos_x - 1 >= 0:
                self._pos_x -= 1
        elif action == MOVE_RIGHT:
            if self._pos_x + 1 < self._size:
                self._pos_x += 1
        elif action == MOVE_UP:
            if self._pos_y + 1 < self._size:
                self._pos_y += 1
        elif action == MOVE_DOWN:
            if self._pos_y - 1 >= 0:
                self._pos_y -= 1
        else:
            raise ValueError("Illegal action passed")

        # Handle reaching final state
        if self._in_reward:
            return self._state, self.REWARD, True, {}

        # Return default observation
        return self._state, 0, False, {}

    def render(self, mode='human'):
        if mode == 'human':
            print(self._visualize())
        elif mode == 'ansi':
            return self._visualize()
        else:
            raise ValueError('Unknown visualisation mode')

    def get_transitions(self):
        return self._transitions

    def _visualize(self):
        print("")
        print(self._state)
        for y in reversed(range(-1, self._size)):
            for x in range(-1, self._size):
                if x == -1 and y == -1:
                    print(f"{'':^3}", end='')
                elif x == -1:
                    print(f"{y:>3}", end='')
                elif y == -1:
                    print(f"{x:^3}", end='')
                elif x == self._pos_x and y == self._pos_y:
                    print(f"{'X':^3}", end='')
                elif x == self._size and y == self._size:
                    print(f"{'$':^3}", end='')
                else:
                    print(f"{'_':^3}", end='')
            print("")

    def _calculate_transitions(self):
        MAX_POS = self._size - 1

        def _handle_state(state):
            moves = []
            (x, y) = state

            # handle inner rectangle - 4 actions available
            if 0 < x < MAX_POS and 0 < y < MAX_POS:
                moves.append(((x, y), MOVE_LEFT, (x - 1, y)))
                moves.append(((x, y), MOVE_RIGHT, (x + 1, y)))
                moves.append(((x, y), MOVE_UP, (x, y + 1)))
                moves.append(((x, y), MOVE_DOWN, (x, y - 1)))

            # handle bounds (except corners) - 3 actions available
            if x == 0 and y not in [0, MAX_POS]:  # left bound
                moves.append(((x, y), MOVE_RIGHT, (x + 1, y)))
                moves.append(((x, y), MOVE_UP, (x, y + 1)))
                moves.append(((x, y), MOVE_DOWN, (x, y - 1)))

            if x == MAX_POS and y not in [0, MAX_POS]:  # right bound
                moves.append(((x, y), MOVE_LEFT, (x - 1, y)))
                moves.append(((x, y), MOVE_UP, (x, y + 1)))
                moves.append(((x, y), MOVE_DOWN, (x, y - 1)))

            if x not in [0, MAX_POS] and y == 0:  # lower bound
                moves.append(((x, y), MOVE_LEFT, (x - 1, y)))
                moves.append(((x, y), MOVE_RIGHT, (x + 1, y)))
                moves.append(((x, y), MOVE_UP, (x, y + 1)))

            if x not in [0, MAX_POS] and y == MAX_POS:  # upper bound
                moves.append(((x, y), MOVE_LEFT, (x - 1, y)))
                moves.append(((x, y), MOVE_RIGHT, (x + 1, y)))
                moves.append(((x, y), MOVE_DOWN, (x, y - 1)))

            # handle corners - 2 actions available
            if x == 0 and y == 0:  # left-down
                moves.append(((x, y), MOVE_RIGHT, (x + 1, y)))
                moves.append(((x, y), MOVE_UP, (x, y + 1)))

            if x == 0 and y == MAX_POS:  # left-up
                moves.append(((x, y), MOVE_RIGHT, (x + 1, y)))
                moves.append(((x, y), MOVE_DOWN, (x, y - 1)))

            if x == MAX_POS and y == 0:  # right-down
                moves.append(((x, y), MOVE_LEFT, (x - 1, y)))
                moves.append(((x, y), MOVE_UP, (x, y + 1)))

            return moves

        transitions = []
        for x in range(0, self._size):
            for y in range(0, self._size):
                transitions += _handle_state((x, y))

        return transitions

    def _state_action(self):
        """
        Return states and possible actions in each of them
        """

        # Assign all actions for all states (mapping)
        m = {}
        for x in range(0, self._size):
            for y in range(0, self._size):
                m[(x, y)] = [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN]

        # Remove actions from certain states
        top_row = dict(filter(lambda i: i[0][0] == self._size, m.items()))
        bottom_row = dict(filter(lambda i: i[0][0] == 1, m.items()))
        left_col = dict(filter(lambda i: i[0][1] == 1, m.items()))
        right_col = dict(filter(lambda i: i[0][1] == self._size, m.items()))

        for actions in top_row.values():
            actions.remove(MOVE_UP)

        for actions in bottom_row.values():
            actions.remove(MOVE_DOWN)

        for actions in left_col.values():
            actions.remove(MOVE_LEFT)

        for actions in right_col.values():
            actions.remove(MOVE_RIGHT)

        # No actions possible when found reward
        m[(self._size - 1, self._size - 1)] = []

        # Cast (int, int) key to (str, str)
        m = {(str(k[0]), str(k[1])): v for k, v in m.items()}

        return m
