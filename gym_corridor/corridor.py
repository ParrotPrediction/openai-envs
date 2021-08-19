from random import randint

import gym
from gym.spaces import Discrete

MOVE_LEFT = 0
MOVE_RIGHT = 1


class Corridor(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    REWARD = 1000

    def __init__(self, size=20):
        self._size = size
        self._position = None
        self._transitions = self._calculate_transitions()

        self.observation_space = Discrete(size)
        self.action_space = Discrete(2)

    def reset(self):
        self._position = randint(0, self._size - 2)
        return str(self._position)

    def step(self, action):
        if action == MOVE_LEFT:
            self._position -= 1
        elif action == MOVE_RIGHT:
            self._position += 1
        else:
            raise ValueError("Illegal action passed")

        if self._position == self._size - 1:
            return str(self._position), self.REWARD, True, {}

        if self._position == -1:
            self._position = 0

        return str(self._position), 0, False, {}

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
        corridor = ["" for _ in range(0, self._size - 1)]
        corridor[self._position - 1] = "X"
        corridor[self._size - 2] = "$"
        return "[" + ".".join(corridor) + "]"

    def _calculate_transitions(self):
        START, END = 0, self._size - 1
        LEFT, RIGHT = 0, 1

        def _handle_state(state):
            moves = []
            if state == START:
                moves.append((state, RIGHT, state + 1))
            else:
                moves.append((state, LEFT, state - 1))
                moves.append((state, RIGHT, state + 1))

            return moves

        transitions = []

        for state in range(START, END):
            transitions += _handle_state(state)

        return transitions

    def _state_action(self):
        """
        Return states and possible actions in each of them
        """
        mapping = {}
        for p in range(0, self._size):
            mapping[p] = [MOVE_LEFT, MOVE_RIGHT]

        # Corner cases
        # mapping[0] = [MOVE_RIGHT]
        mapping[self._size - 1] = []

        # Cast int key str
        mapping = {str(k): v for k, v in mapping.items()}

        return mapping
