from collections import namedtuple
from enum import unique, IntEnum

import gym
from gym.spaces import Discrete


@unique
class Action(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


ActionState = namedtuple('ActionState', "action state")


class SimpleMaze(gym.Env):
    REWARD = 1
    TRANSITIONS = {
        0: [ActionState(Action.EAST, 1), ActionState(Action.SOUTH, 3)],
        1: [ActionState(Action.WEST, 0), ActionState(Action.EAST, 2)],
        2: [ActionState(Action.WEST, 1), ActionState(Action.SOUTH, 5)],
        3: [ActionState(Action.NORTH, 0)],
        4: [ActionState(Action.SOUTH, 7)],
        5: [ActionState(Action.NORTH, 2), ActionState(Action.SOUTH, 8)],
        6: [],
        7: [ActionState(Action.NORTH, 4), ActionState(Action.EAST, 8), ActionState(Action.WEST, 6)],
        8: [ActionState(Action.NORTH, 5), ActionState(Action.WEST, 7)]
    }
    # Wall 1, Path 0
    PERCEPTIONS = {
        0: [1, 0, 0, 1],
        1: [1, 0, 1, 0],
        2: [1, 1, 0, 0],
        3: [0, 1, 1, 1],
        4: [1, 1, 0, 1],
        5: [0, 1, 0, 1],
        6: [1, 0, 1, 1],
        7: [0, 0, 1, 0],
        8: [0, 1, 1, 0]
    }

    def __init__(self):
        self._position = None
        self.observation_space = Discrete(4)
        self.action_space = Discrete(len(Action))

    def reset(self):
        self._position = 3
        return self._perception()

    def step(self, action):
        assert action in list(map(int, Action))

        for transition in self.TRANSITIONS[self._position]:
            if transition.action == action:
                self._position = transition.state

        if self._position == 6:
            return self._perception(), self.REWARD, True, None

        return self._perception(), 0, False, None

    def _perception(self):
        return list(map(str, self.PERCEPTIONS[self._position]))

    def render(self, mode='human'):
        return f"State: {self._position}"
