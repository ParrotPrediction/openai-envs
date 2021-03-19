import random

import gym
import numpy as np

from gym_maze.common import MAZE_ANIMAT, MAZE_WALL, MAZE_PATH, MAZE_REWARD


class MazeObservationSpace(gym.Space):
    def __init__(self, n):
        # n is the number of visible neighbour fields, typically 8
        self.np_random = np.random.RandomState()
        self.n = n
        gym.Space.__init__(self, (self.n,), str)

    def seed(self, seed):
        self.np_random.seed(seed)

    def sample(self):
        states = map(str, [MAZE_PATH, MAZE_WALL, MAZE_REWARD])
        return tuple(random.choice(list(states)) for _ in range(self.n))

    def contains(self, x):
        states = map(str, [MAZE_PATH, MAZE_WALL, MAZE_ANIMAT, MAZE_REWARD])
        return all(elem in states for elem in x)

    def to_jsonable(self, sample_n):
        return list(sample_n)

    def from_jsonable(self, sample_n):
        return tuple(sample_n)
