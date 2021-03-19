import gym
from gym import spaces

from gym_maze.common.maze_observation_space import MazeObservationSpace


class RotatingMaze(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, matrix):
        self.matrix = matrix

        self.action_space = spaces.Discrete(8)
        self.observation_space = MazeObservationSpace(8)

    def step(self, action: int):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass
