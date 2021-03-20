import io
import logging
import sys

import numpy as np

import gym
from gym import spaces

from gym_maze.common.maze_observation_space import MazeObservationSpace
from gym_maze.common.maze_renderer import render
from gym_maze.internal.rotating_maze_impl import RotatingMazeImpl
from gym_maze.utils.rotating_utils import get_all_transitions


class RotatingMaze(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, matrix):
        self.matrix = np.copy(matrix)
        self.maze = RotatingMazeImpl(np.copy(self.matrix))
        self.action_space = spaces.Discrete(3)
        self.observation_space = MazeObservationSpace(8)

    def reset(self):
        logging.debug("Resetting the environment")
        self.maze = RotatingMazeImpl(np.copy(self.matrix))
        self.maze.insert_agent()
        return self._observe()

    def step(self, action: int):
        assert action in [0, 1, 2]

        if action == 0:
            self.maze.step_ahead()
        elif action == 1:
            self.maze.turn_left()
        elif action == 2:
            self.maze.turn_right()

        return self._observe(), self._get_reward(), self._is_over(), {}

    def render(self, mode='human'):
        if mode == 'human':
            render(sys.stdout, self.maze.matrix)
        elif mode == 'ansi':
            output = io.StringIO()
            render(output, self.maze.matrix)
            return output.getvalue()

    def _observe(self):
        return self.maze.perception()

    def _get_reward(self):
        if self._is_over():
            return 1000

        return 0

    def _is_over(self):
        return self.maze.is_done()

    def get_all_possible_transitions(self):
        """Debugging only"""
        return get_all_transitions(self.matrix)
