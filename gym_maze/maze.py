import io
import logging
import sys

import gym
import numpy as np
from gym import spaces

from gym_maze.common.maze_observation_space import MazeObservationSpace
from gym_maze.common.maze_renderer import render
from gym_maze.internal.maze_impl import MazeImpl
from gym_maze.utils import get_all_possible_transitions


class Maze(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, matrix):
        self.matrix = np.copy(matrix)
        self.maze = MazeImpl(np.copy(matrix))

        self.action_space = spaces.Discrete(8)
        self.observation_space = MazeObservationSpace(8)
        self._transitions = self._calculate_transitions()

    def reset(self):
        logging.debug("Resetting the environment")
        self.maze = MazeImpl(np.copy(self.matrix))
        self.maze.insert_agent()
        return self._observe()

    def step(self, action: int):
        self.maze.move(action)
        return self._observe(), self._get_reward(), self._is_over(), {}

    def render(self, mode='human'):
        if mode == 'human':
            render(sys.stdout, self.maze.matrix)
        elif mode == 'ansi':
            output = io.StringIO()
            render(output, self.maze.matrix)
            return output.getvalue()
        else:
            super(Maze, self).render(mode=mode)

    def _observe(self):
        return self.maze.perception()

    def _get_reward(self):
        if self._is_over():
            return 1000

        return 0

    def _is_over(self):
        return self.maze.is_done()

    def get_transitions(self):
        return self._transitions

    def get_goal_state(self):
        return self.maze.get_goal_state()

    def _calculate_transitions(self):
        return get_all_possible_transitions(self)
