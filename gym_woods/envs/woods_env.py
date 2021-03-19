import logging
import random
import sys

import gym
import numpy as np
from gym import spaces, utils

from gym_maze.internal.maze_impl import ACTION_LOOKUP, find_action_by_direction
from gym_woods.woods import Woods


class WoodsObservationSpace(gym.Space):
    """

    Mapping:
    . - path
    * - agent
    O, Q - wall
    F, G - reward
    """

    SYMBOLS = ('.', '*', 'O', 'Q', 'F', 'G')

    def seed(self, seed):
        pass

    def __init__(self, n):
        # n is the number of visible neighbour fields, typically 8
        self.n = n
        gym.Space.__init__(self, (self.n,), str)

    def sample(self):
        return tuple(random.choice(self.SYMBOLS) for _ in range(self.n))

    def contains(self, x):
        return all(elem in self.SYMBOLS for elem in x)


class AbstractWoods(gym.Env):

    def __init__(self, matrix):
        self.maze = Woods(matrix)
        self.pos_x = None
        self.pos_y = None

        self.action_space = spaces.Discrete(8)
        self.observation_space = WoodsObservationSpace(8)

    def reset(self):
        logging.debug('Resetting the environment')
        self._insert_animat()
        return self._observe()

    def step(self, action):
        previous_observation = self._observe()
        self._take_action(action, previous_observation)

        observation = self._observe()
        reward = self._get_reward()
        episode_over = self._is_over()

        return observation, reward, episode_over, {}

    def render(self, mode='human'):
        if mode == 'human':
            snapshot = np.copy(self.maze.matrix)
            snapshot[self.pos_y, self.pos_x] = 'X'

            sys.stdout.write("\n")
            for row in snapshot:
                sys.stdout.write(" ".join(self._render(el) for el in row))
                sys.stdout.write("\n")
            sys.stdout.flush()

        else:
            super(AbstractWoods, self).render(mode=mode)

    def _take_action(self, action, observation):
        """Executes the action inside the maze"""
        animat_moved = False
        action_type = ACTION_LOOKUP[action]

        if action_type == "N" and not self.is_wall(observation[0]):
            self.pos_y -= 1
            animat_moved = True

            if self.pos_y < 0:
                self.pos_y = self.maze.max_y - 1

        if action_type == 'NE' and not self.is_wall(observation[1]):
            self.pos_x += 1
            self.pos_y -= 1
            animat_moved = True

            if self.pos_y < 0:
                self.pos_y = self.maze.max_y - 1

            if self.pos_x >= self.maze.max_x:
                self.pos_x = 0

        if action_type == "E" and not self.is_wall(observation[2]):
            self.pos_x += 1
            animat_moved = True

            if self.pos_x >= self.maze.max_x:
                self.pos_x = 0

        if action_type == 'SE' and not self.is_wall(observation[3]):
            self.pos_x += 1
            self.pos_y += 1
            animat_moved = True

            if self.pos_x >= self.maze.max_x:
                self.pos_x = 0

            if self.pos_y >= self.maze.max_y:
                self.pos_y = 0

        if action_type == "S" and not self.is_wall(observation[4]):
            self.pos_y += 1
            animat_moved = True

            if self.pos_y >= self.maze.max_y:
                self.pos_y = 0

        if action_type == 'SW' and not self.is_wall(observation[5]):
            self.pos_x -= 1
            self.pos_y += 1
            animat_moved = True

            if self.pos_x < 0:
                self.pos_x = self.maze.max_x - 1

            if self.pos_y >= self.maze.max_y:
                self.pos_y = 0

        if action_type == "W" and not self.is_wall(observation[6]):
            self.pos_x -= 1
            animat_moved = True

            if self.pos_x < 0:
                self.pos_x = self.maze.max_x - 1

        if action_type == 'NW' and not self.is_wall(observation[7]):
            self.pos_x -= 1
            self.pos_y -= 1
            animat_moved = True

            if self.pos_x < 0:
                self.pos_x = self.maze.max_x - 1

            if self.pos_y < 0:
                self.pos_y = self.maze.max_y - 1

        return animat_moved

    def _insert_animat(self):
        possible_coords = self.maze.possible_insertion_cords

        starting_position = random.choice(possible_coords)
        self.pos_x = starting_position[0]
        self.pos_y = starting_position[1]

    def _observe(self):
        return self.maze.perception(self.pos_x, self.pos_y)

    def _perception(self, posx, posy):
        return self.maze.perception(posx, posy)

    def _get_reward(self):
        if self.maze.is_reward(self.pos_x, self.pos_y):
            return 1000

        return 0

    def _is_over(self):
        return self.maze.is_reward(self.pos_x, self.pos_y)

    @staticmethod
    def is_wall(obs):
        return obs in ['O', 'Q']

    @staticmethod
    def _render(el):
        if el in ('O', 'Q'):
            return utils.colorize('■', 'gray')
        elif el == '.':
            return utils.colorize('□', 'white')
        elif el in ('F', 'G'):
            return utils.colorize('$', 'yellow')
        elif el == '*':
            return utils.colorize('A', 'red')
        else:
            return utils.colorize(el, 'cyan')

    def _state_action(self):
        """
        Return states and possible actions in each of them
        """
        mapping = {}

        for x, y in self.maze.possible_insertion_cords:
            [n, ne, e, se, s, sw, w, nw] = self.maze.perception(x, y)
            key = (x, y)
            mapping[key] = []

            actions_perceptions = {
                'N': n,
                'NE': ne,
                'E': e,
                'SE': se,
                'S': s,
                'SW': sw,
                'W': w,
                'NW': nw
            }

            for action, perception in actions_perceptions.items():
                if not self.is_wall(perception):
                    mapping[key].append(find_action_by_direction(action))

        # Cast (int, int) key to (str, str)
        mapping = {(str(k[0]), str(k[1])): v for k, v in mapping.items()}

        return mapping
