import gym
import logging

from gym.spaces import Discrete
import random

import gym_handeye.utils.utils as utils

ACTION_LOOKUP = {
    0: 'N',
    1: 'E',
    2: 'S',
    3: 'W',
    4: 'G',
    5: 'R'
}


class HandEye(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size, note_in_hand, test_only_changes):
        logging.debug('Starting environment HandEye')
        self.grid_size = grid_size
        self.note_in_hand = note_in_hand
        self.test_only_changes = test_only_changes

        self.grip_pos_x = None
        self.grip_pos_y = None
        self.block_in_hand = False
        self.block_pos_x = None
        self.block_pos_y = None

        self.env_size = self.grid_size * self.grid_size + 1

        self.observation_space = Discrete(self.env_size)  # camera monitors the whole grid
        self.action_space = Discrete(6)  # N - north, S - south, W - west, E - east, G - grip, R - release

        self.goal_generator_state = 0

        self.observation = ['w' for x in range(self.env_size - 1)]  # array with what we observe
        self.observation.append('0')

        self.reset()

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        :param action: an action provided by the environment
        :return: observation (tuple): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        logging.debug('Executing a step, action = {}'.format(action))
        previous_observation = self._observe()
        self._take_action(action, previous_observation)

        observation = self._observe()
        reward = 0
        episode_over = False
        if (self.test_only_changes and previous_observation == observation) or (
                not self.test_only_changes and previous_observation != observation):
            episode_over = True

        return tuple(observation), reward, episode_over, {}

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
        :return: observation (object): the initial observation of the
            space.
        """
        logging.debug('Resetting the environment')

        self._random_positions()

        return tuple(self.observation)

    def render(self, mode='human', close=False):
        """
        Renders the environment.
        :param mode (str): the mode to render with
        :param close (bool): close all open renderings
        :return:
        """
        if close:
            return

        logging.debug('Rendering the environment')

        if mode == 'human':
            return self._observe()
        else:
            super(HandEye, self).render(mode=mode)

    def close(self):
        return

    def seed(self, seed=None):
        gym.logger.warn("Could not seed environment %s", self)
        return

    @property
    def unwrapped(self):
        """
        Completely unwrap this env.
        :return:  gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def _move_gripper(self, x_end, y_end):
        """
        Executes a moving action with all involved consequences.
        :param x_end: End gripper x position
        :param y_end: End gripper y position
        :return:
        """
        x_start = self.grip_pos_x
        y_start = self.grip_pos_y

        self.grip_pos_x = x_end
        self.grip_pos_y = y_end

        if self.block_in_hand:
            self.block_pos_x = x_end
            self.block_pos_y = y_end
            self.observation[y_start * self.grid_size + x_start] = 'w'
            self.observation[y_end * self.grid_size + x_end] = 'b'
        else:
            self.observation[y_start * self.grid_size + x_start] = 'w'
            self.observation[y_end * self.grid_size + x_end] = 'g'

        if self.block_pos_x == x_start and self.block_pos_y == y_start:
            self.observation[y_start * self.grid_size + x_start] = 'b'

        self.observation[self.env_size - 1] = '0'

        if self.block_pos_x == x_end and self.block_pos_y == y_end:
            if not self.block_in_hand:
                self.observation[self.env_size - 1] = '1'
            else:
                if self.note_in_hand:
                    self.observation[self.env_size - 1] = '2'

        return

    def _grip_block(self):
        """
        Executes a gripping action.
        :return:
        """
        if self.block_in_hand:
            return

        if self.block_pos_x == self.grip_pos_x and self.block_pos_y == self.grip_pos_y:
            self.block_in_hand = True
            self.observation[self.grip_pos_y * self.grid_size + self.grip_pos_x] = 'b'
            if self.note_in_hand:
                self.observation[self.env_size - 1] = '2'
            else:
                self.observation[self.env_size - 1] = '0'

        return

    def _release_block(self):
        """
        Releases a block if a block is held in the hand.
        :return:
        """
        if self.block_in_hand:
            self.observation[self.env_size - 1] = '1'
            self.observation[self.grip_pos_y * self.grid_size + self.grip_pos_x] = 'g'
            self.block_in_hand = False
        return

    def _observe(self):
        return self.observation

    def _take_action(self, action, observation):
        action_type = ACTION_LOOKUP[action]

        if action_type == "N" and self.grip_pos_y > 0:
            self._move_gripper(self.grip_pos_x, self.grip_pos_y - 1)

        elif action_type == "E" and self.grip_pos_x < self.grid_size - 1:
            self._move_gripper(self.grip_pos_x + 1, self.grip_pos_y)

        elif action_type == "S" and self.grip_pos_y < self.grid_size - 1:
            self._move_gripper(self.grip_pos_x, self.grip_pos_y + 1)

        elif action_type == "W" and self.grip_pos_x > 0:
            self._move_gripper(self.grip_pos_x - 1, self.grip_pos_y)

        elif action_type == "G":
            self._grip_block()

        elif action_type == "R":
            self._release_block()

        return

    def _random_positions(self):
        for i, element in enumerate(self.observation):
            self.observation[i] = 'w'
        self.observation[self.env_size - 1] = '0'

        # random block position
        self.block_pos_x = random.randint(0, self.grid_size - 1)
        self.block_pos_y = random.randint(0, self.grid_size - 1)

        self.observation[self.block_pos_y * self.grid_size + self.block_pos_x] = 'b'

        # random gripper position
        if random.choice([True, False]):
            # block in hand
            self.block_in_hand = True

            self.grip_pos_x = self.block_pos_x
            self.grip_pos_y = self.block_pos_y

            if self.note_in_hand:
                self.observation[self.env_size - 1] = '2'  # observation
        else:
            # block not in hand
            self.block_in_hand = False

            self.grip_pos_x = random.randint(0, self.grid_size - 1)
            self.grip_pos_y = random.randint(0, self.grid_size - 1)

            self.observation[self.grip_pos_y * self.grid_size + self.grip_pos_x] = 'g'

            if self.block_pos_x == self.grip_pos_x and self.block_pos_y == self.grip_pos_y:  # is above block
                self.observation[self.env_size - 1] = '1'

        return self.observation

    def get_goal_state(self, perception):
        if self.goal_generator_state == 5:
            self.goal_generator_state = 0
            return

        goal_state = ['w' for x in range(self.env_size - 1)]  # array with what we observe
        goal_state.append('0')

        if self.block_in_hand:
            if self.goal_generator_state == 2:
                while True:
                    x = random.randint(0, self.grid_size - 1)
                    y = random.randint(0, self.grid_size - 1)
                    if x == self.grip_pos_x or y == self.grip_pos_y:
                        break
                goal_state[self.grip_pos_y * self.grid_size + self.grip_pos_x] = 'w'
                goal_state[y * self.grid_size + x] = 'b'
                self.goal_generator_state = 3
            else:
                goal_state[self.grip_pos_y * self.grid_size + self.grip_pos_x] = 'g'
                goal_state[self.env_size - 1] = '1'
                self.goal_generator_state = 4
        else:
            if self.observation[self.env_size - 1] == '1':
                if self.goal_generator_state == 4:
                    while True:
                        x = random.randint(0, self.grid_size - 1)
                        y = random.randint(0, self.grid_size - 1)
                        if x == self.grip_pos_x or y == self.grip_pos_y:
                            break
                    goal_state[self.grip_pos_y * self.grid_size + self.grip_pos_x] = 'b'
                    goal_state[y * self.grid_size + x] = 'g'
                    goal_state[self.env_size - 1] = '0'
                    self.goal_generator_state = 5
                else:
                    if self.note_in_hand:
                        goal_state[self.env_size - 1] = '2'
                    else:
                        goal_state[self.env_size - 1] = '0'
                    goal_state[self.grip_pos_y * self.grid_size + self.grip_pos_x] = 'b'
                    self.goal_generator_state = 2
            else:
                goal_state[self.grip_pos_y * self.grid_size + self.grip_pos_x] = 'w'
                goal_state[self.block_pos_y * self.grid_size + self.block_pos_x] = 'g'
                goal_state[self.env_size - 1] = '1'
                self.goal_generator_state = 1

        # TODO: perception->setPerception(goalState)
        # (will be done after implementing Action Planning in pyalcs)
        return

    @staticmethod
    def get_all_possible_actions():
        return list(range(0, len(ACTION_LOOKUP)))

    def get_all_possible_transitions(self):
        """
        Returns all possible transitions of environment
        This information is used to calculate the agent's knowledge
        :param self
        :return: all transitions
        """

        return utils.get_all_possible_transitions(self.grid_size)
