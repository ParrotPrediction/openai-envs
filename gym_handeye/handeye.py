import sys

import gym
import logging

import numpy as np

from gym.spaces import Discrete
import random

ACTION_LOOKUP = {
    0: 'N',
    1: 'E',
    2: 'S',
    3: 'W',
    4: 'G',
    5: 'R'
}


class HandEye(gym.Env):
    """The main OpenAI Gym class. It encapsulates an environment with
        arbitrary behind-the-scenes dynamics. An environment can be
        partially or fully observed.
        The main API methods that users of this class need to know are:
            step
            reset
            render
            close
            seed
        And set the following attributes:
            action_space: The Space object corresponding to valid actions
            observation_space: The Space object corresponding to valid observations
            reward_range: A tuple corresponding to the min and max possible rewards
        Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
        The methods are accessed publicly as "step", "reset", etc.. The
        non-underscored versions are wrapper methods to which we may add
        functionality over time.
        """

    # Set this in SOME subclasses
    metadata = {'render.modes': ['human']}

    # reward_range = (-float('inf'), float('inf'))
    # spec = None

    # Set these in ALL subclasses
    # action_space = None
    # observation_space = None

    def __init__(self, grid_size, note_in_hand, test_only_changes):
        self.grid_size = grid_size
        # grid size is 3 when plain is 3x3

        self.note_in_hand = note_in_hand
        # whether tacticle sensor should switch to 2 if the block is held by the gripper
        # (if False, then it switches back to 0)

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

        self.env = ['w' for x in range(self.env_size - 1)]  # array with what we observe
        self.env.append('0')

        self.reset()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        previous_observation = self._observe()
        self._take_action(action, previous_observation)

        observation = self._observe()
        reward = 0
        episode_over = False
        if (self.test_only_changes and previous_observation == observation) or (not self.test_only_changes and previous_observation != observation):
            episode_over = True

        return observation, reward, episode_over, {}

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """

        for i, element in enumerate(self.env):
            self.env[i] = 'w'
        self.env[self.env_size - 1] = '0'

        # random block position
        self.block_pos_x = random.randint(0, self.grid_size - 1)
        self.block_pos_y = random.randint(0, self.grid_size - 1)

        self.env[self.block_pos_y * self.grid_size + self.block_pos_x] = 'b'

        # random gripper position
        if (random.choice([True, False])):
            # block in hand
            self.block_in_hand = True

            self.grip_pos_x = self.block_pos_x
            self.grip_pos_y = self.block_pos_y

            if self.note_in_hand:
                self.env[self.env_size - 1] = '2'  # observation
        else:
            # block not in hand
            self.block_in_hand = False

            self.grip_pos_x = random.randint(0, self.grid_size - 1)
            self.grip_pos_y = random.randint(0, self.grid_size - 1)

            self.env[self.grip_pos_y * self.grid_size + self.grip_pos_x] = 'g'

            if self.block_pos_x == self.grip_pos_x and self.block_pos_y == self.grip_pos_y:  # is above block
                self.env[self.env_size - 1] = '1'

        # TODO: return observation (?)
        return self.env

    def render(self, mode='human', close = False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
                    :param **kwargs:
        """
        if close:
            return

        if mode == 'human':
            return self._observe()
        else: super(HandEye, self).render(mode=mode)

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        gym.logger.warn("Could not seed environment %s", self)
        return

    @property
    def unwrapped(self):
        """Completely unwrap this env.
        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def _move_gripper(self, x_start, y_start, x_end, y_end):
        """
        Executes a moving action with all involved consequences.
        :param x_start:
        :param y_start:
        :param x_end:
        :param y_end:
        :return:
        """
        if self.block_in_hand:
            self.block_pos_x = x_end
            self.block_pos_y = y_end
            self.env[y_start * self.grid_size + x_start] = 'w'
            self.env[y_end * self.grid_size + x_end] = 'b'
        else:
            self.env[y_start * self.grid_size + x_start] = 'w'
            self.env[y_end * self.grid_size + x_end] = 'g'

        if self.block_pos_x == x_start and self.block_pos_y == y_start:
            self.env[y_start * self.grid_size + x_start] = 'b'

        self.env[self.env_size - 1] = '0'

        if self.block_pos_x == x_end and self.block_pos_y == y_end:
            if not self.block_in_hand:
                self.env[self.env_size - 1] = '1'
            else:
                if self.note_in_hand:
                    self.env[self.env_size - 1] = '2'

        return

    def _grip_block(self, x, y):
        """
        Executes a gripping action.
        :param x:
        :param y:
        :return:
        """
        if self.block_in_hand:
            return

        if self.block_pos_x == x and self.block_pos_y == y:
            self.block_in_hand = True
            self.env[y * self.grid_size + x] = 'b'
            if self.note_in_hand:
                self.env[self.env_size - 1] = '2'
            else:
                self.env[self.env_size - 1] = '0'

        return

    def _release_block(self, x, y):
        """
        Releases a block if a block is held in the hand.
        :param x:
        :param y:
        :return:
        """
        if self.block_in_hand:
            self.env[self.env_size - 1] = '1'
            self.env[y * self.grid_size + x] = 'g'
            self.block_in_hand = False
        return

    def _observe(self):
        return self.env

    def _take_action(self, action, observation):
        action_type = ACTION_LOOKUP[action]

        if action_type == "N" and self.grip_pos_y > 0:
            self._move_gripper(self.grip_pos_x, self.grip_pos_y, self.grip_pos_x, self.grip_pos_y - 1)
            self.grip_pos_y = self.grip_pos_y - 1

        if action_type == "E" and self.grip_pos_x < self.grid_size - 1:
            self._move_gripper(self.grip_pos_x, self.grip_pos_y, self.grip_pos_x + 1, self.grip_pos_y)
            self.grip_pos_x = self.grip_pos_x + 1

        if action_type == "S" and self.grip_pos_y < self.grid_size - 1:
            self._move_gripper(self.grip_pos_x, self.grip_pos_y, self.grip_pos_x, self.grip_pos_y + 1)
            self.grip_pos_y = self.grip_pos_y + 1

        if action_type == "W" and self.grip_pos_x > 0:
            self._move_gripper(self.grip_pos_x, self.grip_pos_y, self.grip_pos_x - 1, self.grip_pos_y)
            self.grip_pos_x = self.grip_pos_x - 1

        if action_type == "G":
            self._grip_block(self.grip_pos_x, self.grip_pos_y)

        if action_type == "R":
            self._release_block(self.grip_pos_x, self.grip_pos_y)

        return

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
            if self.env[self.env_size - 1] == '1':
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

        #perception->setPerception(goalState)
        return