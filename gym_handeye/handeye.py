import sys

import gym
import logging

from gym.spaces import Discrete
import gym_handeye.utils.utils as utils
from gym_handeye.handeye_simulator import HandEyeSimulator, SURFACE, BLOCK, GRIPPER, ACTION_LOOKUP


class HandEye(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size, note_in_hand, test_only_changes=0):
        """

        :param grid_size: specifies the size of the monitored plain
        :param note_in_hand: specifies if the tacticle sensor should switch to '2' if the block is held by the gripper
        (if False, then goes back to '0')
        :param test_only_changes: specifies if only condition-action combinations should be tested that invoke
        a change (1), non changes (-1) or all possibilities (0) should be tested
        """
        logging.debug('Starting environment HandEye')
        self.grid_size = grid_size
        self.note_in_hand = note_in_hand
        self.test_only_changes = test_only_changes

        self.handeye = HandEyeSimulator(grid_size, note_in_hand)

        self.observation_space = Discrete(self.handeye.env_size)
        self.action_space = Discrete(len(ACTION_LOOKUP))

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
        self._take_action(action)

        observation = self._observe()
        reward = 0
        episode_over = False
        if self._should_end_testing(previous_observation, observation):
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

        self.handeye.set_random_positions()

        return tuple(self._observe())

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
            outfile = sys.stdout
            outfile.write("\n")

            j = 0
            for item in self.handeye.observation:
                if item not in [SURFACE, GRIPPER, BLOCK]:
                    break
                outfile.write(self._render_element(item))
                j += 1
                if j >= self.grid_size:
                    outfile.write("\n")
                    j = 0
        else:
            super(HandEye, self).render(mode=mode)

    def close(self):
        """
        Closes the environment.
        :return:
        """
        return

    @property
    def unwrapped(self):
        """
        Completely unwrap this env.
        :return:  gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    @staticmethod
    def get_all_possible_actions():
        """
        Returns all possible actions in this environment.
        :return:
        """
        return list(range(0, len(ACTION_LOOKUP)))

    def get_all_possible_transitions(self):
        """
        Returns all possible transitions of the environment
        This information is used to calculate the agent's knowledge
        :param self
        :return: all transitions as list of tuples: (start_state, action, end_state)
        """

        return utils.get_all_possible_transitions(self.grid_size)

    def get_goal_state(self):
        """
        Returns goal_state - an observation that is the environment's next goal.
        Non deterministic.
        :return:
        """
        return self.handeye.get_goal_state()

    def _should_end_testing(self, previous, obs):
        """
        Returns if the test should end based on self.test_only_changes parameter.
        :param previous: previous observation
        :param obs: current observation
        :return:
        """
        return (self.test_only_changes == 1 and not self._change_detected(previous, obs)) or (
                self.test_only_changes == -1 and self._change_detected(previous, obs))

    def _change_detected(self, previous, current):
        """
        Returns true if a change was detected between observations (previous and current).
        :param previous: previous observation
        :param current: current observation
        :return:
        """
        return previous != current

    def _observe(self):
        """
        Returns observation of this environment.
        :return:
        """
        return self.handeye.observe()

    def _take_action(self, action):
        """
        Executes an action with all consequences. Returns true if executing an action was successful.
        :param action:
        :return:
        """
        return self.handeye.take_action(action)

    @staticmethod
    def _render_element(el):
        """
        Renders a single element.
        :param el:
        :return:
        """
        if el == BLOCK:
            return gym.utils.colorize('■', 'blue')
        elif el == SURFACE:
            return gym.utils.colorize('□', 'white')
        elif el == GRIPPER:
            return gym.utils.colorize('#', 'green')
