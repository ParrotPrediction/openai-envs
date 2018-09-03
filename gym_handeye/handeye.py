import sys

import gym
import logging

ACTION_LOOKUP = {
    0: 'N',
    1: 'E',
    2: 'S',
    3: 'W',
    4: 'G',
    5: 'R'
}

from gym.spaces import Discrete
import gym_handeye.utils.utils as utils
from gym_handeye.handeye_simulator import HandEyeSimulator


class HandEye(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size, note_in_hand, test_only_changes):
        """

        :param grid_size: specifies the size of the monitored plain
        :param note_in_hand: specifies if the tacticle sensor should switch to '2' if the block is held by the gripper
        (if False, then goes back to '0')
        :param test_only_changes: specifies if only condition-action combinations should be tested that invoke a change
        """
        logging.debug('Starting environment HandEye')
        self.grid_size = grid_size
        self.note_in_hand = note_in_hand
        self.test_only_changes = test_only_changes

        self.handeye = HandEyeSimulator(grid_size, note_in_hand, test_only_changes)

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
                if item not in ['w', 'g', 'b']:
                    break
                outfile.write(self._render_element(item))
                j += 1
                if j >= self.grid_size:
                    outfile.write("\n")
                    j = 0
        else:
            super(HandEye, self).render(mode=mode)

    def close(self):
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
        return list(range(0, len(ACTION_LOOKUP)))

    def _should_end_testing(self, previous, obs):
        return (self.test_only_changes and self._no_change_detected(previous, obs)) or (
                (not self.test_only_changes) and self._change_detected(previous, obs))

    def _no_change_detected(self, previous, current):
        return previous == current

    def _change_detected(self, previous, current):
        return not self._no_change_detected(previous, current)

    def get_all_possible_transitions(self):
        """
        Returns all possible transitions of environment
        This information is used to calculate the agent's knowledge
        :param self
        :return: all transitions
        """

        return utils.get_all_possible_transitions(self.grid_size)

    def _observe(self):
        return self.handeye.observe()

    def _take_action(self, action):
        return self.handeye.take_action(action)

    def get_goal_state(self):
        return self.handeye.get_goal_state()

    @staticmethod
    def _render_element(el):
        if el == 'b':
            return gym.utils.colorize('■', 'blue')
        elif el == 'w':
            return gym.utils.colorize('□', 'white')
        elif el == 'g':
            return gym.utils.colorize('#', 'green')
