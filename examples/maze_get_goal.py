import logging
from random import choice

import gym
import sys
sys.path.insert(0, '/home/e-dzia/openai-envs')
# noinspection PyUnresolvedReferences
import gym_maze

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    maze = gym.make('MazeF1-v0')

    goal = maze.maze.get_goal_state()

    logging.info("Finished")
