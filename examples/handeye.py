import logging
from random import choice

import gym

# noinspection PyUnresolvedReferences
import gym_handeye

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    hand_eye = gym.make('HandEye3-v0')

    possible_actions = list(range(6))

    for i_episode in range(1):
        observation = hand_eye.reset()

        for t in range(100):
            logging.info("Time: [{}], observation: [{}]".format(t, observation))

            action = choice(possible_actions)

            logging.info("\t\tExecuted action: [{}]".format(action))
            observation, reward, done, info = hand_eye.step(action)

            if done:
                logging.info("Episode finished after {} timesteps.".format(t + 1))
                logging.info("Last reward: {}".format(reward))
                break

    logging.info("Finished")