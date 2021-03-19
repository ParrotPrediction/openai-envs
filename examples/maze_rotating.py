import logging

import gym
import gym_maze  # noqa: F401

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    maze = gym.make('Maze288-v0')
    maze.render()
    # for i_episode in range(1):
    #     observation = maze.reset()
    #
    #     for t in range(100):
    #         logging.info(f"Time: [{t}], observation: [{observation}]")
    #
    #         action = maze.action_space.sample()
    #
    #         logging.info("\t\tExecuted action: [{}]".format(action))
    #
    #         observation, reward, done, info = maze.step(action)
    #
    #         if done:
    #             logging.info(f"Episode finished after {t+1} steps.")
    #             logging.info(f"Last reward: {reward}")
    #             break

    logging.info("Finished")
