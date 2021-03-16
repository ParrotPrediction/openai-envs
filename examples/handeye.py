import logging

import gym
import gym_handeye  # noqa: F401

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    hand_eye = gym.make('HandEye3-v0')

    for i_episode in range(1):
        observation = hand_eye.reset()

        for t in range(100):
            logging.info(f"Time: [{t}], observation: [{observation}]")

            action = hand_eye.action_space.sample()

            logging.info(f"\t\tExecuted action: [{action}]")
            observation, reward, done, info = hand_eye.step(action)

            if done:
                logging.info(f"Episode finished after {t+1} timesteps.")
                logging.info(f"Last reward: {reward}")
                break

    logging.info("Finished")
