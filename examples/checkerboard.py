import logging
import gym

# noinspection PyUnresolvedReferences
import gym_checkerboard

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    env = gym.make('checkerboard-2D-5div-v0')

    for trial in range(10):
        state = env.reset()
        logging.info(f"Trial: {trial}, state: {state}")

        action = env.action_space.sample()
        logging.info(f"\tPerforming action: {action}")

        state, reward, done, info = env.step(action)
        logging.info(f"\tObtained reward: {reward}, state: {state}")

    logging.info("Finished")