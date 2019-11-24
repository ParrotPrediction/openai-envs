import numpy as np
from gym.envs.classic_control.mountain_car import MountainCarEnv


class EnergyMountainCar(MountainCarEnv):

    def step(self, action):
        state, reward, done, info = super().step(action)

        (position, velocity) = state

        # Height is approximated from
        # https://repl.it/@khozzy/MountainCar-height
        height = 0.5 * (np.cos(2.6 * position - 1.6) + 1)

        # Represent reward as the energy
        energy = 9.81 * height + velocity**2 / 2

        return state, energy, done, info
