import pytest
import gym

# noinspection PyUnresolvedReferences
import gym_mountain_car


class TestEnergyMountainCar:

    def test_should_initialize(self):
        env = gym.make('EnergyMountainCar-v0')
        assert env is not None

    def test_should_make_move(self):
        env = gym.make('EnergyMountainCar-v0')
        env.seed(1)

        # initial conditions
        (position, velocity) = env.reset()
        assert position != 0
        assert velocity == 0

        # push right
        (position, velocity), reward, done, info = env.step(2)
        assert position != 0
        assert velocity != 0
