from gym_mountain_car.EnergyMountainCar import EnergyMountainCar
from gym.envs.registration import register

register(
    id='EnergyMountainCar-v0',
    entry_point='gym_mountain_car:EnergyMountainCar',
    max_episode_steps=200,
    reward_threshold=-110.0,
)
