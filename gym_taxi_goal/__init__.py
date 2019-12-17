from gym.envs.registration import register

from gym_taxi_goal.taxi_goal import TaxiGoalEnv  # noqa: F401

register(
    id='TaxiGoal-v0',
    entry_point='gym_taxi_goal:TaxiGoalEnv',
    max_episode_steps=200
)
