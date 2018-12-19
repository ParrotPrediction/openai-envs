import gym
import pytest
# noinspection PyUnresolvedReferences
import gym_maze


class TestTaxiGoal:
    @pytest.mark.parametrize("_taxi_x, _taxi_y, _passloc, _destidx, "
                             "_goal_x, _goal_y, _goal_pass, _goal_dest",
                             [
                                 (0, 0, 1, 2, 0, 4, 1, 2),
                                 (0, 4, 1, 2, 0, 4, 4, 2),
                                 (0, 4, 4, 2, 4, 0, 4, 2),
                                 (4, 0, 4, 2, 4, 0, 2, 2)
                             ])
    def test_should_return_goal_state(self, _taxi_x, _taxi_y, _passloc,
                                      _destidx, _goal_x, _goal_y, _goal_pass,
                                      _goal_dest):
        # given
        taxi = gym.make('TaxiGoal-v0')
        taxi.env.s = taxi.env.encode(_taxi_x, _taxi_y, _passloc, _destidx)

        # when
        goal_state = taxi.env.get_goal_state()

        # then
        assert goal_state == taxi.env.encode(_goal_x, _goal_y, _goal_pass,
                                             _goal_dest)
