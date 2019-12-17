import gym
import pytest
# noinspection PyUnresolvedReferences
import gym_taxi_goal  # noqa: F401


class TestTaxiGoal:
    @pytest.mark.parametrize("_taxi_x, _taxi_y, _passloc, _destidx, "
                             "_goal_x, _goal_y, _goal_pass, _goal_dest",
                             [
                                 (0, 0, 1, 2, 0, 4, 1, 2),
                                 (0, 4, 1, 2, 0, 4, 4, 2),
                                 (0, 4, 4, 2, 4, 0, 4, 2),
                                 (4, 0, 4, 2, 4, 0, 2, 2),

                                 (0, 0, 3, 0, 4, 3, 3, 0),
                                 (4, 3, 3, 0, 4, 3, 4, 0),
                                 (4, 3, 4, 0, 0, 0, 4, 0),
                                 (0, 0, 4, 0, 0, 0, 0, 0),

                                 (0, 0, 3, 3, 4, 3, 3, 3),
                                 (4, 3, 3, 3, 4, 3, 4, 3),
                                 (4, 3, 4, 3, 4, 3, 3, 3)
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
        if _goal_x < 0 or _goal_y < 0 or _goal_pass < 0 or _goal_dest < 0:
            _goal_state = None
        else:
            _goal_state = taxi.env.encode(_goal_x, _goal_y, _goal_pass,
                                          _goal_dest)
        assert goal_state == _goal_state
