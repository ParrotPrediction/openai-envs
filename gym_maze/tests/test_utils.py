import gym
# noinspection PyUnresolvedReferences
import gym_maze  # noqa: F401


class TestUtils:
    def test_should_calculate_transitions(self):
        # given
        maze = gym.make("Woods1-v0")

        # when
        transitions = maze.env.get_all_possible_transitions()

        # then
        assert 37 == len(transitions)
