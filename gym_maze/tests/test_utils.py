import gym
# noinspection PyUnresolvedReferences
import gym_maze  # noqa: F401


class TestUtils:
    def test_should_calculate_transitions(self):
        # given
        maze = gym.make("Maze4-v0")

        # when
        transitions = maze.env.get_all_possible_transitions()

        # then
        assert 115 == len(transitions)
