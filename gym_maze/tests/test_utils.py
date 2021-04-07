import gym
import gym_maze  # noqa: F401


class TestUtils:
    def test_should_calculate_transitions(self):
        # given
        maze = gym.make("Maze4-v0")

        # when
        transitions = maze.env.get_transitions()

        # then
        assert len(transitions) == 115
