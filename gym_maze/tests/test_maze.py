import gym
# noinspection PyUnresolvedReferences
import gym_maze


class TestMaze:
    def test_should_(self):
        # given
        maze = gym.make("Woods1-v0")

        # when
        transitions = maze.env.get_all_possible_transitions()

        # then
        assert 37 == len(transitions)
