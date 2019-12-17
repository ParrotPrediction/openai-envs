import gym
import pytest
# noinspection PyUnresolvedReferences
import gym_maze  # noqa: F401


class TestMaze:
    @pytest.mark.parametrize("_env_name, _x, _y", [
        ('Maze4-v0', 6, 1),
        ('Maze5-v0', 7, 1),
        ('Maze6-v0', 7, 1),
        ('MazeF1-v0', 2, 1)
    ])
    def test_should_return_reward_state(self, _env_name, _x, _y):
        # given
        maze = gym.make(_env_name)

        # when
        x = maze.env.maze._goal_x
        y = maze.env.maze._goal_y

        # then
        assert x == _x
        assert y == _y

    @pytest.mark.parametrize("_env_name, _current_x, _current_y, _goal_state",
                             [
                                 ('Maze4-v0', 6, 1, None),
                                 ('Maze4-v0', 5, 1, tuple('11110001')),
                                 ('Maze4-v0', 5, 2, tuple('11110001')),
                                 ('Maze4-v0', 6, 2, tuple('11110001')),
                                 ('Maze5-v0', 7, 1, None),
                                 ('Maze5-v0', 6, 1, tuple('11110101')),
                                 ('Maze5-v0', 7, 2, tuple('11110101')),
                                 ('MazeF1-v0', 2, 1, None),
                                 ('MazeF1-v0', 1, 1, tuple('11111001'))
                             ])
    def test_should_return_goal_state(self, _env_name, _current_x, _current_y,
                                      _goal_state):
        # given
        maze = gym.make(_env_name)

        # when
        goal_state = maze.env.maze.get_goal_state(_current_x, _current_y)

        # then
        assert goal_state == _goal_state
