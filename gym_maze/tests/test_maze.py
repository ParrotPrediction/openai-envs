import gym
import gym_maze  # noqa: F401
import numpy as np
import pytest
from gym_maze.common import MAZE_ANIMAT


class TestMaze:
    @pytest.mark.parametrize("_env_name, _goal", [
        ('Maze4-v0', (1, 6)),
        ('Maze5-v0', (1, 7)),
        ('Maze6-v0', (1, 7)),
        ('MazeF1-v0', (1, 2))
    ])
    def test_should_return_reward_state(self, _env_name, _goal):
        maze = gym.make(_env_name)
        assert maze.env.maze._goal == _goal

    @pytest.mark.parametrize("_env_name, _xy, _goal_state", [
        ('Maze4-v0', (1, 5), list('11110001')),
        ('Maze4-v0', (2, 5), list('11110001')),
        ('Maze4-v0', (2, 6), list('11110001')),
        ('Maze5-v0', (1, 6), list('11110101')),
        ('Maze5-v0', (2, 7), list('11110101')),
        ('MazeF1-v0', (1, 1), list('11111001'))
    ])
    def test_should_return_goal_state(self, _env_name, _xy, _goal_state):
        maze = gym.make(_env_name)
        maze.env.maze.insert_agent(_xy)

        assert maze.env.maze.get_goal_state() == _goal_state

    def test_should_reset_the_environment(self):
        # given
        env = gym.make('Maze4-v0')

        assert env.env.maze.matrix is not None
        assert np.sum(np.where(env.env.maze.matrix == MAZE_ANIMAT, 1, 0)) == 0

        # when & then
        env.reset()
        assert np.sum(np.where(env.env.maze.matrix == MAZE_ANIMAT, 1, 0)) == 1

        env.reset()
        assert np.sum(np.where(env.env.maze.matrix == MAZE_ANIMAT, 1, 0)) == 1

    def test_should_get_correct_transitions(self):
        env = gym.make('MazeF1-v0')

        transitions = {}
        for start, direction, end in env.env.get_transitions():
            if start not in transitions:
                transitions[start] = {}

            transitions[start][direction] = end

        assert len(transitions) == 5
        assert transitions[(1, 1)] == {2: (1, 2), 4: (2, 1)}
        assert transitions[(2, 1)] == {0: (1, 1), 4: (3, 1), 3: (3, 2),
                                       1: (1, 2)}
        assert transitions[(3, 1)] == {0: (2, 1), 4: (4, 1), 2: (3, 2)}
        assert transitions[(3, 2)] == {7: (2, 1), 6: (3, 1), 5: (4, 1)}
        assert transitions[(4, 1)] == {0: (3, 1), 1: (3, 2)}

    @pytest.mark.skip(reason="include wall movements in transitions")
    @pytest.mark.parametrize("_env_name, _c", [
        ('Maze4-v0', 208),
        ('Maze6-v0', 288),
    ])
    def test_should_calculate_number_of_transitions(self, _env_name, _c):
        env = gym.make(_env_name)
        assert len(env.env.get_transitions()) == _c
