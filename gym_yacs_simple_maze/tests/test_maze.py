import pytest

import gym
from gym.spaces import Discrete
import gym_yacs_simple_maze  # noqa: F401
from gym_yacs_simple_maze.maze import Action


class TestSimpleMaze:

    @pytest.fixture
    def env(self):
        return gym.make("SimpleMaze-v0")

    def test_should_initialize(self, env):
        assert env is not None
        assert env.observation_space == Discrete(4)
        assert env.action_space == Discrete(4)
        assert env.env._position is None

    def test_should_reset_state(self, env):
        obs = env.reset()
        assert obs == ['0', '1', '1', '1']
        assert env.env._position == 3

    def test_should_perform_happy_path(self, env):
        # given
        moves = [
            {'action': Action.NORTH, 'exp_state': '1001'},  # 3 -> 0
            {'action': Action.EAST, 'exp_state': '1010'},  # 0 -> 1
            {'action': Action.EAST, 'exp_state': '1100'},  # 1 -> 2
            {'action': Action.SOUTH, 'exp_state': '0101'},  # 2 -> 5
            {'action': Action.SOUTH, 'exp_state': '0110'},  # 5 -> 8
            {'action': Action.WEST, 'exp_state': '0010'},  # 8 -> 7
        ]

        # when
        env.reset()

        # then
        for step in moves:
            state, reward, done, _ = env.step(step['action'].value)
            assert state == list(step['exp_state'])
            assert reward == 0
            assert done is False

        # final step
        state, reward, done, _ = env.step(Action.WEST.value)
        assert state == list('1011')
        assert reward == 1
        assert done is True

    def test_should_ignore_hitting_the_wall(self, env):
        # given
        env.reset()
        assert env.env._position == 3

        # when
        state, reward, done, _ = env.step(Action.WEST.value)

        assert env.env._position == 3
