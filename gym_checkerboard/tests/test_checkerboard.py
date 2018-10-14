import gym
import pytest

# noinspection PyUnresolvedReferences
import gym_checkerboard


class TestCheckerboard:

    @pytest.fixture(name='env')
    def checkerboard(self):
        return gym.make('checkerboard-2D-5div-v0')

    def test_initializing_environment(self, env):
        assert env is not None
        assert env.action_space.n == 2
        assert list(env.observation_space.low) == [0., 0., 0.]
        assert list(env.observation_space.high) == [1., 1., 1.]

    def test_should_reset_environment(self, env):
        # when
        state = env.reset()

        # then
        assert len(state) == 3
        assert state[-1] == 0

    def test_should_execute_step(self, env):
        # given
        action = env.action_space.sample()

        # when
        env.reset()
        state, reward, done, _ = env.step(action)

        # then
        assert len(state) == 3
        assert state[-1] in [0, 1]
        assert done in [True, False]
