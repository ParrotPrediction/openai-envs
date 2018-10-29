import gym
import pytest


class TestXAxisBoundaryEnv:

    @pytest.fixture
    def env(self):
        return gym.make('orthogonal-single-boundary-v0')

    def test_should_initialize(self, env):
        assert env is not None
        assert env.action_space.n == 2
        assert env.observation_space.shape == (3,)

    def test_should_reset_environment(self, env):
        # when
        obs = env.reset()

        # then
        assert obs is not None
        assert len(obs) == 3
        assert obs[-1] == 0

    def test_should_execute_step(self, env):
        # given
        env.reset()
        action = env.action_space.sample()

        # when
        state, reward, done, _ = env.step(action)

        # then
        assert state is not None
        assert done is not None
        assert reward in [0, 1]
        assert type(state) is list

    @pytest.mark.parametrize("_x, _action, _correct", [
        (0.0, 1, True),
        (0.2, 0, False),
        (0.49, 1, True),
        (0.5, 1, False),
        (1.0, 0, True),
    ])
    def test_correct_action(self, _x, _action, _correct, env):
        # given
        env.reset()
        env._state[0] = _x

        # when
        state, reward, done, _ = env.step(_action)

        # then
        assert done is True
        if _correct:
            assert reward == 1
            assert state[-1] == 1
        else:
            assert reward == 0
            assert state[-1] == 0
