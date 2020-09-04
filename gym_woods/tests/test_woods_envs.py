import gym
import pytest

# noinspection PyUnresolvedReferences
import gym_woods  # noqa: F401


class TestWoodsEnvs:

    @pytest.mark.parametrize("_env_name, ", [
        'Woods1-v0',
        'Woods2-v0',
        'Woods14-v0',
    ])
    def test_should_initialize(self, _env_name):
        # when
        env = gym.make(_env_name)

        # then
        assert env is not None
        assert env.action_space.n == 8
        assert env.observation_space.n == 8
        assert env.observation_space.SYMBOLS == tuple('.*OQFG')
        assert env.env.maze is not None

    @pytest.mark.parametrize("_env_name, ", [
        'Woods1-v0',
        'Woods2-v0',
        'Woods14-v0',
    ])
    def test_should_reset(self, _env_name):
        # given
        env = gym.make(_env_name)

        # when
        state = env.reset()

        # then
        x, y = env.env.pos_x, env.env.pos_y
        assert len(state) == 8
        assert x is not None
        assert y is not None
        assert env.env.maze.matrix[y, x] == '.'

    @pytest.mark.parametrize("_env_name, ", [
        'Woods1-v0',
        'Woods2-v0',
    ])
    def test_should_make_toroidal_move(self, _env_name):
        # given
        env = gym.make(_env_name)
        env.reset()

        max_x = env.env.maze.max_x - 1
        max_y = env.env.maze.max_y - 1

        def _place_animat(env, x, y):
            if env.env.maze.matrix[y, x] != '.':
                raise ValueError('Not a path')

            env.env.pos_x = x
            env.env.pos_y = y

        def _assert_cords(env, x1, y1):
            assert env.env.pos_x == x1
            assert env.env.pos_y == y1

        def _assert_not_done(reward, done):
            assert reward == 0
            assert done is False

        # when moving N near border
        _place_animat(env, 2, 0)
        obs, reward, done, _ = env.step(0)
        _assert_cords(env, 2, max_y)
        _assert_not_done(reward, done)

        # when moving NE near corner
        _place_animat(env, max_x, 0)
        obs, reward, done, _ = env.step(1)
        _assert_cords(env, 0, max_y)
        _assert_not_done(reward, done)

        # when moving E near border
        _place_animat(env, max_x, 1)
        obs, reward, done, _ = env.step(2)
        _assert_cords(env, 0, 1)
        _assert_not_done(reward, done)

        # when moving SE near corner
        _place_animat(env, max_x, max_y)
        obs, reward, done, _ = env.step(3)
        _assert_cords(env, 0, 0)
        _assert_not_done(reward, done)

        # when moving S near border
        _place_animat(env, 2, max_y)
        obs, reward, done, _ = env.step(4)
        _assert_cords(env, 2, 0)
        _assert_not_done(reward, done)

        # when moving SE near corner
        _place_animat(env, 0, max_y)
        obs, reward, done, _ = env.step(5)
        _assert_cords(env, max_x, 0)
        _assert_not_done(reward, done)

        # when moving E near border
        _place_animat(env, 0, 2)
        obs, reward, done, _ = env.step(6)
        _assert_cords(env, max_x, 2)
        _assert_not_done(reward, done)

        # when moving NW near corner
        _place_animat(env, 0, 0)
        obs, reward, done, _ = env.step(7)
        _assert_cords(env, max_x, max_y)
        _assert_not_done(reward, done)

    def test_should_get_all_states_and_actions(self):
        # given
        woods = gym.make('Woods1-v0')
        woods.reset()

        # when
        mapping = woods.env._state_action()

        # then
        assert len(mapping) == 16
        assert mapping[('0', '2')] == [0, 4, 5, 6, 7]
        assert mapping[('3', '0')] == [0, 1, 2, 3, 4, 6, 7]
