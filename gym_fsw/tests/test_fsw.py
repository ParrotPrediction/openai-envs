import gym


class TestFiniteStateWorld:

    def test_should_initialize(self):
        # when
        fsw = gym.make('fsw-5-v0')

        # then
        assert fsw is not None
        assert 1 == fsw.observation_space.n
        assert 2 == fsw.action_space.n

    def test_should_start_from_initial_position(self):
        # given
        fsw = gym.make('fsw-5-v0')

        # when
        state = fsw.reset()

        # then
        assert '0' == state

    def test_should_folloūw_optimal_path_and_find_reward(self):
        # given
        fsw = gym.make('fsw-5-v0')
        fsw.reset()

        # when & then
        state, _, _, _ = fsw.step(0)
        assert state == '1'

        state, _, _, _ = fsw.step(1)
        assert state == '2'

        state, _, _, _ = fsw.step(0)
        assert state == '3'

        state, _, _, _ = fsw.step(1)
        assert state == '4'

        state, reward, done, _ = fsw.step(0)
        assert state == '5'
        assert reward == 100
        assert done is True

    def test_should_follow_suboptimal_path_and_find_reward(self):
        # given
        fsw = gym.make('fsw-5-v0')
        fsw.reset()

        # when & then
        state, _, _, _ = fsw.step(1)
        assert state == '5'

        state, _, _, _ = fsw.step(0)
        assert state == '1'

        state, _, _, _ = fsw.step(0)
        assert state == '6'

        state, _, _, _ = fsw.step(1)
        assert state == '2'

        state, _, _, _ = fsw.step(1)
        assert state == '7'

        state, _, _, _ = fsw.step(0)
        assert state == '3'

        state, _, _, _ = fsw.step(0)
        assert state == '8'

        state, _, _, _ = fsw.step(1)
        assert state == '4'

        state, _, _, _ = fsw.step(1)
        assert state == '9'

        state, reward, done, _ = fsw.step(0)
        assert state == '5'
        assert reward == 100
        assert done is True

