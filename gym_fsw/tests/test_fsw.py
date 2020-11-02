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

    def test_should_follow_optimal_path_and_find_reward(self):
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
        state, _, done, _ = fsw.step(1)
        assert state == '5'
        assert done is False

        state, _, done, _ = fsw.step(0)
        assert state == '1'
        assert done is False

        state, _, done, _ = fsw.step(0)
        assert state == '6'
        assert done is False

        state, _, done, _ = fsw.step(1)
        assert state == '2'
        assert done is False

        state, _, done, _ = fsw.step(1)
        assert state == '7'
        assert done is False

        state, _, done, _ = fsw.step(0)
        assert state == '3'
        assert done is False

        state, _, done, _ = fsw.step(1)
        assert state == '4'
        assert done is False

        state, reward, done, _ = fsw.step(0)
        assert state == '10'
        assert reward == 100
        assert done is True


    def test_should_initialize_bigger_environment(self):
        # given
        fsw = gym.make('fsw-10-v0')
        fsw.reset()

        # when ^ then
        state, _, _, _ = fsw.step(0)
        assert state == '1'

        state, _, _, _ = fsw.step(1)
        assert state == '2'

        state, _, _, _ = fsw.step(1)
        assert state == '13'

    def test_should_get_all_states_and_actions(self):
        # given
        fsw = gym.make('fsw-5-v0')
        fsw.reset()

        # when
        mapping = fsw._state_action()

        # then
        assert len(mapping) == 11
