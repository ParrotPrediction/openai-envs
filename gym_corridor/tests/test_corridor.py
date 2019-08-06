import logging
import sys

import gym

# noinspection PyUnresolvedReferences
import gym_corridor
from gym_corridor.corridor import MOVE_LEFT, MOVE_RIGHT

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


class TestCorridor:

    def test_should_initialize(self):
        # when
        corr = gym.make('corridor-20-v0')

        # then
        assert corr is not None
        assert 1 == corr.observation_space.n
        assert 2 == corr.action_space.n

    def test_should_visualize(self):
        # given
        corr = gym.make('corridor-20-v0')

        # when
        obs = corr.reset()
        vis = corr.render(mode='ansi')

        # then
        assert 1 <= int(obs) < 20
        assert len(vis) == 22
        assert 1 == vis.count('X')
        assert 1 == vis.count('$')
        assert 18 == vis.count('.')

    def test_should_hit_left_wall(self):
        # given
        corr = gym.make('corridor-20-v0')
        reward = 0
        done = False

        # when
        obs = corr.reset()

        while not done:
            obs, reward, done, _ = corr.step(MOVE_LEFT)

        # then
        assert obs == '1'
        assert reward == 0
        assert done is True

    def test_should_get_reward(self):
        # given
        corr = gym.make('corridor-20-v0')
        reward = 0
        done = False

        # when
        obs = corr.reset()

        while not done:
            obs, reward, done, _ = corr.step(MOVE_RIGHT)

        # then
        assert obs == '20'
        assert reward == 1000
        assert done is True

    def test_should_move_in_both_directions(self):
        # given
        corr = gym.make('corridor-20-v0')
        p0 = corr.reset()

        while p0 in ["1", "19"]:
            p0 = corr.reset()

        # when & then
        p1, _, _, _ = corr.step(MOVE_LEFT)
        assert int(p1) == int(p0) - 1

        p2, _, _, _ = corr.step(MOVE_RIGHT)
        assert int(p2) == int(p0)
