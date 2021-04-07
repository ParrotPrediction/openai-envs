import numpy as np
import pytest

from gym_maze.common import MAZE_ANIMAT
from gym_maze.internal.rotating_maze_impl import RotatingMazeImpl


class TestAbstractRotatingMazeImpl:

    @pytest.fixture
    def maze(self):
        return RotatingMazeImpl(np.asarray([
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 9, 0, 1],
            [1, 1, 1, 1, 1],
        ]))

    def test_should_initialize(self, maze):
        assert maze.matrix is not None

    def test_should_insert_agent(self, maze):
        # when
        maze.insert_agent()

        # then
        agent_xy = maze.agent_position
        assert maze.matrix[agent_xy[0], agent_xy[1]] == MAZE_ANIMAT

    @pytest.mark.parametrize('_xy, _p', [
        ((1, 2), list('110000110')),
        ((3, 3), list('011111900')),
    ])
    def test_should_get_perception(self, _xy, _p, maze):
        maze.insert_agent(_xy)
        assert maze.perception() == _p

    def test_agent_should_turn_left(self, maze):
        # given
        maze.insert_agent((1, 2))
        init_obs = list('110000110')
        assert maze.perception() == init_obs

        # when & then
        maze.turn_left()
        assert maze.perception() == list('111100000')

        maze.turn_left()
        assert maze.perception() == list('001111000')

        maze.turn_left()
        assert maze.perception() == list('000011110')

        maze.turn_left()
        assert maze.perception() == init_obs

    def test_agent_should_turn_right(self, maze):
        # given
        maze.insert_agent((3, 1))
        init_obs = list('009111110')
        assert maze.perception() == init_obs

        # when & then
        maze.turn_right()
        assert maze.perception() == list('911111000')

        maze.turn_right()
        assert maze.perception() == list('111100910')

        maze.turn_right()
        assert maze.perception() == list('110091110')

        maze.turn_right()
        assert maze.perception() == init_obs

    def test_should_step_ahead(self, maze):
        # given
        maze.insert_agent((3, 1))
        assert maze.perception() == list('009111110')

        # when stepping into path
        maze.step_ahead()
        assert maze.perception() == list('100901110')

        # when stepping into wall
        maze.step_ahead()
        assert maze.perception() == list('100901110')

    def test_should_get_reward(self, maze):
        # given
        maze.insert_agent((3, 1))
        assert maze.is_done() is False

        # when
        maze.step_ahead()
        maze.turn_right()
        maze.step_ahead()
        maze.turn_right()
        maze.step_ahead()

        # then
        assert maze.is_done() is True

    def test_should_get_correct_perceptions(self, maze):
        # given
        maze.insert_agent((3, 1))
        assert maze.perception() == list('009111110')

        # when & then
        maze.turn_left()
        assert maze.perception() == list('110091110')

        maze.turn_right()
        assert maze.perception() == list('009111110')

        maze.step_ahead()
        assert maze.perception() == list('100901110')

        maze.turn_right()
        assert maze.perception() == list('090111100')

        maze.step_ahead()
        assert maze.perception() == list('009001000')

    def test_should_remain_in_position_when_hitting_wall(self, maze):
        # given
        maze.insert_agent((3, 1))
        assert maze.perception() == list('009111110')

        # when
        maze.turn_right()
        maze.turn_right()
        assert maze.perception() == list('111100910')

        maze.step_ahead()
        assert maze.perception() == list('111100910')

        maze.turn_left()
        maze.step_ahead()
        assert maze.found_reward
        assert maze.perception() == list('011100009')
