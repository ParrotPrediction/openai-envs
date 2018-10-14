import numpy as np
import pytest

from gym_checkerboard.checkerboard_simulator import CheckerboardSimulator, \
    WHITE, BLACK


class TestCheckerboard:

    @pytest.mark.parametrize("_dim, _ndiv", [
        (1, 1), (2, 3), (3, 3), (3, 5)
    ])
    def test_should_build_board(self, _dim, _ndiv):
        # given
        chb = CheckerboardSimulator(dim=_dim, ndiv=_ndiv)

        # when
        b = chb.board

        # then
        assert type(b) is np.ndarray
        assert b.size == pow(_ndiv, _dim)
        assert b.shape == (_ndiv,) * _dim

        # check if colors are alternating
        b_flat = np.reshape(b, (1, pow(_ndiv, _dim)))
        color = BLACK  # first is always black

        for c in np.nditer(b_flat):
            assert c == color
            # alternate color
            color = WHITE if color == BLACK else BLACK

    @pytest.mark.parametrize("_cords, _color", [
        ([.25, .25], 1),
        ([.4, .7], 0),
    ])
    def test_should_return_proper_color(self, _cords, _color):
        # given
        chb = CheckerboardSimulator(dim=2, ndiv=3)

        # then
        assert chb.get_color(*_cords) == _color

    @pytest.mark.parametrize("_val, _result", [
        (.0, 0), (.1, 0), (.199, 0),
        (.2, 1), (.399, 1),
        (.8, 4), (.999, 4)
    ])
    def test_should_return_index(self, _val, _result):
        # given
        chb = CheckerboardSimulator(dim=2, ndiv=5)

        # then
        assert chb._get_index(_val) == _result
