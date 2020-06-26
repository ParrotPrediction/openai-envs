import numpy as np
import pytest

from gym_woods.woods import Woods


class TestWoods:

    SCHEMA = np.asarray([
            list("...."),
            list("OOG."),
            list(".F.."),
        ])

    def test_should_calculate_boundaries(self):
        # when
        woods = Woods(self.SCHEMA)

        # then
        assert woods.max_x == 4
        assert woods.max_y == 3

    def test_should_get_insertion_coordinates(self):
        # given
        woods = Woods(self.SCHEMA)

        # when
        cords = woods.possible_insertion_cords

        # then
        assert len(cords) == 8
        assert (0, 0) in cords
        assert (3, 0) in cords
        assert (0, 1) not in cords

    def test_should_raise_error_with_invalid_cords(self):
        # given
        woods = Woods(self.SCHEMA)

        # then
        woods.perception(0, 2)

        woods.perception(1, 1)

        with pytest.raises(ValueError):
            # negative value
            woods.perception(-1, 0)

        with pytest.raises(ValueError):
            # x outside range
            woods.perception(4, 1)

        with pytest.raises(ValueError):
            # y outside range
            woods.perception(1, 3)

    def test_should_calculate_perception(self):
        # given
        woods = Woods(self.SCHEMA)

        # when & then
        assert list("F..GOO..") == woods.perception(1, 0)
        assert list("...O.G..") == woods.perception(3, 0)
        assert list("..G.F.O.") == woods.perception(1, 1)
        assert list("OOF.....") == woods.perception(0, 2)

    def test_should_detect_reward(self):
        # given
        woods = Woods(self.SCHEMA)

        # then
        assert woods.is_reward(0, 0) is False
        assert woods.is_reward(0, 1) is False
        assert woods.is_reward(2, 1) is True
        assert woods.is_reward(1, 2) is True
