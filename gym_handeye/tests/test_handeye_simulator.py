import logging
import sys

from gym_handeye import HandEyeSimulator

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


class TestHandEyeSimulator:
    def test_parse_block_not_under_hand(self):
        # given
        mock = HandEyeSimulator(3, True)
        start = ["g", "w", "w", "w", "w", "w", "w", "w", "b", "0"]

        # when
        mock.parse_observation(start)
        end = mock.observe()

        # then
        assert start == end

    def test_parse_block_in_hand(self):
        # given
        mock = HandEyeSimulator(3, True)
        start = ["w", "w", "w", "w", "w", "w", "w", "w", "b", "2"]

        # when
        mock.parse_observation(start)
        end = mock.observe()

        # then
        assert start == end

    def test_parse_block_under_hand(self):
        # given
        mock = HandEyeSimulator(3, True)
        start = ["w", "w", "w", "w", "w", "w", "w", "w", "g", "1"]

        # when
        mock.parse_observation(start)
        end = mock.observe()

        # then
        assert start == end

    def test_move_north_not_in_hand(self):
        # given
        mock = HandEyeSimulator(3, True)
        start = ["w", "w", "w", "w", "w", "w", "w", "w", "g", "1"]
        mock.parse_observation(start)

        # when
        was_executed = mock.take_action(0)  # move north
        end = mock.observe()

        # then
        assert ["w", "w", "w", "w", "w", "g", "w", "w", "b", "0"] == end
        assert was_executed is True

    def test_move_north_in_hand(self):
        # given
        mock = HandEyeSimulator(3, True)
        start = ["w", "w", "w", "w", "w", "w", "w", "w", "b", "2"]
        mock.parse_observation(start)

        # when
        was_executed = mock.take_action(0)  # move north
        end = mock.observe()

        # then
        assert ["w", "w", "w", "w", "w", "b", "w", "w", "w", "2"] == end
        assert was_executed is True

    def test_move_blocked(self):
        # given
        mock = HandEyeSimulator(3, True)
        start = ["g", "w", "w", "w", "w", "w", "w", "w", "b", "0"]
        mock.parse_observation(start)

        # when
        was_executed = mock.take_action(0)  # move north
        end = mock.observe()

        # then
        assert ["g", "w", "w", "w", "w", "w", "w", "w", "b", "0"] == end
        assert was_executed is False

    def test_grip_block(self):
        # given
        mock = HandEyeSimulator(3, True)
        start = ["w", "w", "w", "w", "w", "w", "w", "w", "g", "1"]
        mock.parse_observation(start)

        # when
        was_executed = mock.take_action(4)  # grip
        end = mock.observe()

        # then
        assert ["w", "w", "w", "w", "w", "w", "w", "w", "b", "2"] == end
        assert was_executed is True

    def test_grip_block_blocked(self):
        # given
        mock = HandEyeSimulator(3, True)
        start = ["b", "w", "w", "w", "w", "w", "w", "w", "g", "0"]
        mock.parse_observation(start)

        # when
        was_executed = mock.take_action(4)  # grip
        end = mock.observe()

        # then
        assert ["b", "w", "w", "w", "w", "w", "w", "w", "g", "0"] == end
        assert was_executed is False

    def test_release_block(self):
        # given
        mock = HandEyeSimulator(3, True)
        start = ["w", "w", "w", "w", "w", "w", "w", "w", "b", "2"]
        mock.parse_observation(start)

        # when
        was_executed = mock.take_action(5)  # release
        end = mock.observe()

        # then
        assert ["w", "w", "w", "w", "w", "w", "w", "w", "g", "1"] == end
        assert was_executed is True

    def test_release_block_blocked(self):
        # given
        mock = HandEyeSimulator(3, True)
        start = ["w", "w", "w", "g", "w", "w", "w", "w", "b", "0"]
        mock.parse_observation(start)

        # when
        was_executed = mock.take_action(5)  # release
        end = mock.observe()

        # then
        assert ["w", "w", "w", "g", "w", "w", "w", "w", "b", "0"] == end
        assert was_executed is False
