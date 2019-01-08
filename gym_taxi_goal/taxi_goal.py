import gym
from gym.envs.toy_text.taxi import TaxiEnv

STATE_MOVE_TO_PASSENGER = 1
STATE_PICKUP_PASSENGER = 2
STATE_MOVE_TO_DEST = 3
STATE_DROPOFF_PASSENGER = 4
STATE_OVER = 5


class TaxiGoalEnv(TaxiEnv):
    def __init__(self):
        TaxiEnv.__init__(self)
        self.goal_generator_state = STATE_OVER

    def get_goal_state(self):
        """
        Goal generator func.
        1. Move to the passenger location.
        2. Pick up the passenger.
        3. Move to destination location.
        4. Drop off the passenger.
        :return:
        """
        decoded = list(self.decode(self.s))
        taxirow = decoded[0]
        taxicol = decoded[1]
        passloc = decoded[2]
        destidx = decoded[3]

        taxiloc = (taxirow, taxicol)

        if self.goal_generator_state == STATE_DROPOFF_PASSENGER:
            self.goal_generator_state = STATE_OVER
            return None

        if passloc == 4:  # it means passenger is in the taxi
            if taxiloc == self.locs[destidx]:  # taxi is in dest
                passloc = destidx
                self.goal_generator_state = STATE_DROPOFF_PASSENGER
            else:  # taxi not in dest
                taxirow = self.locs[destidx][0]
                taxicol = self.locs[destidx][1]
                self.goal_generator_state = STATE_MOVE_TO_DEST
        else:  # passenger not in taxi
            if taxiloc == self.locs[passloc]:  # taxi over passenger
                passloc = 4
                self.goal_generator_state = STATE_PICKUP_PASSENGER
            else:  # taxi not over passenger
                taxirow = self.locs[passloc][0]
                taxicol = self.locs[passloc][1]
                self.goal_generator_state = STATE_MOVE_TO_PASSENGER

        return self.encode(taxirow, taxicol, passloc, destidx)
