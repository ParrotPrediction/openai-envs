import random

from gym_handeye.handeye import ACTION_LOOKUP

BLOCK_NOT_UNDER_GRIPPER = '0'
BLOCK_UNDER_GRIPPER = '1'
BLOCK_IN_HAND = '2'

SURFACE = 'w'
BLOCK = 'b'
GRIPPER = 'g'

STATE_BEGINNING = 0
STATE_MOVE_OVER_BLOCK = 1
STATE_GRIP_BLOCK = 2
STATE_MOVE_WITH_BLOCK = 3
STATE_RELEASE_BLOCK = 4
STATE_MOVE_NOT_OVER_BLOCK = 5


class HandEyeSimulator():
    """
    Class responsible for the logic of HandEye environment
    """

    def __init__(self, grid_size, note_in_hand, test_only_changes):
        """

        :param grid_size: specifies the size of the monitored plain
        :param note_in_hand: specifies if the tacticle sensor should switch to '2' if the block is held by the gripper
        (if False, then goes back to '0')
        :param test_only_changes: specifies if only condition-action combinations should be tested that invoke a change
        """
        self.grid_size = grid_size
        self.note_in_hand = note_in_hand
        self.test_only_changes = test_only_changes

        self.grip_pos_x = None
        self.grip_pos_y = None
        self.block_in_hand = False
        self.block_pos_x = None
        self.block_pos_y = None

        self.env_size = self.grid_size * self.grid_size + 1

        self.goal_generator_state = STATE_BEGINNING

        self.observation = self._get_empty_observation()

    def _move_gripper(self, x_end, y_end):
        """
        Executes a moving action with all involved consequences.
        :param x_end: End gripper x position
        :param y_end: End gripper y position
        :return:
        """
        x_start = self.grip_pos_x
        y_start = self.grip_pos_y

        self.grip_pos_x = x_end
        self.grip_pos_y = y_end

        if self.block_in_hand:
            self.block_pos_x = x_end
            self.block_pos_y = y_end
            self._set_observation_grid(x_start, y_start, SURFACE)
            self._set_observation_grid(x_end, y_end, BLOCK)
        else:
            self._set_observation_grid(x_start, y_start, SURFACE)
            self._set_observation_grid(x_end, y_end, GRIPPER)

        if self.block_pos_x == x_start and self.block_pos_y == y_start:
            self._set_observation_grid(x_start, y_start, BLOCK)

        self._set_observation_gripper_state(BLOCK_NOT_UNDER_GRIPPER)

        if self._is_above_block():
            if not self.block_in_hand:
                self._set_observation_gripper_state(BLOCK_UNDER_GRIPPER)
            else:
                if self.note_in_hand:
                    self._set_observation_gripper_state(BLOCK_IN_HAND)
        return

    def _grip_block(self):
        """
        Executes a gripping action.
        :return: True if block was gripped
        """
        if self.block_in_hand:
            return False

        if self._is_above_block():
            self.block_in_hand = True
            self._set_observation_grid(self.grip_pos_x, self.grip_pos_y, BLOCK)
            if self.note_in_hand:
                self._set_observation_gripper_state(BLOCK_IN_HAND)
            else:
                self._set_observation_gripper_state(BLOCK_NOT_UNDER_GRIPPER)
            return True

        return False

    def _release_block(self):
        """
        Releases a block if a block is held in the hand.
        :return: True if block was released
        """
        if self.block_in_hand:
            self._set_observation_gripper_state(BLOCK_UNDER_GRIPPER)
            self._set_observation_grid(self.grip_pos_x, self.grip_pos_y, GRIPPER)
            self.block_in_hand = False
            return True

        return False

    def observe(self):
        return self.observation

    def take_action(self, action):
        """
        Executes an action if possible
        :param action: Action to execute
        :return: True if action was executed
        """
        action_type = ACTION_LOOKUP[action]

        if action_type == "N" and self.grip_pos_y > 0:
            self._move_gripper(self.grip_pos_x, self.grip_pos_y - 1)
            return True

        elif action_type == "E" and self.grip_pos_x < self.grid_size - 1:
            self._move_gripper(self.grip_pos_x + 1, self.grip_pos_y)
            return True

        elif action_type == "S" and self.grip_pos_y < self.grid_size - 1:
            self._move_gripper(self.grip_pos_x, self.grip_pos_y + 1)
            return True

        elif action_type == "W" and self.grip_pos_x > 0:
            self._move_gripper(self.grip_pos_x - 1, self.grip_pos_y)
            return True

        elif action_type == "G":
            if self._grip_block():
                return True

        elif action_type == "R":
            if self._release_block():
                return True

        return False

    def set_random_positions(self):
        """
        Non deterministic function, sets block on random position, then sets block in hand or not
        (if not in hand, then sets griper on random position).
        :return: observation after setting block and gripper
        """
        self.observation = self._get_empty_observation()

        # random block position
        self.block_pos_x = self._get_random_position()
        self.block_pos_y = self._get_random_position()

        self._set_observation_grid(self.block_pos_x, self.block_pos_y, BLOCK)

        if random.choice([True, False]):
            self._set_block_in_hand()
        else:
            self._set_block_in_hand()

        return self.observation

    def _get_empty_observation(self):
        """
        Returns empty observation array
        :return: empty observation (only SURFACE and BLOCK_NOT_UNDER_GRIPPER)
        """
        obs = [SURFACE for x in range(self.env_size - 1)]
        obs.append(BLOCK_NOT_UNDER_GRIPPER)
        return obs

    def _is_above_block(self):
        """
        Checks if gripper is above block
        :return: True if gripper is above block
        """
        return self.block_pos_x == self.grip_pos_x and self.block_pos_y == self.grip_pos_y

    def _set_block_in_hand(self):
        """
        Sets gripper position when block is in hand
        :return:
        """
        self.block_in_hand = True

        self.grip_pos_x = self.block_pos_x
        self.grip_pos_y = self.block_pos_y

        if self.note_in_hand:
            self._set_observation_gripper_state(BLOCK_IN_HAND)

    def _set_block_not_in_hand(self):
        """
        Sets gripper position when block is not in hand.
        :return:
        """
        self.block_in_hand = False

        self.grip_pos_x = self._get_random_position()
        self.grip_pos_y = self._get_random_position()

        self._set_observation_grid(self.grip_pos_x, self.grip_pos_y, GRIPPER)

        if self._is_above_block():
            self._set_observation_gripper_state(BLOCK_UNDER_GRIPPER)

    def _set_observation_grid(self, x, y, observe):
        """
        Sets what we observe at (x,y)
        :param x:
        :param y:
        :param observe: BLOCK, GRIPPER or SURFACE
        :return:
        """
        self._set_grid(self.observation, x, y, observe)

    def _set_observation_gripper_state(self, state):
        """
        Sets what gripper feels
        :param state: BLOCK_NOT_UNDER_GRIPPER, BLOCK_UNDER_GRIPPER, BLOCK_IN_HAND
        :return:
        """
        self._set_gripper_state(self.observation, state)

    def _set_goal_grid(self, x, y, observe):
        """
        Sets what we observe at (x,y)
        :param x:
        :param y:
        :param observe: BLOCK, GRIPPER or SURFACE
        :return:
        """
        self._set_grid(self.goal_state, x, y, observe)

    def _set_goal_gripper_state(self, state):
        """
        Sets what gripper feels
        :param state: BLOCK_NOT_UNDER_GRIPPER, BLOCK_UNDER_GRIPPER, BLOCK_IN_HAND
        :return:
        """
        self._set_gripper_state(self.goal_state, state)

    def _set_gripper_state(self, observation, state):
        """
        Sets what gripper feels
        :param state: BLOCK_NOT_UNDER_GRIPPER, BLOCK_UNDER_GRIPPER, BLOCK_IN_HAND
        :return:
        """
        if state not in [BLOCK_NOT_UNDER_GRIPPER, BLOCK_UNDER_GRIPPER, BLOCK_IN_HAND]:
            return
        observation[self.env_size - 1] = state

    def _set_grid(self, observation, x, y, observe):
        """
        Sets what we observe at (x,y)
        :param x:
        :param y:
        :param observe: BLOCK, GRIPPER or SURFACE
        :return:
                """
        if observe not in [BLOCK, GRIPPER, SURFACE]:
            return
        observation[y * self.grid_size + x] = observe

    def get_goal_state(self):
        """
        The goal generator generates continuously the following states:
        1. Move over the block.
        2. Grip the block.
        3. Move with the block to a random position.
        4. Release the block.
        5. Move to a random position not over the block.
        Returns observation that is the environment's next goal.
        :return: observation that is environment's next goal
        """
        if self.goal_generator_state == STATE_MOVE_NOT_OVER_BLOCK:
            self.goal_generator_state = STATE_BEGINNING
            return

        self.goal_state = self._get_empty_observation()

        if self.block_in_hand:
            if self.goal_generator_state == STATE_GRIP_BLOCK:
                self._state_move_with_block()
            else:
                self._state_release_block()
        else:
            if self.observation[self.env_size - 1] == BLOCK_UNDER_GRIPPER:
                if self.goal_generator_state == STATE_RELEASE_BLOCK:
                    self._state_move_not_over_block()
                else:
                    self._state_grip_block()
            else:
                self._state_move_over_block()

        return self.goal_state

    def _state_move_not_over_block(self):
        """
        Changes the goal_state: the gripper is in random position not over block.
        :return:
        """
        while True:
            x = self._get_random_position()
            y = self._get_random_position()
            if x == self.grip_pos_x or y == self.grip_pos_y:
                break
        self._set_goal_grid(self.grip_pos_x, self.grip_pos_y, BLOCK)
        self._set_goal_grid(x, y, GRIPPER)
        self._set_goal_gripper_state(BLOCK_NOT_UNDER_GRIPPER)
        self.goal_generator_state = STATE_MOVE_NOT_OVER_BLOCK

    def _state_move_with_block(self):
        """
        Changes the goal_state: the gripper is moving with block in hand to ranom position.
        :return:
        """
        while True:
            x = self._get_random_position()
            y = self._get_random_position()
            if x == self.grip_pos_x or y == self.grip_pos_y:
                break
        self._set_goal_grid(self.grip_pos_x, self.grip_pos_y, SURFACE)
        self._set_goal_grid(x, y, BLOCK)
        self.goal_generator_state = STATE_MOVE_WITH_BLOCK

    def _state_release_block(self):
        """
        Changes the goal_state: the gripper realases the block.
        :return:
        """
        self._set_goal_grid(self.grip_pos_x, self.grip_pos_y, GRIPPER)
        self._set_goal_gripper_state(BLOCK_UNDER_GRIPPER)
        self.goal_generator_state = STATE_RELEASE_BLOCK

    def _state_move_over_block(self):
        """
        Changes the goal_state: the gripper moves over block.
        :return:
        """
        self._set_goal_grid(self.grip_pos_x, self.grip_pos_y, SURFACE)
        self._set_goal_grid(self.block_pos_x, self.block_pos_y, GRIPPER)
        self._set_goal_gripper_state(BLOCK_UNDER_GRIPPER)
        self.goal_generator_state = STATE_MOVE_OVER_BLOCK

    def _state_grip_block(self):
        """
        Changes the goal_state: the gripper grips the block.
        :return:
        """
        if self.note_in_hand:
            self._set_goal_gripper_state(BLOCK_IN_HAND)
        else:
            self._set_goal_gripper_state(BLOCK_NOT_UNDER_GRIPPER)
        self._set_goal_grid(self.grip_pos_x, self.grip_pos_y, BLOCK)
        self.goal_generator_state = STATE_GRIP_BLOCK

    def parse_observation(self, observation):
        self.observation = self._get_empty_observation()
        i = 0
        for item in observation:
            self.observation[i] = item
            i += 1

        self.block_pos_x = -1
        self.block_pos_y = -1
        self.grip_pos_x = -1
        self.grip_pos_y = -1
        self.block_in_hand = False
        for i, field in enumerate(observation):
            if field == 'b':
                self.block_pos_x = i % self.grid_size
                self.block_pos_y = int(i / self.grid_size)
            if field == 'g':
                self.grip_pos_x = i % self.grid_size
                self.grip_pos_y = int(i / self.grid_size)
        if self.grip_pos_x == -1:
            self.block_in_hand = True
            self.grip_pos_x = self.block_pos_x
            self.grip_pos_y = self.block_pos_y
        elif self.block_pos_x == -1:
            self.block_pos_x = self.grip_pos_x
            self.block_pos_y = self.grip_pos_y

    def _get_random_position(self):
        return random.randint(0, self.grid_size - 1)
