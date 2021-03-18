import random

import numpy as np

from gym_maze.common import MAZE_REWARD, MAZE_WALL, MAZE_PATH, MAZE_ANIMAT
from gym_maze.common.maze_utils import get_reward_xy, \
    get_possible_neighbour_cords
from gym_maze.internal.abstract_maze import AbstractMaze

ACTION_LOOKUP = {
    0: 'N',
    1: 'NE',
    2: 'E',
    3: 'SE',
    4: 'S',
    5: 'SW',
    6: 'W',
    7: 'NW'
}


def find_action_by_direction(direction):
    for key, val in ACTION_LOOKUP.items():
        if val == direction:
            return key


class MazeImpl(AbstractMaze):

    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix)
        self.x = None  # TODO remove this
        self.y = None
        self._goal = get_reward_xy(matrix)

    def move(self, action: int):
        perception = self.perception()

        def _can_move(el: str):
            return el != str(MAZE_WALL)

        animat_moved = False
        action_type = ACTION_LOOKUP[action]

        if action_type == "N" and _can_move(perception[0]):
            self.y -= 1
            animat_moved = True

        if action_type == 'NE' and _can_move(perception[1]):
            self.x += 1
            self.y -= 1
            animat_moved = True

        if action_type == "E" and _can_move(perception[2]):
            self.x += 1
            animat_moved = True

        if action_type == 'SE' and _can_move(perception[3]):
            self.x += 1
            self.y += 1
            animat_moved = True

        if action_type == "S" and _can_move(perception[4]):
            self.y += 1
            animat_moved = True

        if action_type == 'SW' and _can_move(perception[5]):
            self.x -= 1
            self.y += 1
            animat_moved = True

        if action_type == "W" and _can_move(perception[6]):
            self.x -= 1
            animat_moved = True

        if action_type == 'NW' and _can_move(perception[7]):
            self.x -= 1
            self.y -= 1
            animat_moved = True

        return animat_moved

    def get_goal_state(self):
        """
        Goal generator function used in Action Planning.
        Returns next perception toward reaching the goal
        :return:
            perception of a goal state
        """
        def adjust(p):
            return [str(MAZE_PATH) if e == str(MAZE_ANIMAT) else e for e in p]

        if str(MAZE_REWARD) in self.perception():
            return adjust(self.perception(self._goal))
        elif self.agent_position == self._goal:
            return None
        else:
            pos_x, pos_y = random.choice(get_possible_neighbour_cords(*self._goal))
            while not self.is_path((pos_x, pos_y)):
                pos_x, pos_y = random.choice(get_possible_neighbour_cords(*self._goal))
            return adjust(self.perception((pos_x, pos_y)))
