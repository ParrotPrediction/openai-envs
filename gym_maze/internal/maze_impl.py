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
        self.found_reward = False
        self._goal = get_reward_xy(matrix)

    def is_done(self) -> bool:
        return self.found_reward

    def move(self, action: int) -> None:
        perception = self.perception()
        x, y = self.agent_position
        next_state = None

        def _can_move(el: str):
            return el != str(MAZE_WALL)

        action_type = ACTION_LOOKUP[action]

        if action_type == "N" and _can_move(perception[0]):
            next_state = (x-1, y)

        if action_type == 'NE' and _can_move(perception[1]):
            next_state = (x-1, y+1)

        if action_type == "E" and _can_move(perception[2]):
            next_state = (x, y+1)

        if action_type == 'SE' and _can_move(perception[3]):
            next_state = (x+1, y+1)

        if action_type == "S" and _can_move(perception[4]):
            next_state = (x+1, y)

        if action_type == 'SW' and _can_move(perception[5]):
            next_state = (x+1, y-1)

        if action_type == "W" and _can_move(perception[6]):
            next_state = (x, y-1)

        if action_type == 'NW' and _can_move(perception[7]):
            next_state = (x-1, y-1)

        if next_state:
            if self.matrix[next_state] == MAZE_REWARD:
                self.found_reward = True
            else:
                self.matrix[x, y] = MAZE_PATH
                self.matrix[next_state] = MAZE_ANIMAT

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
            ncords = get_possible_neighbour_cords(*self._goal)
            pos_x, pos_y = random.choice(ncords)

            while not self.is_path((pos_x, pos_y)):
                pos_x, pos_y = random.choice(ncords)

            return adjust(self.perception((pos_x, pos_y)))
