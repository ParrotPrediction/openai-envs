import numpy as np
from gym.utils import colorize

from gym_maze.common import MAZE_ANIMAT, MAZE_WALL, MAZE_PATH, MAZE_REWARD


def render(out, board: np.ndarray):
    out.write("\n")
    board = np.copy(board)

    for row in board:
        out.write(" ".join(_render_element(el) for el in row))
        out.write("\n")


def _render_element(el):
    if el == MAZE_WALL:
        return colorize('■', 'gray')
    elif el == MAZE_PATH:
        return colorize('□', 'white')
    elif el == MAZE_REWARD:
        return colorize('$', 'yellow')
    elif el == MAZE_ANIMAT:
        return colorize('A', 'red')
    else:
        return colorize(el, 'cyan')
