from typing import Tuple

import numpy as np

from gym_maze.common import MAZE_PATH, MAZE_REWARD, MAZE_ANIMAT


def get_possible_insertion_coordinates(matrix: np.ndarray) \
        -> Tuple[Tuple[int, int]]:
    """
    Returns a list with coordinates in the environment where
    an agent can be placed (only on the path).
    :return: list of tuples (X,Y) containing coordinates
    """
    x_idx, y_idx = np.where(matrix == MAZE_PATH)
    return tuple(zip(x_idx, y_idx))


def get_animat_xy(matrix: np.ndarray) -> Tuple[int, int]:
    animats = np.argwhere(matrix == MAZE_ANIMAT)
    assert len(animats) == 1
    return tuple(animats[0])


def get_reward_xy(matrix: np.ndarray) -> Tuple[int, int]:
    rewards = np.argwhere(matrix == MAZE_REWARD)
    assert len(rewards) == 1
    return tuple(rewards[0])


def adjacent_cells(matrix, x, y) -> Tuple:
    max_x, max_y = matrix.shape

    assert 0 <= x < max_x
    assert 0 <= y < max_y

    # Position N
    if y == 0:
        n = None
    else:
        n = matrix[x - 1, y]

    # Position NE
    if x == max_x - 1 or y == 0:
        ne = None
    else:
        ne = matrix[x - 1, y + 1]

    # Position E
    if x == max_x - 1:
        e = None
    else:
        e = matrix[x, y + 1]

    # Position SE
    if x == max_x - 1 or y == max_y - 1:
        se = None
    else:
        se = matrix[x + 1, y + 1]

    # Position S
    if y == (max_y - 1):
        s = None
    else:
        s = matrix[x + 1, y]

    # Position SW
    if x == 0 or y == max_y - 1:
        sw = None
    else:
        sw = matrix[x + 1, y - 1]

    # Position W
    if x == 0:
        w = None
    else:
        w = matrix[x, y - 1]

    # Position NW
    if x == 0 or y == 0:
        nw = None
    else:
        nw = matrix[x - 1, y - 1]

    return n, ne, e, se, s, sw, w, nw


def get_possible_neighbour_cords(pos_x, pos_y) -> tuple:
    """
    Returns a tuple with coordinates for
    N, NE, E, SE, S, SW, W, NW neighbouring cells.
    """
    n = (pos_x, pos_y - 1)
    ne = (pos_x + 1, pos_y - 1)
    e = (pos_x + 1, pos_y)
    se = (pos_x + 1, pos_y + 1)
    s = (pos_x, pos_y + 1)
    sw = (pos_x - 1, pos_y + 1)
    w = (pos_x - 1, pos_y)
    nw = (pos_x - 1, pos_y - 1)

    return n, ne, e, se, s, sw, w, nw
