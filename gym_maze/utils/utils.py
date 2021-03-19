from typing import Tuple

import numpy as np
import networkx as nx

from gym_maze.common import MAZE_PATH, MAZE_REWARD
from gym_maze.common.maze_utils import get_possible_neighbour_cords
from gym_maze.internal.maze_impl import find_action_by_direction


def get_all_possible_transitions(maze):
    """
    Returns all possible transitions within the maze.
    [POINT]->[ACTION]->[POINT]
    This information is used to calculate the agent's knowledge
    :param maze: an instance of the maze
    :return:
    """
    transitions = []

    g = _create_graph(maze)

    path_nodes = (node for node, data
                  in g.nodes(data=True) if data['type'] == 'path')

    for node in path_nodes:
        for neighbour in nx.all_neighbors(g, node):
            direction = distinguish_direction(node, neighbour)
            action = find_action_by_direction(direction)

            transitions.append((node, action, neighbour))

    return transitions


def _create_graph(env):
    matrix = env.maze.matrix

    # Create uni-directed graph
    g = nx.Graph()

    # Add nodes
    for path in np.argwhere(matrix == MAZE_PATH):
        g.add_node(tuple(path), type='path')

    for reward in np.argwhere(matrix == MAZE_REWARD):
        g.add_node(tuple(reward), type='reward')

    # Add edges
    path_nodes = [cords for cords, attribs
                  in g.nodes(data=True) if attribs['type'] == 'path']

    for n in path_nodes:
        neighbour_cells = get_possible_neighbour_cords(*n)
        allowed_cells = [c for c in neighbour_cells
                         if matrix[c] == MAZE_PATH or matrix[c] == MAZE_REWARD]
        edges = [(n, dest) for dest in allowed_cells]

        g.add_edges_from(edges)

    return g


def distinguish_direction(start: Tuple[int, int], end: Tuple[int, int]):
    direction = ''

    if end[1] + 1 == start[1]:
        direction += 'N'

    if end[0] - 1 == start[0]:
        direction += 'S'

    if end[1] - 1 == start[1]:
        direction += 'W'

    if end[0] + 1 == start[0]:
        direction += 'E'

    return direction
