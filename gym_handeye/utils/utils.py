import gym_handeye as he
from gym_handeye.handeye_simulator import SURFACE, BLOCK, GRIPPER, \
    BLOCK_NOT_UNDER_GRIPPER, BLOCK_UNDER_GRIPPER, \
    BLOCK_IN_HAND


def get_all_possible_transitions(grid_size):
    """
    Returns all possible transitions of environment
    This information is used to calculate the agent's knowledge
    :param grid_size: size of the grid
    :return: all transitions as a list of tuples:
    (start_state, action, end_state)
    """
    env_size = grid_size * grid_size + 1
    states = []

    for i in range(env_size - 1):
        for j in range(env_size - 1):
            start = [SURFACE for x in range(env_size - 1)]
            start.append(BLOCK_NOT_UNDER_GRIPPER)
            if i == j:
                for k in range(2):
                    if k == 0:
                        start[i] = GRIPPER
                        start[env_size - 1] = BLOCK_UNDER_GRIPPER
                    else:
                        start[i] = BLOCK
                        start[env_size - 1] = BLOCK_IN_HAND
                    states.extend(get_transitions(grid_size, start))
            else:
                start[i] = GRIPPER
                start[j] = BLOCK
                states.extend(get_transitions(grid_size, start))

    return states


def get_transitions(grid_size, start):
    """
    Returns transitions for specified start position.
    :param grid_size: size of the grid
    :param start: start state of the transition
    :return: transitions as a list of tuples: (start_state, action, end_state)
    """
    states = []
    actions = he.handeye.HandEye.get_all_possible_actions()
    mock_handeye = he.HandEyeSimulator(grid_size, True)
    mock_handeye.parse_observation(start)

    for action in actions:
        mock_handeye.take_action(action)
        end = mock_handeye.observe()

        if start != end:
            states.append((tuple(start), action, tuple(end)))

        mock_handeye.parse_observation(start)
    return states
