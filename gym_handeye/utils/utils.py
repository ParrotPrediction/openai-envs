import gym_handeye as he
from gym_handeye.handeye_simulator import SURFACE, BLOCK, GRIPPER, BLOCK_NOT_UNDER_GRIPPER, BLOCK_UNDER_GRIPPER, BLOCK_IN_HAND


def get_all_possible_transitions(grid_size):
    """
    Returns all possible transitions of environment
    This information is used to calculate the agent's knowledge
    :param grid_size: size of grid
    :return: all transitions
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
                    add_transitions(grid_size, start, states)
            else:
                start[i] = GRIPPER
                start[j] = BLOCK
                add_transitions(grid_size, start, states)

    return states


def add_transitions(grid_size, start, states):
    actions = he.handeye.HandEye.get_all_possible_actions()
    mock_handeye = he.HandEyeSimulator(grid_size, True, False)
    mock_handeye.parse_observation(start)

    for action in actions:
        mock_handeye.take_action(action)
        end = mock_handeye.observe()

        if start != end:
            states.append((tuple(start), action, tuple(end)))

        mock_handeye.parse_observation(start)
    return


if __name__ == "__main__":
    print(len(get_all_possible_transitions(2)))
    print(len(get_all_possible_transitions(3)))
    print(len(get_all_possible_transitions(4)))
    print(len(get_all_possible_transitions(5)))
