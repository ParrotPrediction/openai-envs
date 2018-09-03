import gym_handeye as he


# TODO: possibly change into something different? (read articles)
def get_all_possible_transitions(grid_size):
    """
    Returns all possible transitions of environment
    This information is used to calculate the agent's knowledge
    :param grid_size: size of grid
    :return: all transitions
    """
    env_size = grid_size * grid_size + 1
    states = []
    actions = he.handeye.HandEye.get_all_possible_actions()

    for i in range(env_size - 1):
        for j in range(env_size - 1):
            start = ['w' for x in range(env_size - 1)]
            start.append('0')
            if i == j:
                for k in range(2):
                    if k == 0:
                        start[i] = 'g'
                        start[env_size - 1] = '1'
                    else:
                        start[i] = 'b'
                        start[env_size - 1] = '2'
            else:
                start[i] = 'g'
                start[j] = 'b'

            mock_handeye = he.HandEyeSimulator(grid_size, True, False)
            mock_handeye.parse_observation(start)

            for action in actions:
                mock_handeye.take_action(action)
                end = mock_handeye.observe()

                # if start != end:
                states.append((tuple(start), action, tuple(end)))

                mock_handeye.parse_observation(start)

    return states


if __name__ == "__main__":
    print(get_all_possible_transitions(2))
