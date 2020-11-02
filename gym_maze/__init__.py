from gym.envs.registration import register

# noinspection PyUnresolvedReferences
from .maze import Maze  # noqa: F401
from .maze import PATH_MAPPING, WALL_MAPPING, REWARD_MAPPING  # noqa: F401

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


register(
    id='MazeF1-v0',
    entry_point='gym_maze.envs:MazeF1',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MazeF2-v0',
    entry_point='gym_maze.envs:MazeF2',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MazeF3-v0',
    entry_point='gym_maze.envs:MazeF3',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MazeF4-v0',
    entry_point='gym_maze.envs:MazeF4',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='Maze4-v0',
    entry_point='gym_maze.envs:Maze4',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='Maze5-v0',
    entry_point='gym_maze.envs:Maze5',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='Maze6-v0',
    entry_point='gym_maze.envs:Maze6',
    max_episode_steps=50,
    nondeterministic=True
)

register(
    id='MazeT2-v0',
    entry_point='gym_maze.envs:MazeT2',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MazeT3-v0',
    entry_point='gym_maze.envs:MazeT3',
    max_episode_steps=50,
    nondeterministic=False
)

register(
    id='MazeT4-v0',
    entry_point='gym_maze.envs:MazeT4',
    max_episode_steps=50,
    nondeterministic=True
)
