from gym.envs.registration import register

from gym_maze.common import MAZE_PATH as PATH_MAPPING  # noqa: F401
from gym_maze.common import MAZE_REWARD as REWARD_MAPPING  # noqa: F401
from gym_maze.common import MAZE_WALL as WALL_MAPPING  # noqa: F401
from gym_maze.maze import Maze  # noqa: F401

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
