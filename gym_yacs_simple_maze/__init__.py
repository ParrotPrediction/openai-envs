from gym.envs.registration import register
from gym_yacs_simple_maze.maze import SimpleMaze  # noqa: F401

register(
    id='SimpleMaze-v0',
    entry_point='gym_yacs_simple_maze.maze:SimpleMaze',
    max_episode_steps=50,
    nondeterministic=False
)
