from gym.envs.registration import register

register(
    id='orthogonal-single-boundary-v0',
    entry_point='gym_real_value_toy.orthogonal:XAxisBoundaryEnv',
    kwargs={'threshold_x': .5}
)
