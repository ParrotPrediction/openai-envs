from gym_real_value_toy.toy_env import ToyEnv


class XAxisBoundaryEnv(ToyEnv):

    def __init__(self, threshold_x: float = 0.5) -> None:
        super().__init__()
        self.threshold_x = threshold_x

    @property
    def _correct_answer(self) -> bool:
        return self._state[0] < self.threshold_x
