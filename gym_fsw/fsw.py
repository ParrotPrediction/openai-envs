import gym
from gym.spaces import Discrete


class FiniteStateWorld(gym.Env):

    def __init__(self, size) -> None:
        self.pos = None
        self.size = size
        self.states = size * 2 + 1

        self.observation_space = Discrete(self.states)
        self.action_space = Discrete(2)

    def reset(self):
        self.pos = 0
        return self._observation

    def step(self, action):
        if self.pos < self.size:
            best_action = self.pos % 2
            if best_action == action:
                self.pos += 1
                if self.pos == self.size:
                    self.pos = self.size * 2
            else:
                self.pos += self.size
        else:
            self.pos = self.pos - self.size + 1
            if self.pos == self.size:
                self.pos = self.size * 2

        if self.pos == self.size * 2:
            return self._observation, 100, True, None

        return self._observation, 0, False, None

    def render(self, mode='human'):
        return self._observation

    @property
    def _observation(self):
        return str(self.pos)

    def state_action(self):
        """
        Return states and possible actions in each of them
        """
        mapping = {}
        for p in range(0, self.states):
            mapping[p] = [0, 1]

        # Final state - no actions
        mapping[self.states - 1] = []

        # Cast int key str
        mapping = {str(k): v for k, v in mapping.items()}

        return mapping
