import gym
from gym.spaces import Discrete


class FiniteStateWorld(gym.Env):

    def __init__(self, size) -> None:
        self.pos = None
        self.size = size
        self.states = size * 2 + 1

        self.observation_space = Discrete(1)
        self.action_space = Discrete(2)

    def reset(self):
        self.pos = 0
        return self.pos

    def step(self, action):
        if self.pos >= self.size:
            self.pos = self.pos - self.size + 1
        else:
            best_action = self.pos % 2  # alternating shortest action to goal

            if action == best_action:
                self.pos += 1
            else:
                self.pos += self.size

        if self.pos == self.size:
            return self.pos, 100, True, None

        return self.pos, 0, False, None

    def render(self, mode='human'):
        return self.pos

