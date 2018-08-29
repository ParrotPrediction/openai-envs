import sys
sys.path.append('/home/e-dzia/openai-envs')
from gym_handeye import HandEye


class MockHandEye(HandEye):
    def observe(self):
        observation = ['w' for x in range(self.env_size - 1)]
        observation.append('0')

        if self.grip_pos_x != self.block_pos_x and self.grip_pos_y != self.block_pos_y:
            observation[self.grip_pos_y * self.grid_size + self.grip_pos_x] = 'g'
            observation[self.block_pos_y * self.grid_size + self.block_pos_x] = 'b'
        else:
            if self.block_in_hand:
                observation[self.grip_pos_y * self.grid_size + self.grip_pos_x] = 'b'
                if self.note_in_hand:
                    observation[self.env_size - 1] = '2'
            else:
                observation[self.env_size - 1] = '1'
                observation[self.grip_pos_y * self.grid_size + self.grip_pos_x] = 'g'

        return observation

    def parse_observation(self, observation):
        self.block_pos_x = -1
        self.block_pos_y = -1
        self.grip_pos_x = -1
        self.grip_pos_y = -1
        self.block_in_hand = False
        for i, field in enumerate(observation):
            if field == 'b':
                self.block_pos_x = i % self.grid_size
                self.block_pos_y = int(i / self.grid_size)
            if field == 'g':
                self.grip_pos_x = i % self.grid_size
                self.grip_pos_y = int(i / self.grid_size)
        if self.grip_pos_x == -1:
            self.block_in_hand = True
            self.grip_pos_x = self.block_pos_x
            self.grip_pos_y = self.block_pos_y
        elif self.block_pos_x == -1:
            self.block_pos_x = self.grip_pos_x
            self.block_pos_y = self.grip_pos_y