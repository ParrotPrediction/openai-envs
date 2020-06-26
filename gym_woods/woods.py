import numpy as np


class Woods:

    def __init__(self, matrix):
        self.matrix = matrix
        self.max_x = self.matrix.shape[1]
        self.max_y = self.matrix.shape[0]

        # Because some boards might be toroidal (no boundaries) they will be
        # duplicated into 3x3 grid with the original being in center
        self.big_board = np.empty(
            (self.max_y * 3, self.max_x * 3), dtype=np.str)

        # place each original row in big_board
        for row_idx in range(0, self.max_y):
            original_row = self.matrix[row_idx, :]
            extended_original_row = np.concatenate([original_row] * 3)

            self.big_board[row_idx, :] = extended_original_row
            self.big_board[row_idx + 3, :] = extended_original_row
            self.big_board[row_idx + 6, :] = extended_original_row

    @property
    def possible_insertion_cords(self):
        y_idx, x_idx = np.where(self.matrix == '.')
        return tuple(zip(x_idx, y_idx))

    def perception(self, x, y):
        if not 0 <= x < self.max_x:
            raise ValueError('X position not within allowed range')

        if not 0 <= y < self.max_y:
            raise ValueError('Y position not within allowed range')

        # translate x, y => x = x+max_x, y = y+max_y
        x += self.max_x
        y += self.max_y

        n = self.big_board[y - 1, x]
        ne = self.big_board[y - 1, x + 1]
        e = self.big_board[y, x + 1]
        se = self.big_board[y + 1, x + 1]
        s = self.big_board[y + 1, x]
        sw = self.big_board[y + 1, x - 1]
        w = self.big_board[y, x - 1]
        nw = self.big_board[y - 1, x - 1]

        return list(map(str, [n, ne, e, se, s, sw, w, nw]))

    def is_reward(self, x, y):
        return self.matrix[y, x] in ('F', 'G')
