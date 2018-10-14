import numpy as np

WHITE = 0
BLACK = 1


class CheckerboardSimulator:

    def __init__(self, dim: int, ndiv: int) -> None:
        """
        :param n: dimensionality of solution space
        :param nd: division of each dimension,
        should be odd for colors to be alternating
        """
        self.n = dim
        self.nd = ndiv
        self.board = self._build_board()

    def _build_board(self) -> np.ndarray:
        x = np.empty(pow(self.nd, self.n), dtype=np.bool)

        # alternating cell colors
        x[:] = WHITE
        x[::2] = BLACK

        # reshape back to the original dimension
        return np.reshape(x, (self.nd,) * self.n)

    def get_color(self, *cords) -> int:
        """
        :param cords: floating point coords
        :return: integer representing color
        """
        indices = [(self._get_index(cord),) for cord in cords]
        return self.board[tuple(indices)][0]

    def _get_index(self, val: float) -> int:
        y = np.linspace(0, 1, self.nd + 1)
        return np.where(y <= val)[0][-1]
