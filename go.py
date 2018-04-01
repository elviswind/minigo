from collections import namedtuple
import numpy as np
import pandas

N = 30
EMPTY_BOARD = np.zeros([N], dtype=np.float32)
df = pandas.read_csv('d.csv', index_col=0)
d = df.iloc[:, :N].as_matrix().astype(np.float32)
d = np.concatenate([d, d * -1], axis=0)
names = df.index
names = list(names) + ['-1' + n for n in names]
M = len(d)


class PositionWithContext(namedtuple('SgfPosition', ['position', 'next_move', 'result'])):
    pass


def replay_position(position):
    assert position.n == len(position.recent), "Position history is incomplete"
    pos = Position()
    for player_move in position.recent:
        choice = player_move
        yield PositionWithContext(pos, choice, position.score())
        pos = pos.play_move(choice)


class Position():
    def __init__(self, board=None, select=None, n=0, remain=10, recent=None):
        self.board = board if board is not None else np.copy(EMPTY_BOARD)
        self.selected = select if select is not None else np.zeros([M], dtype=np.float64)
        self.remain = remain
        self.n = n
        self.recent = recent if recent is not None else []

    def __str__(self, colors=True):
        return str(np.linalg.norm(self.board)) + '-----' + str(self.remain) + '-----' + str(
            self.selected) + '-----' + str(self.board)

    def all_legal_moves(self):
        # by default, can choose every one
        legal_moves = np.ones([M], dtype=np.int8)
        arr = np.where(self.selected == 1)[0]
        if len(arr) > 0:
            legal_moves[:arr[-1] + 1] = 0
        return legal_moves

    def play_move(self, choice):
        s = np.copy(self.selected)
        s[choice] = 1
        return Position(self.board + d[choice], s, self.n + 1, self.remain - 1, self.recent + [choice])

    def is_game_over(self):
        return self.remain == 0

    def score(self):
        return 1 - np.linalg.norm(self.board)
