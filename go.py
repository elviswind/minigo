from collections import namedtuple
import numpy as np
import pandas

df = pandas.read_csv('d.csv', index_col=0)
d = df.as_matrix().astype(np.float32)
ndf = np.ones((df.shape[0], df.shape[1] + 1)).astype(np.float32)
for i in range(1, ndf.shape[1]):
    ndf[:, i] = ndf[:, i - 1] * (df[str(i - 1)] + 1)

# d = np.concatenate([d, d * -1], axis=0)
names = df.index
# names = list(names) + ['-1' + n for n in names]

l = d.shape[0]
cor = np.ones([l, l])
for i in range(l):
    for j in range(i + 1, l):
        s = (d[i] * d[j]).sum()
        cor[i][j] = s
        cor[j][i] = s

d = ndf
M = len(d) + 1
N = d.shape[1]
EMPTY_BOARD = np.zeros([N], dtype=np.float32)


class PositionWithContext(namedtuple('SgfPosition', ['position', 'next_move', 'result'])):
    pass


def replay_position(position, result):
    assert position.n == len(position.recent), "Position history is incomplete"
    pos = Position()
    for player_move in position.recent:
        choice = player_move
        yield PositionWithContext(pos, choice, result)
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
        legal_moves = np.ones([M], dtype=np.int8)
        arr = np.where(self.selected[:-1] == 1)[0]
        if len(arr) > 0:
            legal_moves[:arr[-1] + 1] = 0

        for a in arr:
            legal_moves[list(np.abs(cor[a]) > 0.75) + [False]] = 0

        if len(arr) < 5:
            legal_moves[-1] = 0

        return legal_moves

    def play_move(self, choice):
        s = np.copy(self.selected)
        s[choice] = 1
        if choice == M - 1:
            return Position(self.board, s, self.n + 1, 0, self.recent + [choice])

        return Position(self.board + d[choice], s, self.n + 1, self.remain - 1, self.recent + [choice])

    def is_game_over(self):
        return self.remain == 0 or len(np.where(self.all_legal_moves() == 1)[0]) == 0

    def score(self):
        if self.remain > 5:
            return -1

        x = np.arange(N)
        s = self.board / self.board[0]
        (a, b) = np.polyfit(x, s, 1)
        y = x * a + b
        mse = ((y - s) ** 2).sum() / N
        score = 1 - mse * 100
        if score < -1:
            return -1
        elif score > 1:
            return 1
        return score

    def report(self):
        score = self.score()
        dump = str(np.array(names)[np.where(self.selected[:-1] == 1)])
        if score > 0.8:
            with open('log.txt', 'a') as log:
                log.write(str(score) + ' ---- ' + dump + '\n')
        print(dump)


class IllegalMove(Exception):
    pass
