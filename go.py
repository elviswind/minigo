from collections import namedtuple
import numpy as np
import pandas

df = pandas.read_csv('d.csv', index_col=0)

a = df.iloc[:, :-1]
b = df.iloc[:, 1:]
b.columns = a.columns = range(df.shape[1] - 1)
r = (b - a) / a
p = 1 / np.linalg.norm(r, axis=1)
r = (r.T * p).T.as_matrix()
l = r.shape[0]
cor = np.ones([l, l])
for i in range(l):
    for j in range(i + 1, l):
        s = (r[i] * r[j]).sum()
        cor[i][j] = s
        cor[j][i] = s

d = (df.as_matrix().T * p).T.astype(np.float32)
INIT_BOARD = np.zeros(d.shape[1]).astype(np.float32)

names = list(df.index)
M = len(d) + 1
N = d.shape[1]

MAX = 6
MIN = 4


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
    def __init__(self, board=None, select=None, n=0, remain=MAX, recent=None):
        self.board = board if board is not None else np.copy(INIT_BOARD)
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

        if len(arr) < MIN:
            legal_moves[-1] = 0

        return legal_moves

    def play_move(self, choice):
        s = np.copy(self.selected)
        if (self.all_legal_moves()[choice] == 0):
            print('playing bad move')
        s[choice] = 1
        if choice == M - 1:
            return Position(self.board, s, self.n + 1, 0, self.recent + [choice])

        return Position(self.board + d[choice], s, self.n + 1, self.remain - 1, self.recent + [choice])

    def is_game_over(self):
        return self.remain == 0 or len(np.where(self.all_legal_moves() == 1)[0]) == 0

    def score(self):
        if self.remain > MAX - MIN:
            return -1

        s = self.board / self.n
        m = s.mean()
        s = s / m
        x = np.arange(len(s)) + 1
        a, b = np.polyfit(x, np.log(s), 1)
        y = np.exp(a * x + b)

        # reg times
        last = s[0] < y[0]
        reg = 0
        regDist = [0]
        for i in range(1, len(s)):
            if (s[i] < y[i]) != last:
                regDist.append(i)
                reg += 1
                last = s[i] < y[i]

        # reg distribution
        regDist = np.array(regDist)
        regStd = (regDist[1:] - regDist[:-1]).std()

        score = 2 - regStd
        print(reg, score)

        if reg <= 20:
            score = -1

        if score > 1:
            score = 1
        elif score < -1:
            score = -1

        return score

    def report(self):
        score = self.score()
        dump = str(np.array(names)[np.where(self.selected[:-1] == 1)])
        if score > 0.5:
            with open('log.txt', 'a') as log:
                log.write(str(score) + ' ---- ' + dump + '\n')
        print(dump)


class IllegalMove(Exception):
    pass
