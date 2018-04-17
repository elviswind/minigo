from collections import namedtuple
import numpy as np
import pandas

df = pandas.read_csv('d.csv', index_col=0)
d = df.as_matrix().astype(np.float32)
d = np.concatenate([d, d * -1], axis=0)
names = df.index
names = list(names) + ['-1' + n for n in names]
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
            if a < M / 2:
                legal_moves[int(a + M / 2)] = 0
            elif a >= M / 2:
                legal_moves[int(a - M / 2)] = 0

        if len(arr) < 5:
            legal_moves[-1] = 0

        forex = 66
        for a in arr:
            if a < forex:
                b = names[a][:3]
                c = names[a][-3:]
                for i in range(forex):
                    if b in names[i] or c in names[i]:
                        legal_moves[i] = 0
                        legal_moves[int(i + M / 2)] = 0
            elif a < forex + M / 2 and a >= (M - 1) / 2:
                b = names[a][2:5]
                c = names[a][-3:]
                for i in range(forex):
                    if b in names[int(i + M / 2)] or c in names[int(i + M / 2)]:
                        legal_moves[i] = 0
                        legal_moves[int(i + M / 2)] = 0

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
        score = 2 * (0.3 - np.std(self.board))
        if score < -1:
            return -1
        elif score > 1:
            return 1
        return score

    def report(self):
        score = self.score()
        dump = str(np.array(names)[np.where(self.selected[:-1] == 1)])
        if score > 0.2:
            with open('log.txt', 'a') as log:
                log.write(str(score) + ' ---- ' + dump + '\n')
        print(dump)


class IllegalMove(Exception):
    pass
