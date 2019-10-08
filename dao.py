import pandas
import numpy as np
import time
import os
from collections import namedtuple
from . import utils
from . import preprocessing

MAX = 10
POOL_SIZE = 5000
LOOPS = 20
BLACK_LIST = []
PREFER_LIST = []

threshold = 0.3
N = 127
M = 800
START_BOARD = np.zeros([N, 1], dtype=np.float32)
d = None
eligible = None

def get_dpr(df):
    a = df.iloc[:, :-1]
    b = df.iloc[:, 1:]
    b.columns = a.columns = range(df.shape[1] - 1)
    r = (b - a) / a
    p = 1 / np.linalg.norm(r, axis=1)
    r = (r.T * p).T.values
    l = r.shape[0]

    correlation = np.ones([l, l])
    for i in range(l):
        for j in range(i + 1, l):
            s = (r[i] * r[j]).sum()
            correlation[i][j] = s
            correlation[j][i] = s
    correlation[np.abs(correlation) > threshold] = 0
    correlation[np.logical_and(np.abs(correlation) <= threshold, correlation != 0)] = 1

    return (np.concatenate((np.ones([r.shape[0], 1]) * MAX,
                            np.add.accumulate(r, axis=1) + np.ones(r.shape) * MAX), axis=1),p,r,
            correlation)

def get_d(path):
    df = pandas.read_csv(path, index_col=0)
    a = df.iloc[:, :-1]
    b = df.iloc[:, 1:]
    b.columns = a.columns = range(df.shape[1] - 1)
    r = (b - a) / a
    p = 1 / np.linalg.norm(r, axis=1)
    r = (r.T * p).T.values
    l = r.shape[0]

    correlation = np.ones([l, l])
    for i in range(l):
        for j in range(i + 1, l):
            s = (r[i] * r[j]).sum()
            correlation[i][j] = s
            correlation[j][i] = s
    correlation[np.abs(correlation) > threshold] = 0
    correlation[np.logical_and(np.abs(correlation) <= threshold, correlation != 0)] = 1

    return (np.concatenate((np.ones([r.shape[0], 1]) * MAX,
                            np.add.accumulate(r, axis=1) + np.ones(r.shape) * MAX), axis=1),
            correlation)

def init(path):
    global N
    global M
    global START_BOARD
    global d, eligible
    d, eligible = get_d(path)
    d = d.astype(np.float32)

    N = d.shape[1]
    M = d.shape[0] + 1
    START_BOARD = np.zeros([N, 1], dtype=np.float32)

init('d.csv')

class IllegalMove(Exception):
    pass


class PositionForTraining(namedtuple('SgfPosition', ['position', 'next_move', 'result'])):
    pass


class Position():
    def __init__(self):
        self.recent = []
        self.to_play = 1
        self.n = 0

    def replicate(self):
        new_pos = Position()
        new_pos.recent = self.recent.copy()
        new_pos.n = self.n
        return new_pos

    def all_legal_moves(self):
        legal_moves = np.zeros([M], dtype=np.int8)
        if (M - 1) in self.recent:
            return legal_moves
        good = find_eligible(self.recent)
        legal_moves[good] = 1
        return legal_moves

    def get_state(self):
        if len(self.recent) == 0:
            return START_BOARD
        else:
            return np.reshape(d[self.recent].sum(axis=0) / len(self.recent), (N, 1)).astype(np.float32)

    def play_move(self, a):
        new_pos = self.replicate()
        legal = self.all_legal_moves()
        if legal[a] == 0:
            raise IllegalMove("Move at {} not allowed\n".format(a))

        new_pos.recent.append(a)
        new_pos.n += 1
        return new_pos

    def is_game_over(self):
        return len(self.recent) >= MAX or (M - 1) in self.recent

    def score(self):
        result = test_choice([i for i in self.recent if i != M - 1])
        return result[1]

    def result(self):
        score = self.score()
        return score

    def result_string(self):
        return 'score'

    def describe(self):
        result = test_choice([i for i in self.recent if i != M - 1])
        return str(result)


def replay_position(position, result):
    assert position.n == len(position.recent), "Position history is incomplete"
    pos = Position()
    for next_move in position.recent:
        yield PositionForTraining(pos, next_move, result)
        pos = pos.play_move(next_move)


def find_eligible(got):
    if len(got) == 0: return list(range(d.shape[0]))
    origin = eligible[got[0]]
    for i in range(1, len(got)):
        origin = np.logical_and(eligible[got[i]], origin)
    ret = np.where(origin)[0].tolist() + [d.shape[0]]
    return [i for i in ret if i not in BLACK_LIST]


def add_probability_mod(p, l):
    #p = np.log(p + 1.0 + 1 / (np.power(100, l) + 100))
    if len(PREFER_LIST) > 0:
        mod = np.ones(len(p))
        mod[PREFER_LIST] = 10
        mod = mod / mod.sum()
        p = p + 0.2 * mod
    p = p / p.sum()
    return p


def factorial_random(gots, network, repeat):
    if gots is None or len(gots) == 0:
        gots = []
        f = find_eligible([])
        p = get_probabilities(network, [[]])[0][f]
        p = p / p.sum()
        for j in range(repeat):
            p = add_probability_mod(p, 0)
            c = np.random.choice(f, 1, p=p)[0]
            gots.append([c])

    toContinue = []
    toKeep = []
    for got in gots:
        if M - 1 in got or len(got) == MAX:
            toKeep.append(got)
        else:
            toContinue.append(got)

    if len(toContinue) == 0:
        return toKeep

    fines = []
    for got in toContinue:
        fines.append(find_eligible(got))

    ps = get_probabilities(network, toContinue)
    nextGots = []
    for i in range(len(toContinue)):
        p = ps[i][fines[i]]
        for j in range(repeat):
            p = add_probability_mod(p, len(toContinue[i]))
            c = np.random.choice(fines[i], 1, p=p)[0]
            nextGots.append(toContinue[i] + [c])

    return factorial_random(toKeep + nextGots, network, repeat)

def findDrop(s, debug=True):
    drop = 1 - s / np.maximum.accumulate(s)
    i = np.argmax(drop) # end of the period
    if i == 0: return 0,0
    j = np.argmax(s[:i]) # start of period
    jump = 1
    while i + jump < len(s):
        if s[i + jump] >= s[j]:
            break
        jump += 1
        
    return drop[i] * 100, drop.mean()
    
def stat(s):
    drop, meandrop = findDrop(s, False)
    x = np.arange(s.shape[0])
    a, b = np.polyfit(x, s, 1)
    y = b + a * x
    fitloss = np.abs(s / y - 1).max()
    #fitloss = (np.power(s - y, 2)).sum() / y.shape[0]
    return drop, meandrop, a, fitloss
    
def test_choice(choice):
    s = d[choice].sum(axis=0)
    s /= len(choice)

    drop, meandrop, a, fitloss = stat(s)

    return [sorted(choice), 1 - drop, drop, meandrop, a, fitloss]


def random_test(network, repeat, max):
    gathered = []
    with utils.logged_timer("start selfplay"):
        while len(gathered) < max:
            choices = factorial_random(None, network, repeat)
            for choice in choices:
                if d.shape[0] in choice:
                    choice.remove(d.shape[0])
                gathered.append(test_choice(choice))

    records = sorted(gathered, key=lambda x: x[1], reverse=True)[:POOL_SIZE]
    return records


def make_examples(results, output_dir):
    tf_examples = []
    with utils.logged_timer("start making example 2"):
        dict = {}
        vict = {}

        # count frequency
        frequency = {}
        for result in results:
            for x in result[0]:
                if x in frequency:
                    frequency[x] += 1
                else:
                    frequency[x] = 1

        for result in results:
            record = result[0]
            record.sort(key=lambda x: frequency[x], reverse=True)
            v = result[1] - 0.2
            for i in range(len(record)):
                key = str(record[:i])

                if key not in dict:
                    dict[key] = []

                if len(record) < 10 and i + 1 == len(record):
                    dict[key].append(M - 1)
                else:
                    dict[key].append(record[i])

                if key not in vict:
                    vict[key] = v
                elif v > vict[key]:
                    vict[key] = v

        for key in dict:
            got = eval(key)
            dict_temp = {}
            for i in dict[key]:
                if i in dict_temp:
                    dict_temp[i] += 1
                else:
                    dict_temp[i] = 1

            pi = np.zeros(M).astype(np.float32)
            for i in dict_temp:
                pi[i] = dict_temp[i]

            # pi = (pi / pi.max() - 0.5) * 2
            pi = pi / pi.sum()
            position = np.zeros(N).astype(np.float32)
            if len(got) > 0:
                position = d[got].sum(axis=0)
            tf_examples.append(preprocessing.make_tf_example(np.expand_dims(position, axis=2), pi, vict[key]))

    output_name = '{}-{}'.format(int(time.time()), 1)
    fname = os.path.join(output_dir, "{}.tfrecord.zz".format(output_name))
    preprocessing.write_tf_examples(fname, tf_examples)


def get_probabilities(network, choices):
    waves = []
    for choice in choices:
        wave = START_BOARD
        if len(choice) > 0:
            wave = np.reshape(d[choice].sum(axis=0) / len(choice), (N, 1)).astype(np.float32)
        waves.append(wave)

    p, _ = network.run_many(waves)
    # if np.isnan(p).any():
    #     return np.ones([M]) / M
    return p

def find_eligible_no_network(got):
    if len(got) == 0: return list(range(d.shape[0]))
    origin = eligible[got[0]]
    for i in range(1, len(got)):
        origin = np.logical_and(eligible[got[i]], origin)
    ret = np.where(origin)[0].tolist()
    return [i for i in ret if i not in BLACK_LIST]

def factorial_random_no_network(gots, repeat):
    if gots is None or len(gots) == 0:
        gots = []
        f = find_eligible_no_network([])
        for j in range(repeat):
            c = np.random.choice(f, 1)[0]
            gots.append([c])

    toContinue = []
    toKeep = []
    for got in gots:
        if len(got) == MAX:
            toKeep.append(got)
        else:
            toContinue.append(got)

    if len(toContinue) == 0:
        return toKeep

    fines = []
    for got in toContinue:
        fines.append(find_eligible_no_network(got))
        
    nextGots = []
    for i in range(len(toContinue)):
        for j in range(repeat):
            if len(fines[i]) > 0:
                c = np.random.choice(fines[i], 1)[0]
                nextGots.append(toContinue[i] + [c])

    return factorial_random_no_network(toKeep + nextGots, repeat)

def playByRandom(output_dir, times):
    lasttime = []
    if os.path.exists('lasttime.npy'):
        lasttime = np.load('lasttime.npy', allow_pickle=True).tolist()

    thistime = []
    for i in range(times):
        xs = factorial_random_no_network([], 2)
        for x in xs:
            x.sort()
            thistime.append(test_choice(x))
    thistime = pandas.DataFrame(thistime)
    thistime = thistime[thistime[2] < 2].sort_values(2).values
    print("this run ", ",".join(map(lambda x: str(x[1])[:5], thistime[0:3])),
            ",".join(map(lambda x: str(x[1])[:5], thistime[-3:])))

    i = 0
    j = 0
    tmp = set()
    output = []
    newfound = 0
    if len(lasttime) == 0:
        output = thistime
    else:
        while len(output) < POOL_SIZE:
            a = -10000
            b = -10000
            if i < len(thistime):
                a = thistime[i][1]
                a_key = str(thistime[i][0])
            if j < len(lasttime):
                b = lasttime[j][1]
                b_key = str(lasttime[j][0])

            if a > b and i < len(thistime):
                if a_key not in tmp and len([x for x in BLACK_LIST if x not in thistime[i][0]]) == len(BLACK_LIST):
                    tmp.add(a_key)
                    output.append(thistime[i])
                    newfound += 1
                i += 1
            elif a <= b and j < len(lasttime):
                if b_key not in tmp and len([x for x in BLACK_LIST if x not in lasttime[j][0]]) == len(BLACK_LIST):
                    tmp.add(b_key)
                    output.append(lasttime[j])
                j += 1
            else:
                break

    print("add {} new records ".format(newfound))
    print("after merge ", ",".join(map(lambda x: str(x[1])[:5], output[0:3])),
            ",".join(map(lambda x: str(x[1])[:5], output[-3:])))

    np.save('lasttime.npy', np.array(output))

    with utils.logged_timer("start making example"):
        make_examples(output, output_dir)


def play(network, output_dir):
    for x in range(LOOPS):
        lasttime = []
        if os.path.exists('lasttime.npy'):
            lasttime = np.load('lasttime.npy', allow_pickle=True).tolist()

        thistime = random_test(network, 2, POOL_SIZE * 2)
        print("this run ", ",".join(map(lambda x: str(x[1])[:5], thistime[0:3])),
              ",".join(map(lambda x: str(x[1])[:5], thistime[-3:])))

        i = 0
        j = 0
        tmp = set()
        output = []
        newfound = 0
        if len(lasttime) == 0:
            output = thistime
        else:
            while len(output) < POOL_SIZE:
                a = -10000
                b = -10000
                if i < len(thistime):
                    a = thistime[i][1]
                    a_key = str(thistime[i][0])
                if j < len(lasttime):
                    b = lasttime[j][1]
                    b_key = str(lasttime[j][0])

                if a > b and i < len(thistime):
                    if a_key not in tmp and len([x for x in BLACK_LIST if x not in thistime[i][0]]) == len(BLACK_LIST):
                        tmp.add(a_key)
                        output.append(thistime[i])
                        newfound += 1
                    i += 1
                elif a <= b and j < len(lasttime):
                    if b_key not in tmp and len([x for x in BLACK_LIST if x not in lasttime[j][0]]) == len(BLACK_LIST):
                        tmp.add(b_key)
                        output.append(lasttime[j])
                    j += 1
                else:
                    break

        print("add {} new records ".format(newfound))
        print("after merge ", ",".join(map(lambda x: str(x[1])[:5], output[0:3])),
              ",".join(map(lambda x: str(x[1])[:5], output[-3:])))

        np.save('lasttime.npy', np.array(output))

    with utils.logged_timer("start making example"):
        make_examples(output, output_dir)


def test_make_examples():
    lasttime = np.load('lasttime.npy', allow_pickle=True).tolist()
    make_examples(lasttime, 'temp')
