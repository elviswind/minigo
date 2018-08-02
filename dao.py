import pandas
import numpy as np
import utils
import time
import os


def get_d():
    base = 10
    threshold = 0.6

    df = pandas.read_csv('d.csv', index_col=0)
    # df = df.filter(like='ETF', axis=0)
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

    return (np.concatenate((np.ones([r.shape[0], 1]) * base,
                            np.add.accumulate(r, axis=1) + np.ones(r.shape) * base), axis=1),
            correlation)


d, eligible = get_d()
d = d.astype(np.float32)
N = d.shape[1]
M = d.shape[0] + 1
MAX = 10
POOL_SIZE = 5000
black_list = [83, 923]


def find_eligible(got):
    if len(got) == 0: return list(range(d.shape[0]))
    origin = eligible[got[0]]
    for i in range(1, len(got)):
        origin = np.logical_and(eligible[got[i]], origin)
    ret = np.where(origin)[0].tolist() + [d.shape[0]]
    return [i for i in ret if i not in black_list]


def factorial_random(gots, network, repeat):
    if gots is None or len(gots) == 0:
        gots = []
        f = find_eligible([])
        p = get_probabilities(network, [[]])[0][f]
        for j in range(repeat):
            c = np.random.choice(f, 1, p=p / p.sum())[0]
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
            c = np.random.choice(fines[i], 1, p=p / p.sum())[0]
            nextGots.append(toContinue[i] + [c])

    return factorial_random(toKeep + nextGots, network, repeat)


def test_choice(choice):
    s = d[choice].sum(axis=0)
    s /= len(choice)

    x = np.arange(len(s)) + 1
    a, b = np.polyfit(x, np.log(s), 1)
    y = np.exp(a * x + b)

    loss = (((y - s) / s) ** 2).sum()

    return [sorted(choice), a / loss, a, loss]


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
    import preprocessing
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
        wave = np.zeros([N, 1], dtype=np.float32)
        if len(choice) > 0:
            wave = np.reshape(d[choice].sum(axis=0) / len(choice), (N, 1)).astype(np.float32)
        waves.append(wave)

    p, _ = network.run_many(waves)
    return p


def play(network, output_dir):
    for x in range(10):
        lasttime = []
        if os.path.exists('lasttime.npy'):
            lasttime = np.load('lasttime.npy').tolist()
            print("best of last time ", list(map(lambda x: x[1], lasttime[0:3])))
            print("worst of last time ", list(map(lambda x: x[1], lasttime[-3:])))

        thistime = random_test(network, 2, POOL_SIZE * 2)
        print("best of this time ", list(map(lambda x: x[1], thistime[0:3])))
        print("worst of this time ", list(map(lambda x: x[1], thistime[-3:])))

        records = sorted(thistime + lasttime, key=lambda x: x[1], reverse=True)

        # remove duplicates
        tmp = set()
        output = []
        for record in records:
            key = str(record[0])
            if key not in tmp:
                banned = [i for i in record[0] if i in black_list]
                if len(banned) == 0:
                    output.append(record)
                tmp.add(key)
            if len(output) >= POOL_SIZE:
                break

        np.save('lasttime.npy', np.array(output))

    with utils.logged_timer("start making example"):
        make_examples(records, output_dir)


def test_make_examples():
    lasttime = np.load('lasttime.npy').tolist()
    make_examples(lasttime, 'temp')
