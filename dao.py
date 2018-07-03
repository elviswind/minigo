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


def find_eligible(got):
    if len(got) == 0: return list(range(d.shape[0]))
    origin = eligible[got[0]]
    for i in range(1, len(got)):
        origin = np.logical_and(eligible[got[i]], origin)
    return np.where(origin)[0].tolist() + [d.shape[0]]


def factorial_random(got, network):
    if d.shape[0] in got or len(got) == 10:
        return got
    else:
        fine = find_eligible(got)
        p = get_p(network, got)[fine]
        i = np.random.choice(fine, 1, p=p / p.sum())[0]
        return factorial_random(got + [i], network)


def test_choice(choice):
    s = d[choice].sum(axis=0)
    s /= len(choice)

    x = np.arange(len(s)) + 1
    a, b = np.polyfit(x, np.log(s), 1)
    y = np.exp(a * x + b)

    loss = (((y - s) / s) ** 2).sum()

    return [sorted(choice), a, loss, a / loss]


def random_test(network, output_dir):
    gathered = []
    with utils.logged_timer("start selfplay"):
        while len(gathered) < 100:
            choice = factorial_random([], network)
            if d.shape[0] in choice:
                choice.remove(d.shape[0])
            gathered.append(test_choice(choice))
    records = np.array(sorted(gathered, key=lambda x: x[3]))
    pandas.DataFrame(records).to_csv('temp.csv')
    make_examples(records, output_dir)


def make_examples(results, output_dir):
    tf_examples = []
    import preprocessing
    for result in results:
        start = np.zeros(N).astype(np.float32)
        record = result[0]
        v = result[3]

        remain = 10
        for i in range(len(record)):
            pi = np.ones(M).astype(np.float32) * 0.1
            pi[record[i]] = 0.9
            pi = pi / pi.sum()
            tf_examples.append(preprocessing.make_tf_example(np.expand_dims(start, axis=2), pi, v))

            start = start + d[record[i]]
            remain -= 1

        if len(record) < 10:
            pi = np.ones(M).astype(np.float32) * 0.1
            pi[-1] = 0.9
            pi = pi / pi.sum()
            tf_examples.append(preprocessing.make_tf_example(np.expand_dims(start, axis=2), pi, v))

    output_name = '{}-{}'.format(int(time.time()), 1)
    fname = os.path.join(output_dir, "{}.tfrecord.zz".format(output_name))
    preprocessing.write_tf_examples(fname, tf_examples)


def get_p(network, got):
    wave = np.zeros([d.shape[1], 1], dtype=np.float32)
    if len(got) > 0:
        wave = np.reshape(d[got].sum(axis=0) / len(got), (d.shape[1], 1)).astype(np.float32)

    p, _ = network.run(wave)
    return p


def play(network, output_dir):
    random_test(network, output_dir)
