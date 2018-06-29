import pandas
import numpy as np


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
N = d.shape[1]
M = d.shape[0] + 1


def find_eligible(got):
    if len(got) == 0: return list(range(d.shape[0])) + [d.shape[0]]
    origin = eligible[got[0]]
    for i in range(1, len(got)):
        origin = np.logical_and(eligible[got[i]], origin)
    return np.where(origin)[0].tolist() + [d.shape[0]]


def factorial_random(got, n, network):
    if len(got) < n:
        fine = find_eligible(got)
        print(fine)
        p = get_p(network, got)[fine]
        i = np.random.choice(fine, 1, p=p / p.sum())[0]
        return factorial_random(got + [i], n, network)
    else:
        return got


def test_choice(choice):
    s = d[choice].sum(axis=0)
    s /= len(choice)

    x = np.arange(len(s)) + 1
    a, b = np.polyfit(x, np.log(s), 1)
    y = np.exp(a * x + b)

    loss = (((y - s) / s) ** 2).sum()

    return [sorted(choice), a, loss, a / loss]


def random_test(network):
    gathered = []
    while len(gathered) < 10:
        choice = factorial_random([], 10, network)
        gathered.append(test_choice(choice))
    output = np.array(sorted(gathered, key=lambda x: x[3]))
    return output


def get_p(network, got):
    wave = np.zeros([d.shape[1], 1], dtype=np.float32)
    if len(got) > 0:
        wave = np.reshape(d[got].sum(axis=0) / len(got), (d.shape[1], 1)).astype(np.float32)

    p, _ = network.run(wave)
    return p


def play(network):
    random_test(network)
