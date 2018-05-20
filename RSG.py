import pandas
import numpy as np

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

d = (df.as_matrix().astype(np.float32).T * p).T

import random


def findDist(n):
    data = []
    found = 0
    while found < 30:
        choice = random.sample(range(d.shape[0]), n)
        bad = True
        while bad:
            bad = False
            for k in range(n):
                for l in range(k):
                    if np.abs(cor[k][l]) > 0.75:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                choice = random.sample(range(d.shape[0]), n)

        s = d[choice].sum(axis=0)
        s = s / n
        m = s.mean()
        last = s[0] < m
        reg = 0
        for i in range(1, len(s)):
            if (s[i] < m) != last:
                reg += 1
                last = s[i] < m
        score = reg

        if score >= 35:
            print([sorted(choice), (score - 30) / 20])
            found += 1
        data.append(score)

    return data

findDist(5)
findDist(6)
findDist(7)
findDist(8)
findDist(9)
findDist(10)
