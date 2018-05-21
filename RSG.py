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

d = (df.as_matrix().T * p).T.astype(np.float32)

import random


def findDist(n):
    data = []
    found = 0
    while found < 30:
        choice = random.sample(range(d.shape[0]), n)
        if 372 not in choice:
            choice = choice + [372]
        bad = True
        while bad:
            bad = False
            for k in range(len(choice)):
                for l in range(k):
                    if np.abs(cor[choice[k]][choice[l]]) > 0.75:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                choice = random.sample(range(d.shape[0]), n)
                if 372 not in choice:
                    choice = choice + [372]

        s = d[choice].sum(axis=0)
        s = s / n
        m = s.mean()
        s = s / m
        x = np.arange(len(s)) + 1
        a, b = np.polyfit(x, np.log(s), 1)
        y = np.exp(a * x + b)
        last = s[0] < y[0]
        reg = 0
        for i in range(1, len(s)):
            if (s[i] < y[i]) != last:
                reg += 1
                last = s[i] < y[i]

        if reg >= 35:
            with open('samples.txt', 'a') as log:
                weight = np.ones(len(s))
                weight[-30:] = 4
                tolog = str([sorted(choice), (reg - 30) / 20, (((y - s) * weight) ** 2).sum()])
                print(tolog)
                log.write(tolog + '\n')
            found += 1
        data.append(reg)

    return data


while True:
    findDist(3)
    findDist(4)
    #findDist(5)
    #findDist(6)
    #findDist(7)
    #findDist(8)
    #findDist(9)
    #findDist(10)
