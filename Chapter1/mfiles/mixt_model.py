import numpy as np


def mixt_model(m, S, P, N):
    l, c = m.shape
    P_acc = [P[0]]
    for i in range(1, c):
        t = P_acc[i-1]+P[i]
        P_acc.append(t)

    X = []
    y = []
    P_acc = np.array(P_acc)

    for i in range(N):
        t = np.random.rand()
        ind = np.sum(t > P_acc)
        X.append(np.random.multivariate_normal(m[:, ind], S[:, :, ind]))
        y.append(ind+1)
    return np.array(X).T, np.array(y)
