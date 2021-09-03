import numpy as np


def mixt_value(m, S, P, X):
    l = X.shape
    N = 1
    m = m.T
    l, c = m.shape
    y = []
    for i in range(N):
        temp = []
        for j in range(c):
            t = multivariate_normal.pdf(X[:], mean=m[:, j], cov=S[:, :, j])
            temp.append(t)
        y_temp = np.sum(P*np.array(temp))
        y.append(y_temp+1)
    return y[0]
