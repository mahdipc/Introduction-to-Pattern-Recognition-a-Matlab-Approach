import numpy as np


def mixture_Bayes(m, S, P, P_cl, X):
    cl = len(m)
    l, N = X.shape
    y = []
    for i in range(N):
        temp = []
        for j in range(cl):
            t = mixt_value(m[j], S[j], P[j], X[:, i])
            temp.append(t)
        temp = P_cl*temp
        q1 = np.max(temp)
        q2 = np.argmax(temp)
        y.append(q2+1)
    return y
