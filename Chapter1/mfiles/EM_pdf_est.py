import numpy as np


def EM_pdf_est(X, y, m_ini, s_ini, P_ini):
    l, N = X.shape
    e_min = 10**(-5)
    cl = int(np.max(y))
    acc_tot = []
    Xs = []
    for j in range(cl):
        temp = []
        t = 0
        for i in range(N):
            if(y[i] == j+1):
                temp.append(X[:, i])
                t = t+1
        acc_tot.append(t)
        Xs.append(temp)
    P_cl = np.array(acc_tot)/N
    m = []
    s = []
    P = []
    for j in range(cl):
        [mj, sj, Pj, iter1, Q_tot, e_tot] = em_alg_function(
            Xs[j], m_ini[j], s_ini[j], P_ini[j], e_min)
        m.append(mj)
        s.append(sj)
        P.append(Pj)

    return [m, s, P, P_cl]
