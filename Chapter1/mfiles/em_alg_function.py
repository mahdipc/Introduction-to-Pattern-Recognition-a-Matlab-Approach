
import numpy as np


def em_alg_function(x, m, s, Pa, e_min):
    x = np.array(x)
    m = np.array(m).T
    p, n = x.shape
    J, n = m.shape
    e = e_min+1
    Q_tot = []
    e_tot = []
    iter = 0

    while (e > e_min):
        iter = iter+1
        P_old = Pa
        m_old = m
        s_old = s
        P = np.zeros([J, p])
        for k in range(p):
            temp = gauss(x[k, :], m, s)
            P_tot = temp.dot(Pa.T)
            for j in range(J):
                P[j, k] = temp[j]*Pa[j]/P_tot
        Q = 0
        for k in range(p):
            for j in range(J):
                Q = Q+P[j, k]*(-(n/2)*np.log(2*np.pi*s[j]) -
                               np.sum((x[k, :]-m[j, :])**2)/(2*s[j])+np.log(Pa[j]))
        Q_tot.append(Q)

        for j in range(J):
            a = np.zeros([1, n])
            for k in range(p):
                a = a+P[j, k]*x[k, :]
            m[j, :] = a/np.sum(P[j, :])

        for j in range(J):
            b = 0
            for k in range(p):
                b = b+P[j, k]*((x[k, :]-m[j, :]).dot((x[k, :]-m[j, :]).T))

            s[j] = b/(n*np.sum(P[j, :]))
            if s[j] < 10**(-10):
                s[j] = 0.001

        # Determine the a priori probabilities
        for j in range(J):
            a = 0
            for k in range(p):
                a = a+P[j, k]
            Pa[j] = a/p

        e = np.sum(np.abs(Pa-P_old)) + \
            np.sum(np.sum(np.abs(m-m_old)))+np.sum(np.abs(s-s_old))
        e_tot.append(e)

    return [m, s, Pa, iter, Q_tot, e_tot]
