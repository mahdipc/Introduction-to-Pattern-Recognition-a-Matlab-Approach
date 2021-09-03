import numpy as np


def gauss(x, m, s):
    J, l = m.shape
    p = x.shape
    z = []
    for j in range(J):
        t = (x-m[j, :]).dot(x-m[j, :]).T
        c = 1/(2*np.pi*s[j])**(1/2)
        z.append(c*np.exp(-t/2*s[j]))
    return np.array(z)
