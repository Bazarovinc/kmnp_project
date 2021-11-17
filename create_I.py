# импортирование библиотеки для работы с матрицами, векторами
from typing import List

import numpy as np
import numpy.linalg as ln

from constants import *


def get_voltage(w: int, b1: int, b2: int) -> np.ndarray:
    dif = 0.15 * (w + b1 + b2 - 30)
    if 2 - dif >= 0.75:
        return np.arange(0, 2 - dif + 0.001, 0.001)
    else:
        return np.arange(0, 0.751, 0.001)


def create_I(w: int, b1: int, b2: int) -> List[float]:
    NS = 1
    NC = (b1 + b2) + w
    ND = 1
    Np = NS + NC + ND
    UB = np.concatenate((
        np.zeros((NS, 1)),
        0.4 * np.ones((b1, 1)),
        np.zeros((w, 1)),
        0.4 * np.ones((b2, 1)),
        np.zeros((ND, 1))))
    T = 2 * t0 * np.diag(np.ones(Np), 0) + (-t0 * np.diag(np.ones(Np - 1), 1)) + \
        (-t0 * np.diag(np.ones(Np - 1), -1))
    T += np.diag(UB.T[0], 0)
    VV = get_voltage(w, b1, b2)
    II = []
    sig1 = np.zeros((Np, Np), dtype=complex)
    sig2 = np.zeros((Np, Np), dtype=complex)
    eye = np.eye(Np)
    Uf = VV * np.array(np.concatenate((
        0.5 * np.ones(NS, dtype=float),
        np.linspace(0.5, -0.5, NC),
        -0.5 * np.ones(ND, dtype=float)
    ), axis=0), ndmin=2).T
    Uf = Uf.T
    mu1f = Ef + (VV / 2)
    mu2f = Ef - (VV / 2)
    for i, iV in enumerate(VV):
        mu1 = mu1f[i]
        mu2 = mu2f[i]
        U1 = Uf[:][i]
        f1 = 1.0 / (1 + np.exp((energy - mu1) / kT))
        f2 = 1.0 / (1 + np.exp((energy - mu2) / kT))
        ck_1 = 1 - ((energy + zplus - U1[0] - UB[0]) / (2 * t0))
        ka_1 = np.arccos(ck_1)
        ck_2 = 1 - ((energy + zplus - U1[Np - 1] - UB[Np - 1]) / (2 * t0))
        ka_2 = np.arccos(ck_2)
        TM_l = []
        diag = np.diag(U1.T, 0)
        for i, e in enumerate(energy):
            sig1[0][0] = -t0 * np.exp(1j * ka_1[i])
            gam1 = 1j * (sig1 - sig1.conj().T)
            sig2[Np - 1][Np - 1] = -t0 * np.exp(1j * ka_2[i])
            gam2 = 1j * (sig2 - sig2.conj().T)
            G = ln.inv(((e + zplus) * eye) - T - diag - sig1 - sig2)
            TM_l.append(np.real(np.trace(np.dot(np.dot(np.dot(gam1, G), gam2), G.conj().T))))
        diff_f = f1 - f2
        TM_l = np.array(TM_l)
        II.append(sum(dE * IE * TM_l * diff_f))
    return II
