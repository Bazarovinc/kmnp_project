from typing import List

import numpy as np
import numpy.linalg as ln

from constants import *


def create_d(w: int, b1: int, b2: int) -> List[float]:
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
    voltage = 0
    sig1 = np.zeros((Np, Np), dtype=complex)
    sig2 = np.zeros((Np, Np), dtype=complex)
    eye = np.eye(Np)
    U = np.array(voltage * np.concatenate((
        0.5 * np.ones(NS, dtype=float),
        np.linspace(0.5, -0.5, NC),
        -0.5 * np.ones(ND, dtype=float)
    ), axis=0), ndmin=2).T
    ck_1 = 1 - ((energy + zplus - U[0] - UB[0]) / (2 * t0))
    ka_1 = np.arccos(ck_1)
    ck_2 = 1 - ((energy + zplus - U[Np - 1] - UB[Np - 1]) / (2 * t0))
    ka_2 = np.arccos(ck_2)
    diag = np.diag(U, 0)
    d = []
    for i, e in enumerate(energy):
        sig1[0][0] = -t0 * np.exp(1j * ka_1[i])
        gam1 = 1j * (sig1 - sig1.conj().T)
        sig2[Np - 1][Np - 1] = -t0 * np.exp(1j * ka_2[i])
        gam2 = 1j * (sig2 - sig2.conj().T)
        G = ln.inv(((e + zplus) * eye) - T - diag - sig1 - sig2)
        d.append(np.real(np.trace(np.dot(np.dot(np.dot(gam1, G), gam2), G.conj().T))))
    return d
