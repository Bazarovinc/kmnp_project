import numpy as np

m = 0.25 * 9.1e-31
Ef = 0.1
a = 1e-10
hbar = 1.06e-34
q = 1.6e-19
t0 = (hbar ** 2) / (2 * m * (a ** 2) * q)
energy = np.arange(0, 1.001, 0.001)
IE = (q ** 2) / (2 * np.pi * hbar)
kT = .025
zplus = complex(0, 1e-12)
dE = energy[1] - energy[0]
