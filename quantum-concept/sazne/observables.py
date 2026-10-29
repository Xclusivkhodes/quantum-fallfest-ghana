"""Z-basis observables computed from a probability vector over bitstrings.
Index convention: bit i of the integer index = qubit i (Qiskit little-endian)."""
import numpy as np


def zvals(n):
    idx = np.arange(2 ** n)
    return np.array([1 - 2 * ((idx >> i) & 1) for i in range(n)])  # shape (n, 2^n)


def make_observables(n):
    Z = zvals(n)
    stag = sum(((-1) ** i) * Z[i] for i in range(n)) / n
    nn = sum(Z[i] * Z[i + 1] for i in range(n - 1)) / (n - 1)
    weight = ((1 - Z) // 2).sum(axis=0)
    sector = (weight == n // 2)
    return {"stag_mag": stag.astype(float), "nn_zz": nn.astype(float)}, sector


def expval(p, o):
    return float(np.dot(p, o) / p.sum())
