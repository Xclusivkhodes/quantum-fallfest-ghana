"""Trotterized XXZ chains with random couplings/fields (conserve total Sz)."""
import numpy as np
from qiskit import QuantumCircuit


def random_xxz_instance(n, rng):
    """Random open-chain XXZ Hamiltonian: H = sum_i J_i (XX + YY + D_i ZZ) + sum_i h_i Z_i."""
    return {
        "n": n,
        "J": rng.uniform(0.5, 1.5, n - 1).tolist(),
        "Delta": rng.uniform(0.3, 1.5, n - 1).tolist(),
        "h": rng.uniform(-1.0, 1.0, n).tolist(),
    }


def trotter_circuit(inst, steps, dt=0.25):
    """First-order Trotter circuit from the Neel state |0101..>. Every gate conserves
    Hamming weight, so the ideal output lives in the weight-n/2 sector."""
    n = inst["n"]
    qc = QuantumCircuit(n)
    for q in range(1, n, 2):
        qc.x(q)
    for _ in range(steps):
        for parity in (0, 1):
            for i in range(parity, n - 1, 2):
                J, D = inst["J"][i], inst["Delta"][i]
                qc.rxx(2 * J * dt, i, i + 1)
                qc.ryy(2 * J * dt, i, i + 1)
                qc.rzz(2 * J * D * dt, i, i + 1)
        for i in range(n):
            qc.rz(2 * inst["h"][i] * dt, i)
    return qc
