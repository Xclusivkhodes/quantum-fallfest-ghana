"""Noisy simulation of folded ISA circuits with Aer density-matrix + fake-backend noise.

Pipeline per (instance, steps):
  1. transpile the logical Trotter circuit once (opt level 1, fixed seed) -> ISA unitary U
  2. global unitary folding U (U^dag U)^k, k = (lam-1)/2, re-basis with opt level 0
     (no gate cancellation possible), ALAP-schedule so idle qubits get T1/T2 noise
  3. density-matrix simulation -> exact pre-readout probabilities on the n logical qubits
  4. readout error applied analytically from the backend's per-qubit assignment matrices
Finite-shot statistics are then drawn by multinomial sampling with explicit seeds.
"""
import copy
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


def scaled_backend(backend, g):
    """Copy of a fake backend whose gate errors are multiplied by g and T1/T2 divided by g
    (readout untouched). Used to emulate calibration drift between executions."""
    if g == 1.0:
        return backend
    b2 = copy.deepcopy(backend); T = b2.target
    for op in T.operation_names:
        if op == "measure":
            continue
        for qa, pr in T[op].items():
            if pr is not None and pr.error is not None:
                pr.error = min(pr.error * g, 0.75)
    for qp in T.qubit_properties:
        if qp is not None:
            if qp.t1: qp.t1 = qp.t1 / g
            if qp.t2: qp.t2 = qp.t2 / g
    return b2


def best_chain(backend, n):
    """Deterministic DFS for the n-qubit line minimising readout + 2q-gate error."""
    T = backend.target
    tq = [op for op in T.operation_names if op in ("ecr", "cz", "cx")][0]
    e2 = {}
    for (a, b), pr in T[tq].items():
        e2[(a, b)] = e2[(b, a)] = min(e2.get((a, b), 1), pr.error)
    ro = {q[0]: T["measure"][q].error for q in T["measure"]}
    nbr = {}
    for a, b in e2:
        nbr.setdefault(a, set()).add(b)
    best = (1e9, None)
    def dfs(path, cost):
        nonlocal best
        if cost >= best[0]:
            return
        if len(path) == n:
            best = (cost, list(path)); return
        for nb in sorted(nbr.get(path[-1], ())):
            if nb not in path:
                path.append(nb); dfs(path, cost + ro[nb] + e2[(path[-2], nb)]); path.pop()
    for q in sorted(nbr):
        dfs([q], ro[q])
    return best[1]


class NoisyRunner:
    def __init__(self, backend, seed_transpiler=7, schedule=True, layout=None):
        self.backend = backend
        self.layout = layout
        self.noise_model = NoiseModel.from_backend(backend)
        self.sim = AerSimulator.from_backend(backend, method="density_matrix")
        self.seed = seed_transpiler
        self.schedule = schedule

    def isa(self, logical):
        qc = logical.copy()
        qc.measure_all()
        t = transpile(qc, self.backend, optimization_level=1, seed_transpiler=self.seed,
                      initial_layout=self.layout)
        phys = t.layout.final_index_layout()  # physical qubit of each logical qubit
        u = t.remove_final_measurements(inplace=False)
        return u, phys

    def folded(self, u, phys, lam):
        assert lam % 2 == 1
        k = (lam - 1) // 2
        qc = u.copy()
        inv = u.inverse()
        for _ in range(k):
            qc.barrier()
            qc.compose(inv, inplace=True)
            qc.barrier()
            qc.compose(u, inplace=True)
        kw = dict(optimization_level=0, layout_method="trivial", routing_method="none")
        if self.schedule:
            kw["scheduling_method"] = "alap"
        out = transpile(qc, self.backend, **kw)
        out.save_density_matrix(qubits=list(phys), label="rho")
        return out

    def local_folded(self, u, phys, lam, seed=0):
        """Local folding of 2q gates only: G -> G G^dag G for a subset of 2q gates so that
        the 2q-gate count is scaled by lam (lam in [1,3]). 1q gates are NOT folded, so the
        nominal lam mis-states the true amplification of 1q/idle noise (realistic case)."""
        twoq = [i for i, ins in enumerate(u.data) if ins.operation.num_qubits == 2]
        nfold = int(round((lam - 1) / 2 * len(twoq)))
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(twoq, size=nfold, replace=False).tolist()) if nfold else set()
        qc = QuantumCircuit(*u.qregs, *u.cregs)
        for i, ins in enumerate(u.data):
            qc.append(ins)
            if i in chosen:
                inv = ins.operation.inverse()
                qc.barrier(ins.qubits)
                qc.append(inv, ins.qubits)
                qc.barrier(ins.qubits)
                qc.append(ins.operation, ins.qubits)
        kw = dict(optimization_level=0, layout_method="trivial", routing_method="none")
        if self.schedule:
            kw["scheduling_method"] = "alap"
        out = transpile(qc, self.backend, **kw)
        out.save_density_matrix(qubits=list(phys), label="rho")
        return out

    def readout_matrices(self, phys):
        mats = []
        for q in phys:
            err = self.noise_model._local_readout_errors.get((q,))
            mats.append(np.eye(2) if err is None else np.array(err.probabilities))
        return mats  # mats[i][true, meas]

    def probs(self, circ):
        res = self.sim.run(circ, shots=1).result()
        rho = np.asarray(res.data(0)["rho"])
        return np.clip(np.real(np.diag(rho)), 0, None)


def apply_readout(p, mats):
    """p over n qubits (little-endian); mats[i] is 2x2 P(meas|true) for logical qubit i."""
    n = len(mats)
    t = p.reshape([2] * n)  # axes ordered q_{n-1} ... q_0
    for i, m in enumerate(mats):
        ax = n - 1 - i
        t = np.moveaxis(np.tensordot(t, m, axes=([ax], [0])), -1, ax)
    out = t.reshape(-1)
    return out / out.sum()


def ideal_probs(logical):
    return np.abs(Statevector(logical).data) ** 2
