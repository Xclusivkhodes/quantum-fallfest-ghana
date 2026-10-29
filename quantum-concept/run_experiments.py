"""Stage 1: exact noisy output distributions for every (backend, instance, depth, folding, lam).

Usage:  python run_experiments.py --instances 20 --out results/probs.npz
Everything is seeded (instances: --seed, transpiler: 7, local-fold gate choice: fixed).
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sazne.circuits import random_xxz_instance, trotter_circuit
from sazne.sim import NoisyRunner, ideal_probs, scaled_backend, best_chain

FOLDS = {"global": [1, 3, 5], "local": [1, 2, 3]}
_runners = {}


def _runner(name, g, n):
    key = (name, g)
    if key not in _runners:
        _runners.clear()  # one runner per worker at a time (memory)
        from qiskit_ibm_runtime import fake_provider
        b = getattr(fake_provider, name)()
        _runners[key] = NoisyRunner(scaled_backend(b, g), layout=best_chain(b, n))
    return _runners[key]


def job(args):
    bname, g, idx, inst, steps, dt = args
    r = _runner(bname, g, inst["n"])
    lc = trotter_circuit(inst, steps, dt)
    u, phys = r.isa(lc)
    out = {"backend": bname, "g": g, "idx": idx, "steps": steps, "phys": list(map(int, phys)),
           "n2q": int(sum(1 for i in u.data if i.operation.num_qubits == 2)),
           "ideal": ideal_probs(lc), "readout": np.array(r.readout_matrices(phys))}
    for scheme, lams in FOLDS.items():
        ps = []
        for lam in lams:
            c = r.folded(u, phys, lam) if scheme == "global" else r.local_folded(u, phys, lam, seed=1000 + idx)
            ps.append(r.probs(c))
        out[scheme] = np.array(ps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="FakeSherbrooke,FakeTorino")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--steps", default="1,2,3")
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--drift", default="0.7,0.85,1.0,1.15,1.3", help="noise gains g")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/probs.npz")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    insts = [random_xxz_instance(a.n, rng) for _ in range(a.instances)]
    steps = [int(s) for s in a.steps.split(",")]
    t0 = time.time(); res = []
    for b in a.backends.split(","):
        for g in [float(x) for x in a.drift.split(",")]:
            jobs = [(b, g, i, insts[i], s, a.dt) for i in range(a.instances) for s in steps]
            with ProcessPoolExecutor(a.workers) as ex:
                res.extend(ex.map(job, jobs))
            print(f"{b} g={g}: {len(res)} records, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, records=np.array(res, dtype=object), allow_pickle=True)
    json.dump({"args": vars(a), "instances": insts, "folds": FOLDS}, open(a.out.replace(".npz", "_meta.json"), "w"), indent=1)
    print("saved", a.out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
