"""Verify the key claims of the dumsor QAOA project.
Run:  python ghana-quantum/dumsor/tests/verify_results.py
Independently re-scores all 4,096 schedules with the penalty rules from qaoa_dumsor.py
(synthetic data) and checks results.json and aer_results.json."""
import itertools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
res = json.load(open(os.path.join(ROOT, "results.json"))); aer = json.load(open(os.path.join(ROOT, "aer_results.json")))
fails = 0
def check(name, ok, detail):
    global fails
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"); fails += (not ok)

groups = "ABCDEF"
d = np.array([60, 45, 50, 35, 40, 55.0]); crit = np.array([3, 0, 1, 0, 2, 0.0]); S = np.array([100, 90.0])
def penalty(x):
    x = np.asarray(x).reshape(2, 6)
    return sum((d @ x[t] - S[t]) ** 2 + 150 * (crit @ x[t]) for t in range(2)) + 2000 * np.sum(x[0] * x[1])
allx = np.array(list(itertools.product([0, 1], repeat=12)))
costs = np.array([penalty(x) for x in allx])
best = costs.min(); nopt = int(np.isclose(costs, best).sum())
check("search space", len(allx) == 4096 and res["n_qubits"] == 12, "12 switches -> 2^12 = 4,096 schedules, 12 qubits")
check("brute-force optimum = 175", best == 175 and res["optimum_cost"] == 175, f"recomputed {best:.0f}, file {res['optimum_cost']:.0f}")
check("two schedules tie for best", nopt == 2 == res["n_optimal"], f"recomputed {nopt}, file {res['n_optimal']}")
def sched_vec(s):
    x = np.zeros((2, 6))
    for t, blk in enumerate(s):
        for g in blk["shed"]: x[t, groups.index(g)] = 1
    return x.flatten()
check("stored optimum schedule really scores 175", penalty(sched_vec(res["optimum_schedule"])) == 175, " / ".join("+".join(b["shed"]) for b in res["optimum_schedule"]))
g = penalty(sched_vec(res["greedy_schedule"]))
check("rule of thumb (greedy) = 1675", g == 1675 == res["greedy_cost"], f"recomputed {g:.0f} ({g/best:.1f} x the optimum)")
check("mean random schedule = 15450", abs(costs.mean() - res["mean_random_cost"]) < 1e-6, f"mean over all 4,096 = {costs.mean():,.1f}")
check("ordering optimum < greedy < random", best < g < costs.mean(), f"{best:.0f} < {g:.0f} < {costs.mean():,.0f}")

print("\nQAOA, exact statevector simulation (results.json):")
uni = nopt / 4096
for q in res["qaoa"]:
    check(f"p={q['p']}: chance of a best schedule above blind guessing", q["p_opt"] > uni and abs(q["uniform_p_opt"] - 1/2048) < 1e-12,
          f"{q['p_opt']:.5f} vs {uni:.6f} (= 2/4096) -> {q['p_opt']/uni:.1f}x; optimum seen in 1024 shots: {q['found_opt_in_1024']}")
ps = [q["p_opt"] for q in res["qaoa"]]
check("more rounds -> better odds", all(a < b for a, b in zip(ps, ps[1:])), " -> ".join(f"{p:.4f}" for p in ps))
check("raw single-state chance also above 1/4096", min(ps) / 2 > 1/4096, f"per optimal state >= {min(ps)/2:.5f} vs {1/4096:.6f}")

print("\nQiskit Aer, 20,000 sampled shots, ideal vs noisy (aer_results.json) compared with exact statevector:")
exact = {q["p"]: q for q in res["qaoa"]}
for a in aer:
    e = exact[a["p"]]; tag = a["sim"]
    if a["sim"] == "ideal":
        se = np.sqrt(e["p_opt"] * (1 - e["p_opt"]) / a["shots"])
        check(f"p={a['p']} ideal Aer sample agrees with exact", abs(a["p_opt"] - e["p_opt"]) < 3 * se, f"sampled {a['p_opt']:.4f} vs exact {e['p_opt']:.4f} (3-sigma band {3*se:.4f})")
    check(f"p={a['p']} {tag}: best schedule (175) still found", a["best_cost"] == 175, f"best sampled {a['best_cost']:.0f}, {a['twoq_gates']} two-qubit gates")
for p in sorted({a["p"] for a in aer}):
    i = [a for a in aer if a["p"] == p and a["sim"] == "ideal"][0]; nz = [a for a in aer if a["p"] == p and a["sim"] != "ideal"][0]
    check(f"p={p}: noise lowers the odds (claim)", nz["p_opt"] < i["p_opt"], f"{i['p_opt']*100:.2f}% -> {nz['p_opt']*100:.2f}%; avg penalty {i['exp_cost']:,.0f} -> {nz['exp_cost']:,.0f}")
print(f"\n{'ALL CHECKS PASSED' if fails == 0 else str(fails) + ' CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
