import nbformat as nbf
nb = nbf.v4.new_notebook(); C=[]
md=lambda s: C.append(nbf.v4.new_markdown_cell(s)); code=lambda s: C.append(nbf.v4.new_code_cell(s))
md("""# Fairer "dumsor" schedules with a quantum computer: a small demo

**The problem.** When Ghana doesn't have enough electricity, the Electricity Company of Ghana (ECG) has to switch some areas off. This is *dumsor*. People complain that it is unannounced and unfair: some areas lose power again and again, and sometimes hospitals or water pumps go dark.

**What this notebook does.** It turns "who should be switched off tonight?" into a puzzle a quantum computer can work on. It then compares the quantum result with (1) checking every possible schedule and (2) a simple rule of thumb an operator might use.

**Honest note.** The numbers are made up but realistic; they are not real ECG data. A normal laptop solves a puzzle this small instantly. The point is to show *how* a fair schedule can be written down clearly, and how quantum computers would approach it. It runs in about a minute.

---
**Step 1: Install the tools.** This installs Qiskit (IBM's free quantum-computing toolkit) and its simulator, which imitates a quantum computer on an ordinary computer.""")
code("%pip install -q qiskit qiskit-aer scipy numpy")
md("""**Step 2: Describe the town.** Imagine six groups of neighbourhoods (A to F), each using some electricity (in megawatts, MW). Some contain things that must stay on, like hospitals or water pumps; the "critical" score says how important they are (0 = ordinary homes and shops).

There are two evening time slots: 6–9 pm and 9 pm–midnight. In each slot we are short of power by a set amount, so we must switch off roughly that much.""")
code("""import itertools, time
import numpy as np

groups   = ["A", "B", "C", "D", "E", "F"]
demand   = np.array([60, 45, 50, 35, 40, 55.0])   # MW each group uses (made-up)
critical = np.array([3, 0, 1, 0, 2, 0.0])         # importance: hospitals, water pumps (made-up)
slots    = ["6-9 pm", "9 pm-midnight"]
shortfall = np.array([100, 90.0])                 # MW we must cut in each slot (made-up)

N, T = len(groups), len(slots)
n = N * T                                         # one on/off switch per group per slot = 12 switches
print(f"{N} groups x {T} time slots = {n} yes/no decisions -> {2**n:,} possible schedules")""")
md("""**Step 3: Write down what "a good schedule" means.** Every schedule gets a penalty score, and lower is better. Points are added for:
1. **Cutting the wrong amount**: too much (people lose power for no reason) or too little (the grid could collapse).
2. **Switching off important places** like hospitals.
3. **Unfairness**: switching the same group off in *both* slots. This gets a very big penalty.""")
code("""W_CRITICAL, W_FAIR = 150.0, 2000.0

def penalty(x):
    x = np.asarray(x).reshape(T, N)          # row = time slot, column = group; 1 means "switched off"
    score = 0.0
    for t in range(T):
        score += (demand @ x[t] - shortfall[t]) ** 2   # wrong amount cut
        score += W_CRITICAL * (critical @ x[t])        # hospitals / pumps hit
    score += W_FAIR * np.sum(x[0] * x[1])              # same group off twice
    return score

def describe(x):
    x = np.asarray(x).reshape(T, N)
    for t in range(T):
        off = [groups[i] for i in range(N) if x[t, i]]
        print(f"  {slots[t]:>14}: switch off {'+'.join(off) or 'nobody':<8} "
              f"cuts {demand @ x[t]:.0f} of {shortfall[t]:.0f} MW needed, critical score {critical @ x[t]:.0f}")""")
md("""**Step 4: The guaranteed answer, by checking every schedule.** With only 4,096 possibilities, an ordinary computer can try them all in a blink. This gives the true best schedule, which we use to grade the other methods.""")
code("""t0 = time.time()
all_x = np.array(list(itertools.product([0, 1], repeat=n)))
scores = np.array([penalty(x) for x in all_x])
best_score = scores.min()
best_set = np.where(np.isclose(scores, best_score))[0]
print(f"Checked all {len(all_x):,} schedules in {time.time()-t0:.2f} s")
print(f"Best possible penalty: {best_score:.0f}  ({len(best_set)} schedules tie for best)")
describe(all_x[best_set[0]])
print(f"Average penalty of a random schedule: {scores.mean():,.0f}")""")
md("""**Step 5: A simple rule of thumb.** An operator might go slot by slot: switch off the least-important, biggest groups first until enough is cut, and never hit the same group twice. Let's see how it does.""")
code("""def rule_of_thumb():
    x = np.zeros((T, N)); used = set()
    for t in range(T):
        cut = 0
        for i in sorted(range(N), key=lambda i: (critical[i], -demand[i])):
            if i in used: continue
            if cut >= shortfall[t]: break
            x[t, i] = 1; cut += demand[i]; used.add(i)
    return x.flatten()

x_rule = rule_of_thumb()
print(f"Rule-of-thumb penalty: {penalty(x_rule):.0f}   (best possible is {best_score:.0f})")
describe(x_rule)""")
md("""**Step 6: Turn the puzzle into a quantum circuit.** Each of the 12 decisions becomes one *qubit* (a quantum bit). The method is called **QAOA**: the circuit adds the penalty rules into the qubits, then "mixes" them, and repeats this a few rounds (`rounds` below). Each round has two dials. When the dials are set well, schedules with low penalties come out more often when we measure.

The code below converts our penalty rules into the form the circuit needs, and builds the circuit.""")
code("""from qiskit import QuantumCircuit

# Rewrite penalty(x) as: constant + sum(lin_k x_k) + sum(Q_ab x_a x_b), then in the +/-1 "spin" form a circuit uses.
idx = lambda i, t: t * N + i
lin = np.zeros(n); Q = {}; const = 0.0
for t in range(T):
    const += shortfall[t] ** 2
    for i in range(N):
        lin[idx(i, t)] += demand[i] ** 2 - 2 * shortfall[t] * demand[i] + W_CRITICAL * critical[i]
        for j in range(i + 1, N):
            Q[(idx(i, t), idx(j, t))] = 2 * demand[i] * demand[j]
for i in range(N):
    Q[(idx(i, 0), idx(i, 1))] = W_FAIR
h = -lin / 2; J = {}
for (a, b), v in Q.items():
    h[a] -= v / 4; h[b] -= v / 4; J[(a, b)] = v / 4
scale = max(np.abs(h).max(), max(abs(v) for v in J.values()))   # shrink numbers to a gentle range

def qaoa_circuit(dials, rounds):
    g, b = dials[:rounds], dials[rounds:]
    qc = QuantumCircuit(n)
    qc.h(range(n))                                   # start: every schedule equally likely
    for r in range(rounds):
        for q in range(n): qc.rz(2 * g[r] * h[q] / scale, q)            # add penalty rules
        for (a, c), v in J.items(): qc.rzz(2 * g[r] * v / scale, a, c)  # add pair-wise rules
        for q in range(n): qc.rx(2 * b[r], q)                            # mix
    qc.measure_all()
    return qc

print(qaoa_circuit([0.1, 0.1], 1).count_ops())""")
md("""**Step 7: Tune the dials.** We try many dial settings and keep the ones that give the lowest average penalty. To keep this quick, a fast exact shortcut calculates what the circuit would produce; Step 8 then runs the real circuit. We tune for 1 round and for 2 rounds.""")
code("""from scipy.optimize import minimize

# Exact shortcut: the penalty step only changes phases; the mix step acts on each qubit separately.
spin_energy = np.array([h @ (1 - 2 * x) + sum(v * (1 - 2 * x[a]) * (1 - 2 * x[b]) for (a, b), v in J.items())
                        for x in all_x]) / scale
def state(dials, rounds):
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for r in range(rounds):
        psi = psi * np.exp(-1j * dials[r] * spin_energy)
        c, s = np.cos(dials[rounds + r]), -1j * np.sin(dials[rounds + r])
        psi = psi.reshape([2] * n)
        for q in range(n):
            a0, a1 = np.take(psi, 0, axis=q), np.take(psi, 1, axis=q)
            psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=q)
        psi = psi.reshape(-1)
    return np.abs(psi) ** 2

tuned = {}
rng = np.random.default_rng(1)
for rounds, tries in [(1, 10), (2, 6)]:
    t0 = time.time(); best = None
    for _ in range(tries):
        start = np.concatenate([rng.uniform(0, 0.5, rounds), rng.uniform(0, np.pi / 2, rounds)])
        res = minimize(lambda d: state(d, rounds) @ spin_energy, start, method="COBYLA", options={"maxiter": 400})
        if best is None or res.fun < best.fun: best = res
    tuned[rounds] = best.x
    prob = state(best.x, rounds)
    print(f"{rounds} round(s): chance of best schedule {prob[best_set].sum():.4f} "
          f"(blind guessing: {len(best_set)/2**n:.4f}), average penalty {prob @ scores:,.0f}  [{time.time()-t0:.0f} s]")""")
md("""**Step 8: Run the real quantum circuit on the simulator.** We "measure" the circuit 20,000 times, so it suggests 20,000 schedules, and we grade them. We run it twice: on a perfect simulated quantum computer, and on a *noisy* one that makes small random errors like today's real machines (0.1% error per single-qubit step, 1% per two-qubit step).""")
code("""from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

noise = NoiseModel()
noise.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["rz", "rx", "h"])
noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["rzz"])

rows = []
for rounds in [1, 2]:
    qc = qaoa_circuit(tuned[rounds], rounds)
    for label, sim in [("perfect", AerSimulator()), ("noisy", AerSimulator(noise_model=noise))]:
        counts = sim.run(transpile(qc, sim, seed_transpiler=1), shots=20000, seed_simulator=7).result().get_counts()
        total = sum(counts.values()); hit = 0; avg = 0; lowest = 1e18; best_bits = None
        for bits, c in counts.items():
            x = [int(ch) for ch in bits[::-1]]          # Qiskit lists qubit 0 last
            p = penalty(x); avg += c * p / total
            if np.isclose(p, best_score): hit += c
            if p < lowest: lowest, best_bits = p, x
        rows.append((rounds, label, hit / total, avg, lowest))
        print(f"{rounds} round(s), {label:>7}: best schedule appeared {hit} times out of {total} "
              f"({hit/total:.2%}); average penalty {avg:,.0f}; lowest penalty seen {lowest:.0f}")
print("\\nBest schedule the quantum circuit found:")
describe(best_bits)""")
md("""**Step 9: Scoreboard and what it means.**""")
code("""print(f"{'Method':<40}{'Penalty (lower = better)':>26}")
print(f"{'Check every schedule (guaranteed best)':<40}{best_score:>26.0f}")
print(f"{'Rule of thumb':<40}{penalty(x_rule):>26.0f}")
print(f"{'Random schedule (average)':<40}{scores.mean():>26,.0f}")
for rounds, label, hit, avg, lowest in rows:
    print(f"{'Quantum, '+str(rounds)+' round(s), '+label+' (best seen)':<40}{lowest:>26.0f}")""")
md("""**Takeaways (in plain words)**
- Writing the rules down openly (*cut the right amount, protect hospitals, never hit the same area twice*) already produces a schedule that is **fairer and more accurate than the rule of thumb**. This is useful today, with or without a quantum computer.
- The quantum circuit **does find the best schedule** and suggests it several times more often than blind guessing, but still only a small share of the time. Noise in today's machines makes it worse.
- An ordinary computer solves this small example instantly. Quantum computers would only matter for much bigger versions, and even then it is unproven. We present this as a learning demo and a transparent scheduling model, **not** as a quantum speed-up.""")
nb["cells"]=C
nb["metadata"]["kernelspec"]={"name":"python3","display_name":"Python 3","language":"python"}
nbf.write(nb, "/workspace/notebooks/dumsor_qaoa_demo.ipynb")
