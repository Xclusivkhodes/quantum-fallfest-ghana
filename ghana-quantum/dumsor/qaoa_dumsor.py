"""Toy fair load-shedding (dumsor) schedule as a QUBO, solved with QAOA (Qiskit statevector) vs classical baselines.
Data is SYNTHETIC but shaped like an ECG regional rotation: 6 feeder groups, 2 evening blocks (18-21h, 21-24h).
Demands (MW) and critical-load weights are illustrative assumptions, not ECG data.
"""
import itertools, json, time
import numpy as np
from scipy.optimize import minimize
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector

groups = ["A","B","C","D","E","F"]
d = np.array([60, 45, 50, 35, 40, 55.0])      # MW per feeder group (assumed)
crit = np.array([3, 0, 1, 0, 2, 0.0])          # critical-load weight: hospitals, water pumps (assumed)
S = np.array([100, 90.0])                      # MW shortfall to shed in each block (assumed)
T = len(S); N = len(d); n = N*T
A, B, C = 1.0, 2000.0, 150.0                    # penalty weights: shortfall match, fairness, critical load
scale = 100.0                                   # MW scaling for numerics
idx = lambda i,t: t*N+i

def cost(x):
    x = np.asarray(x).reshape(T,N)
    c = 0.0
    for t in range(T):
        c += A*((d@x[t]-S[t])/scale*10)**2*100     # (MW mismatch)^2, in MW^2
        c += C*(crit@x[t])
    c += B*np.sum(x[0]*x[1])                       # same group shed twice -> unfair
    return c

# QUBO -> Ising via exact expansion: build Q matrix
Q = np.zeros((n,n)); lin = np.zeros(n); const = 0.0
for t in range(T):
    for i in range(N):
        lin[idx(i,t)] += A*(d[i]**2 - 2*S[t]*d[i]) + C*crit[i]
        for j in range(i+1,N):
            Q[idx(i,t),idx(j,t)] += 2*A*d[i]*d[j]
    const += A*S[t]**2
for i in range(N):
    Q[idx(i,0),idx(i,1)] += B
def qubo(x):
    x=np.asarray(x); return const + lin@x + x@Q@x
# sanity
rng=np.random.default_rng(0)
for _ in range(5):
    x=rng.integers(0,2,n); assert abs(qubo(x)-cost(x))<1e-6, (qubo(x),cost(x))

# Ising: x = (1 - z)/2
h = np.zeros(n); J = {}; off = const
for k in range(n):
    off += lin[k]/2; h[k] += -lin[k]/2
for a in range(n):
    for b in range(n):
        if Q[a,b]!=0:
            off += Q[a,b]/4; h[a] += -Q[a,b]/4; h[b] += -Q[a,b]/4; J[(a,b)] = J.get((a,b),0)+Q[a,b]/4
norm = max(np.abs(h).max(), max(abs(v) for v in J.values()))
terms=[]
for k in range(n):
    if h[k]: s=['I']*n; s[n-1-k]='Z'; terms.append((''.join(s),h[k]/norm))
for (a,b),v in J.items():
    s=['I']*n; s[n-1-a]='Z'; s[n-1-b]='Z'; terms.append((''.join(s),v/norm))
H = SparsePauliOp.from_list(terms).simplify()

# Classical exact baseline (brute force 2^12)
t0=time.time()
allx = np.array(list(itertools.product([0,1],repeat=n)))
costs = np.array([qubo(x) for x in allx])
best = costs.min(); opt_set = set(np.where(np.isclose(costs,best))[0]); bf_time=time.time()-t0
xbest = allx[costs.argmin()]
# Greedy baseline (operator-style: shed largest non-critical groups first, rotate)
def greedy():
    x=np.zeros((T,N)); used=set()
    for t in range(T):
        order=sorted(range(N), key=lambda i:(crit[i], -d[i]))
        tot=0
        for i in order:
            if i in used: continue
            if tot>=S[t]: break
            x[t,i]=1; tot+=d[i]; used.add(i)
    return x.flatten()
xg = greedy(); cg = qubo(xg)

def bits_to_index(bitstr):  # qiskit little-endian: rightmost char is qubit 0
    x = np.array([int(c) for c in bitstr[::-1]])
    return x
# Fast exact QAOA simulation: cost Hamiltonian is diagonal, so phase = exp(-i*gamma*Hdiag); mixer = product of RX(2beta).
# Basis index k <-> allx[k] (allx[k][0] is qubit 0 here, independent of qiskit ordering).
Hdiag = (costs - off)/norm  # Ising energies scaled same as H (constant dropped)
def mixer(psi, beta):
    psi = psi.reshape([2]*n)
    c, s_ = np.cos(beta), -1j*np.sin(beta)
    for q in range(n):
        a = np.take(psi,0,axis=q); b = np.take(psi,1,axis=q)
        psi = np.stack([c*a + s_*b, s_*a + c*b], axis=q)
    return psi.reshape(-1)
def qaoa_state(th, p):
    g, bt = th[:p], th[p:]
    psi = np.full(2**n, 2**(-n/2), dtype=complex)
    for k in range(p):
        psi = psi*np.exp(-1j*g[k]*Hdiag); psi = mixer(psi, bt[k])
    return psi
def run_qaoa(p, restarts=20, seed=1):
    rng=np.random.default_rng(seed)
    energy = lambda th: float(np.real(np.sum(np.abs(qaoa_state(th,p))**2*Hdiag)))
    bestres=None
    for r in range(restarts):
        x0 = np.concatenate([rng.uniform(0,0.5,p), rng.uniform(0,np.pi/2,p)])
        res = minimize(energy, x0, method="COBYLA", options={"maxiter":2000})
        if bestres is None or res.fun<bestres.fun: bestres=res
    pr = np.abs(qaoa_state(bestres.x,p))**2
    p_opt = float(sum(pr[k] for k in opt_set))
    exp_cost = float(pr@costs)
    rs=np.random.default_rng(seed); shots = rs.choice(2**n, size=1024, p=pr/pr.sum())
    sampled = float(costs[shots].min())
    return dict(p=p, params=bestres.x.tolist(), p_opt=p_opt, uniform_p_opt=len(opt_set)/2**n, exp_cost=exp_cost,
                best_sampled_cost=sampled, top_state_cost=float(costs[pr.argmax()]),
                found_opt_in_1024=bool(np.isclose(sampled,best)), nfev=int(bestres.nfev))

def describe(x):
    x=np.asarray(x).reshape(T,N); out=[]
    for t in range(T):
        g=[groups[i] for i in range(N) if x[t,i]]
        out.append(dict(block=["18-21h","21-24h"][t], shed=g, shed_MW=float(d@x[t]), target_MW=float(S[t]), critical_weight=float(crit@x[t])))
    return out

results=dict(n_qubits=n, optimum_cost=float(best), n_optimal=len(opt_set), bruteforce_s=bf_time,
             optimum_schedule=describe(xbest), greedy_cost=float(cg), greedy_schedule=describe(xg),
             mean_random_cost=float(costs.mean()))
results["qaoa"]=[]
for p in [1,2,3,4,5]:
    t0=time.time(); r=run_qaoa(p); r["seconds"]=time.time()-t0; results["qaoa"].append(r); print(r, flush=True)
json.dump(results, open("/workspace/ghana-quantum/dumsor/results.json","w"), indent=2, default=float)
print(json.dumps({k:v for k,v in results.items() if k!="qaoa"}, indent=1, default=float))
