"""Re-run the optimised p=1 and p=3 QAOA circuits as real Qiskit circuits on Aer (ideal + a simple depolarizing noise model)."""
import json, numpy as np, runpy, sys
sys.argv=['x']
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
exec(open('qaoa_dumsor.py').read().split('# Fast exact QAOA')[0])   # reuse problem + brute force
R=json.load(open('results.json'))
def circ(params,p):
    g,b=params[:p],params[p:]
    qc=QuantumCircuit(n); qc.h(range(n))
    for k in range(p):
        for q in range(n):
            if h[q]: qc.rz(2*g[k]*h[q]/norm, q)
        for (a,c),v in J.items(): qc.rzz(2*g[k]*v/norm, a, c)
        for q in range(n): qc.rx(2*b[k], q)
    qc.measure_all(); return qc
nm=NoiseModel(); nm.add_all_qubit_quantum_error(depolarizing_error(0.001,1),['rz','rx','h']); nm.add_all_qubit_quantum_error(depolarizing_error(0.01,2),['rzz','cx'])
out=[]
for p in [1,3]:
    prm=[r for r in R['qaoa'] if r['p']==p][0]['params']
    qc=circ(prm,p)
    for label,sim in [('ideal',AerSimulator()),('noisy_1q0.1%_2q1%',AerSimulator(noise_model=nm))]:
        tq=transpile(qc,sim,seed_transpiler=1)
        cnt=sim.run(tq,shots=20000,seed_simulator=7).result().get_counts()
        tot=sum(cnt.values()); popt=0; ec=0; bestc=1e18
        for bs,c in cnt.items():
            x=np.array([int(ch) for ch in bs[::-1]]); cc=qubo(x); ec+=c*cc/tot; bestc=min(bestc,cc)
            if np.isclose(cc,best): popt+=c/tot
        out.append(dict(p=p,sim=label,shots=tot,p_opt=popt,exp_cost=ec,best_cost=bestc,twoq_gates=int(sum(1 for i in tq.data if i.operation.num_qubits==2))))
        print(out[-1],flush=True)
json.dump(out,open('aer_results.json','w'),indent=2)
