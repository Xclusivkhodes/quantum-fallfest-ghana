import numpy as np, json, time
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SLSQP
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from scipy.optimize import minimize
HA2KCAL=627.509474
def run(d, shots=None):
    prob = PySCFDriver(atom=f"H 0 0 0; H 0 0 {d}", basis="sto3g").run()
    mapper = ParityMapper(num_particles=prob.num_particles)
    op = mapper.map(prob.hamiltonian.second_q_op())
    nuc = prob.nuclear_repulsion_energy
    exact = np.linalg.eigvalsh(op.to_matrix()).min()
    hf = HartreeFock(prob.num_spatial_orbitals, prob.num_particles, mapper)
    ans = UCCSD(prob.num_spatial_orbitals, prob.num_particles, mapper, initial_state=hf)
    from qiskit import transpile
    ans = transpile(ans, basis_gates=["cx","rz","sx","x","h","ry","rx"], optimization_level=1)
    if shots: est = AerEstimator(options={"default_precision":1/np.sqrt(shots),"run_options":{"seed":42}})
    else: est = AerEstimator(options={"default_precision":0})
    def f(x):
        return float(est.run([(ans, op, x)]).result()[0].data.evs)
    r = minimize(f, np.zeros(ans.num_parameters), method="COBYLA", options={"maxiter":200})
    # hf energy
    return dict(d=d, qubits=op.num_qubits, params=ans.num_parameters, nuc=nuc,
                exact=exact+nuc, vqe=r.fun+nuc, hf=f(np.zeros(ans.num_parameters))+nuc, nfev=r.nfev)
res=[run(d) for d in [0.4,0.5,0.6,0.735,0.9,1.1,1.3,1.6,2.0,2.5]]
for r in res: r["err_mHa"]=(r["vqe"]-r["exact"])*1000
noisy=run(0.735, shots=8192); noisy["err_mHa"]=(noisy["vqe"]-noisy["exact"])*1000
json.dump(dict(curve=res,shots8192=noisy), open("results.json","w"), indent=1, default=float)
for r in res: print(r)
print("SHOTS", noisy)
