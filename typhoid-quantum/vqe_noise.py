import numpy as np, json
np.random.seed(1)
from qiskit_algorithms.utils import algorithm_globals
algorithm_globals.random_seed=1
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit import transpile
from scipy.optimize import minimize
prob = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g").run()
mapper = ParityMapper(num_particles=prob.num_particles)
op = mapper.map(prob.hamiltonian.second_q_op()); nuc=prob.nuclear_repulsion_energy
exact = np.linalg.eigvalsh(op.to_matrix()).min()+nuc
hf = HartreeFock(prob.num_spatial_orbitals, prob.num_particles, mapper)
ans = transpile(UCCSD(prob.num_spatial_orbitals, prob.num_particles, mapper, initial_state=hf), basis_gates=["cx","rz","sx","x"], optimization_level=1)
print("ops", ans.count_ops(), "depth", ans.depth())
ideal = AerEstimator(options={"default_precision":0})
def E(est,x): return float(est.run([(ans,op,x)]).result()[0].data.evs)+nuc
out={"exact":exact,"cx":int(ans.count_ops().get("cx",0)),"depth":ans.depth()}
for label,p2 in [("shots_only",0),("noise_1pct",0.01),("noise_3pct",0.03)]:
    nm=None
    if p2:
        nm=NoiseModel(); nm.add_all_qubit_quantum_error(depolarizing_error(p2,2),["cx"]); nm.add_all_qubit_quantum_error(depolarizing_error(p2/10,1),["sx","x"])
    est=AerEstimator(options={"default_precision":1/np.sqrt(8192),"backend_options":({"noise_model":nm} if nm else {})})
    from qiskit_algorithms.optimizers import SPSA
    r=SPSA(maxiter=150).minimize(lambda x:E(est,x)-nuc, np.zeros(ans.num_parameters))
    final=np.mean([E(est,r.x) for _ in range(20)])
    out[label]=dict(final_noisy=final, err_mHa=(final-exact)*1000, ideal_at_params=E(ideal,r.x), err_params_mHa=(E(ideal,r.x)-exact)*1000)
    print(label,out[label])
json.dump(out,open("noise_results.json","w"),indent=1,default=float)
