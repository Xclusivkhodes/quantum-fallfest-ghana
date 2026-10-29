"""Thiol vs hydroxyl affinity for Hg(II) -- the chemistry behind thiol-functionalised sorbents for galamsey-polluted water.
Reaction:  Hg(OH)2 + 2 H2S -> Hg(SH)2 + 2 H2O   (negative dE = sulfur ligands preferred)
Pipeline: B3LYP/def2-SVP geometry (def2 ECP on Hg) -> HF -> CCSD(T) classical baseline
          -> (4e,4o) active space -> exact CASCI (classical) vs VQE-UCCSD on 8 qubits (Qiskit statevector).
"""
import json, time, numpy as np
from pyscf import gto, scf, dft, cc, mcscf, ao2mo
from pyscf.geomopt.geometric_solver import optimize
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
HA2KCAL=627.509
geoms = {
 "H2O":   "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
 "H2S":   "S 0 0 0; H 0.96 0.93 0; H -0.96 0.93 0",
 "Hg(OH)2":"Hg 0 0 0; O 2.0 0 0; H 2.4 0.9 0; O -2.0 0 0; H -2.4 -0.9 0",
 "Hg(SH)2":"Hg 0 0 0; S 2.35 0 0; H 2.7 1.3 0; S -2.35 0 0; H -2.7 -1.3 0",
}
res={}
for name,g in geoms.items():
    t0=time.time()
    mol=gto.M(atom=g, basis="def2-svp", ecp="def2-svp", verbose=0)
    mf=dft.RKS(mol); mf.xc="b3lyp"
    mol=optimize(mf, maxsteps=100)
    e_b3=dft.RKS(mol).set(xc="b3lyp").run().e_tot
    hf=scf.RHF(mol).run()
    mycc=cc.CCSD(hf).run(); e_t=mycc.ccsd_t()
    # active space: 4 electrons in 4 orbitals (HOMO-1..LUMO+1)
    cas=mcscf.CASCI(hf,4,4); cas.verbose=0; e_casci=cas.kernel()[0]
    h1, ecore = cas.get_h1eff(); h2 = ao2mo.restore(1, cas.get_h2eff(), 4)
    ham = ElectronicEnergy.from_raw_integrals(h1, h2)
    mapper=JordanWignerMapper(); H=mapper.map(ham.second_q_op())
    exact = eigsh(H.to_matrix(sparse=True), k=6, which="SA")[0]
    # pick lowest eigenvalue in the right (2 alpha, 2 beta) sector via VQE; exact CASCI from pyscf as baseline
    ans=UCCSD(4,(2,2),mapper,initial_state=HartreeFock(4,(2,2),mapper))
    # fast exact simulation of the same UCCSD product formula (dense 256x256), then verify on the Qiskit circuit
    from scipy.linalg import expm
    from qiskit.quantum_info import Statevector
    Hm=H.to_matrix(); Gs=[op.to_matrix() for op in ans.operators]
    psi0=Statevector(ans.initial_state).data; nfev=[0]
    def state(t):
        v=psi0.copy()
        for th,G in zip(t,Gs): v=expm(-1j*th*G)@v
        return v
    def f(t):
        nfev[0]+=1; v=state(t); return float(np.real(np.vdot(v,Hm@v)))
    r=minimize(f, np.zeros(ans.num_parameters), method="BFGS", options={"gtol":1e-7})
    est=StatevectorEstimator(); e_check=float(est.run([(ans,H,r.x)]).result()[0].data.evs)
    print(name,"fast-sim",r.fun,"qiskit-circuit",e_check, flush=True)
    r.fun=e_check
    e_vqe=r.fun+ecore
    res[name]=dict(atoms=mol.atom_coords(unit="Angstrom").round(4).tolist(), symbols=[mol.atom_symbol(i) for i in range(mol.natm)],
        E_B3LYP=e_b3, E_HF=hf.e_tot, E_CCSD_T=mycc.e_tot+e_t, E_CASCI=e_casci, E_VQE=e_vqe,
        VQE_minus_CASCI_mHa=(e_vqe-e_casci)*1000, qubits=H.num_qubits, pauli_terms=len(H), uccsd_params=ans.num_parameters,
        nfev=nfev[0], seconds=time.time()-t0)
    print(name, {k:v for k,v in res[name].items() if k not in("atoms",)}, flush=True)
def rxn(key): return (res["Hg(SH)2"][key]+2*res["H2O"][key]-res["Hg(OH)2"][key]-2*res["H2S"][key])*HA2KCAL
summary={k:rxn(k) for k in ["E_B3LYP","E_HF","E_CCSD_T","E_CASCI","E_VQE"]}
print("Reaction energy kcal/mol:", summary)
json.dump(dict(species=res, reaction_kcal_per_mol=summary), open("results.json","w"), indent=2)
