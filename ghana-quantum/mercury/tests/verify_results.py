"""Verify the key claims of the galamsey mercury VQE project from results.json.
Run:  python ghana-quantum/mercury/tests/verify_results.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
res = json.load(open(os.path.join(ROOT, "results.json")))
HA2KCAL = 627.509474  # 1 hartree in kcal/mol
fails = 0
def check(name, ok, detail):
    global fails
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"); fails += (not ok)
sp = res["species"]
for name, s in sp.items():
    gap = abs(s["E_VQE"] - s["E_CASCI"]) * 1000
    check(f"{name:8s} |VQE - CASCI| tiny", gap < 1e-3, f"{gap:.2e} mHa (chemical accuracy line 1.6 mHa); {s['qubits']} qubits, {s['uccsd_params']} params")
    check(f"{name:8s} VQE not below exact (variational)", s["E_VQE"] >= s["E_CASCI"] - 1e-9, f"VQE {s['E_VQE']:.6f} >= CASCI {s['E_CASCI']:.6f} Ha")
print("\nReaction Hg(OH)2 + 2 H2S -> Hg(SH)2 + 2 H2O (kcal/mol), recomputed from the species energies:")
stored = res["reaction_kcal_per_mol"]
rx = {}
for m in ["E_HF", "E_CASCI", "E_VQE", "E_CCSD_T", "E_B3LYP"]:
    e = (sp["Hg(SH)2"][m] + 2 * sp["H2O"][m] - sp["Hg(OH)2"][m] - 2 * sp["H2S"][m]) * HA2KCAL
    rx[m] = e
    check(f"{m[2:]:7s} reaction energy negative (mercury prefers sulfur)", e < 0, f"{e:.2f} kcal/mol (file {stored[m]:.2f})")
    check(f"{m[2:]:7s} matches stored value", abs(e - stored[m]) < 0.01, f"diff {abs(e-stored[m]):.1e}")
check("VQE vs exact slice reaction energy", abs(rx["E_VQE"] - rx["E_CASCI"]) < 1e-3, f"{abs(rx['E_VQE']-rx['E_CASCI']):.5f} kcal/mol apart")
check("VQE vs CCSD(T) gap is about 1.3 kcal/mol (small active space)", abs(abs(rx["E_VQE"] - rx["E_CCSD_T"]) - 1.28) < 0.05, f"{abs(rx['E_VQE']-rx['E_CCSD_T']):.2f} kcal/mol")
print(f"\n{'ALL CHECKS PASSED' if fails == 0 else str(fails) + ' CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
