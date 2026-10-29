"""Verify the key claims of the typhoid H2 VQE project from its saved result files.
Run from anywhere:  python typhoid-quantum/tests/verify_results.py
Checks results.json (noise-free VQE curve) and noise_results.json (noise study)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CHEM_ACC = 1.6  # milli-Hartree, "chemical accuracy"
res = json.load(open(os.path.join(ROOT, "results.json")))
noise = json.load(open(os.path.join(ROOT, "noise_results.json")))
fails = 0
def check(name, ok, detail):
    global fails
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    fails += (not ok)

curve = res["curve"]
check("problem size", all(c["qubits"] == 2 and c["params"] == 3 for c in curve), f"{len(curve)} bond lengths, 2 qubits, 3 parameters each")
recomputed = [abs(c["vqe"] - c["exact"]) * 1000 for c in curve]
check("stored errors match |VQE - exact|", max(abs(r - c["err_mHa"]) for r, c in zip(recomputed, curve)) < 1e-9, "recomputed from the energies")
worst = max(recomputed); wd = curve[recomputed.index(worst)]["d"]
check("noise-free VQE within chemical accuracy at every distance", worst < CHEM_ACC, f"largest error {worst:.2e} mHa at {wd} A (limit {CHEM_ACC} mHa)")
eq = [c for c in curve if abs(c["d"] - 0.735) < 1e-9][0]
check("error at 0.735 A", abs(eq["vqe"] - eq["exact"]) * 1000 < 1e-5, f"{abs(eq['vqe']-eq['exact'])*1000:.2e} mHa; exact energy {eq['exact']:.6f} Ha")
emin = min(curve, key=lambda c: c["exact"])
check("lowest energy at the normal bond length", abs(emin["d"] - 0.735) < 1e-9, f"minimum {emin['exact']:.6f} Ha at {emin['d']} A")
hf_err = (eq["hf"] - eq["exact"]) * 1000
check("Hartree-Fock (rough guess) is outside chemical accuracy", hf_err > CHEM_ACC, f"HF error {hf_err:.1f} mHa at 0.735 A")

print("\nNoise study at 0.735 A (noise_results.json), circuit: %d CX gates, depth %d" % (noise["cx"], noise["depth"]))
prev = 0
for key, label in [("shots_only", "8,192 shots, perfect chip"), ("noise_1pct", "1% two-qubit gate noise"), ("noise_3pct", "3% two-qubit gate noise")]:
    r = noise[key]; e = abs(r["final_noisy"] - noise["exact"]) * 1000
    check(f"{label}: stored error matches recomputation", abs(e - r["err_mHa"]) < 1e-6, f"{e:.3f} mHa = {e/CHEM_ACC:.1f} x the 1.6 mHa line")
    check(f"{label}: NOT within chemical accuracy (claim: noise ruins the answer)", e > CHEM_ACC, f"{e:.2f} mHa > {CHEM_ACC}")
    check(f"{label}: error grows with noise", e > prev, f"{prev:.2f} -> {e:.2f} mHa"); prev = e
print(f"\n{'ALL CHECKS PASSED' if fails == 0 else str(fails) + ' CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
