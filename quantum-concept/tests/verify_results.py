"""Re-run the pre-registered confirmation analysis (PREREGISTRATION.md) and check H1/H2/H3.
Run:  python quantum-concept/tests/verify_results.py [--from-probs]
Default: copies results/estimates_confirm.csv into a temporary folder, runs the project's own
pooled.py there (so the saved results are not overwritten), compares with results/pooled_confirm.json,
and prints the H1/H2/H3 table. --from-probs first re-runs analyze.py on the raw simulated
probabilities (results/probs_confirm.npz) to regenerate the estimates (slower)."""
import json, os, shutil, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
fails = 0
def check(name, ok, detail):
    global fails
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"); fails += (not ok)
tmp = tempfile.mkdtemp(prefix="sazne_verify_")
os.makedirs(os.path.join(tmp, "results"))
for f in ["pooled.py", "analyze.py"]: shutil.copy(os.path.join(ROOT, f), tmp)
shutil.copytree(os.path.join(ROOT, "sazne"), os.path.join(tmp, "sazne"))
t0 = time.time()
if "--from-probs" in sys.argv:
    shutil.copy(os.path.join(ROOT, "results", "probs_confirm.npz"), os.path.join(tmp, "results"))
    meta = os.path.join(ROOT, "results", "probs_confirm_meta.json")
    if os.path.exists(meta): shutil.copy(meta, os.path.join(tmp, "results"))
    print("$ python analyze.py --probs results/probs_confirm.npz --shot-seeds 20 --tag confirm   (in a temp copy)")
    r = subprocess.run([sys.executable, "analyze.py", "--probs", "results/probs_confirm.npz", "--shot-seeds", "20", "--tag", "confirm"], cwd=tmp, capture_output=True, text=True)
    print((r.stdout.strip().splitlines() or [""])[0], "... (analyze.py exit code", r.returncode, ")")
    import pandas as pd
    key = ["backend", "inst", "steps", "cond", "scheme", "obs", "shot_seed", "method"]
    a = pd.read_csv(os.path.join(tmp, "results", "estimates_confirm.csv")).sort_values(key).reset_index(drop=True)
    b = pd.read_csv(os.path.join(ROOT, "results", "estimates_confirm.csv")).sort_values(key).reset_index(drop=True)
    check("regenerated estimates match saved estimates_confirm.csv", len(a) == len(b) and (a[key] == b[key]).all().all() and float((a.est - b.est).abs().max()) < 1e-9, f"{len(a)} rows, max |diff| {float((a.est-b.est).abs().max()):.1e}")
else:
    shutil.copy(os.path.join(ROOT, "results", "estimates_confirm.csv"), os.path.join(tmp, "results"))
print("$ python pooled.py confirm   (project's own analysis code, in a temp copy)")
r = subprocess.run([sys.executable, "pooled.py", "confirm"], cwd=tmp, capture_output=True, text=True)
check("pooled.py ran", r.returncode == 0, f"exit code {r.returncode}")
new = json.load(open(os.path.join(tmp, "results", "pooled_confirm.json")))
old = json.load(open(os.path.join(ROOT, "results", "pooled_confirm.json")))
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, p + k + "."))
        elif isinstance(v, list): out.update({f"{p}{k}[{i}]": x for i, x in enumerate(v)})
        else: out[p + k] = v
    return out
fn, fo = flat(new), flat(old)
md = max(abs(fn[k] - fo[k]) for k in fo)
check("re-run reproduces saved pooled_confirm.json", set(fn) == set(fo) and md < 1e-12, f"{len(fo)} numbers compared, max |diff| {md:.1e}")
A, B = "SA-ZNE-loglin", "PS+ZNE-lin"
dr, st, pen = new["drift"], new["static"], new["drift_penalty"]
print(f"\nPre-registered confirmation (seed 3031, {dr['n_problems']} problems per condition); ours = {A}, comparator = {B}")
print(f"{'Hypothesis':<44}{'mean diff':>11}  {'95% CI':<22}{'rel':>7}   verdict")
h1 = dr["ci95"][1] < 0
print(f"{'H1 lower pooled error under drift':<44}{dr['mean_diff']:>+11.5f}  [{dr['ci95'][0]:+.5f}, {dr['ci95'][1]:+.5f}]{dr['rel']*100:>+6.1f}%   {'CONFIRMED' if h1 else 'NOT CONFIRMED (CI includes 0)'}")
h2 = st["ci95"][0] < 0 < st["ci95"][1]
print(f"{'H2 parity with static noise':<44}{st['mean_diff']:>+11.5f}  [{st['ci95'][0]:+.5f}, {st['ci95'][1]:+.5f}]{st['rel']*100:>+6.1f}%   {'CONSISTENT (CI includes 0)' if h2 else 'NOT CONSISTENT'}")
h3 = pen["ci95"][1] < 0
print(f"{'H3 smaller drift penalty':<44}{pen['diff']:>+11.5f}  [{pen['ci95'][0]:+.5f}, {pen['ci95'][1]:+.5f}]          {'CONFIRMED' if h3 else 'NOT CONFIRMED'}")
print(f"   drift penalty: {A} {pen[A]:.5f} vs {B} {pen[B]:.5f}  (ratio {pen[B]/pen[A]:.1f}x)")
print("\nPer cell under drift (rel. change, 95% CI):")
for k, v in new["per_cell_drift"].items():
    print(f"   {k:<24}{v['rel']*100:>+7.1f}%  [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]")
check("verdicts match the report (H1 not confirmed, H2 consistent, H3 confirmed)", (not h1) and h2 and h3, f"H1={'confirmed' if h1 else 'not confirmed'}, H2={'consistent' if h2 else 'not'}, H3={'confirmed' if h3 else 'not'}")
shutil.rmtree(tmp)
print(f"\n(analysis time {time.time()-t0:.1f} s)")
print(f"{'ALL CHECKS PASSED' if fails == 0 else str(fails) + ' CHECK(S) FAILED'}")
print("Note: a FAILED H1 is a real scientific result, not a test failure; the test checks that the verdicts are reproduced.")
sys.exit(1 if fails else 0)
