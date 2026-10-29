"""Pooled tests (see PREREGISTRATION.md): SA-ZNE-loglin vs fixed comparator PS+ZNE-lin.
Usage: python pooled.py <tag>   -> results/pooled_<tag>.json"""
import json, sys
import numpy as np, pandas as pd
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
A, B = "SA-ZNE-loglin", "PS+ZNE-lin"
df = pd.read_csv(f"results/estimates_{tag}.csv"); df["ae"] = (df.est - df.ideal).abs()
df = df[df.method.isin([A, B, "SA-ZNE", "PS", "raw"])]
prob = df.groupby(["backend", "cond", "scheme", "inst", "steps", "obs", "method"]).ae.mean().unstack("method").reset_index()
rng = np.random.default_rng(1)
def boot(x, B=10000):
    m = rng.choice(x, size=(B, len(x))).mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
out = {}
for cond in ["drift", "static"]:
    p = prob[prob.cond == cond]; d = (p[A] - p[B]).values
    out[cond] = {"n_problems": int(len(d)), "mae_" + A: float(p[A].mean()), "mae_" + B: float(p[B].mean()),
                 "mae_SA-ZNE(primary,eps-lin)": float(p["SA-ZNE"].mean()), "mae_PS": float(p.PS.mean()), "mae_raw": float(p.raw.mean()),
                 "mean_diff": float(d.mean()), "ci95": boot(d), "rel": float(d.mean() / p[B].mean()), "win_rate": float((d < 0).mean())}
key = ["backend", "scheme", "inst", "steps", "obs"]
m = prob[prob.cond == "drift"].set_index(key)[[A, B]] - prob[prob.cond == "static"].set_index(key)[[A, B]]
dd = (m[A] - m[B]).values
out["drift_penalty"] = {A: float(m[A].mean()), B: float(m[B].mean()), "diff": float(dd.mean()), "ci95": boot(dd)}
out["per_cell_drift"] = {}
for (b, s), g in prob[prob.cond == "drift"].groupby(["backend", "scheme"]):
    d = (g[A] - g[B]).values
    out["per_cell_drift"][f"{b}|{s}"] = {"mean_diff": float(d.mean()), "ci95": boot(d), "rel": float(d.mean() / g[B].mean())}
json.dump(out, open(f"results/pooled_{tag}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
