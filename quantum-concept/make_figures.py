"""Stage 3: PNG figures from results/estimates_<tag>.csv, gains_<tag>.csv, summary_<tag>.json."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

tag = sys.argv[1] if len(sys.argv) > 1 else "main"
os.makedirs("figures", exist_ok=True)
S = json.load(open(f"results/summary_{tag}.json"))
show = ["raw(noRM)", "raw", "PS", "ZNE-exp", "PS+ZNE-lin", "PS+ZNE-exp", "SA-v1-lin", "SA-ZNE", "SA-ZNE-loglin"]
cols = ["#999", "#bbb", "#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3", "#ffd92f", "#a6d854", "#d62728"]

# Fig 1: MAE per condition
keys = list(S)
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
for ax, k in zip(axes.ravel(), keys):
    m = S[k]["methods"]
    y = [m[s]["mean_abs_err"] for s in show]
    e = np.array([[m[s]["mean_abs_err"] - m[s]["ci95"][0], m[s]["ci95"][1] - m[s]["mean_abs_err"]] for s in show]).T
    ax.bar(range(len(show)), y, yerr=e, color=cols, capsize=3)
    ax.set_xticks(range(len(show))); ax.set_xticklabels(show, rotation=60, ha="right", fontsize=8)
    b, c, sch = k.split("|")
    ax.set_title(f"{b.replace('Fake','')} | {c} | {sch} folding", fontsize=10)
    ax.set_ylabel("mean |error| (95% CI)"); ax.set_yscale("log")
fig.suptitle(f"Absolute error vs ideal Trotter value (6-qubit XXZ, {S[keys[0]]['n_problems']} problems per panel, 20 shot-seeds each)")
fig.tight_layout(); fig.savefig("figures/fig1_mae_by_condition.png", dpi=130); plt.close(fig)

# Fig 2: paired scatter SA-ZNE vs best nominal baseline, all conditions
df = pd.read_csv(f"results/estimates_{tag}.csv"); df["ae"] = (df.est - df.ideal).abs()
prob = df.groupby(["backend", "cond", "scheme", "inst", "steps", "obs", "method"]).ae.mean().unstack("method").reset_index()
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, cond in zip(axes, ["static", "drift"]):
    for (b, sch), g in prob[prob.cond == cond].groupby(["backend", "scheme"]):
        ax.scatter(g["PS+ZNE-lin"], g["SA-ZNE-loglin"], s=10, alpha=.6, label=f"{b.replace('Fake','')}/{sch}")
    lim = [1e-4, 0.3]; ax.plot(lim, lim, "k--", lw=1); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("PS+ZNE-lin |error| (nominal λ)"); ax.set_ylabel("SA-ZNE-loglin |error|"); ax.set_title(f"{cond}: below diagonal = SA-ZNE-loglin wins")
    ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/fig2_paired_scatter.png", dpi=130); plt.close(fig)

# Fig 3: does the symmetry gauge track the true amplification?
G = pd.read_csv(f"results/gains_{tag}.csv")
G = G[(G.cond == "drift")]
prand = 20 / 64
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, sch, lams in [(axes[0], "global", [1, 3, 5]), (axes[1], "local", [1, 2, 3])]:
    g = G[G.scheme == sch]
    for b, gb in g.groupby("backend"):
        true_ratio, meas_ratio = [], []
        for j in (2, 3):
            f1 = ((gb.P1 - prand) / (1 - prand)).clip(1e-3, 1); fj = ((gb[f"P{j}"] - prand) / (1 - prand)).clip(1e-3, 1)
            true_ratio += list(gb[f"g{j}"] * lams[j - 1] / (gb.g1 * lams[0]))
            meas_ratio += list(np.log(fj) / np.log(f1))
        ax.scatter(true_ratio, meas_ratio, s=6, alpha=.35, label=b.replace("Fake", ""))
    mx = 8 if sch == "global" else 5
    ax.plot([0, mx], [0, mx], "k--", lw=1); ax.set_xlabel("true gain ratio  g_j·λ_j / (g_1·λ_1)")
    ax.set_ylabel("measured ratio  ln f_j / ln f_1"); ax.set_title(f"{sch} folding under drift"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig3_gauge_tracking.png", dpi=130); plt.close(fig)

# Fig 4: error by Trotter depth (from per-problem table)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, cond in zip(axes, ["static", "drift"]):
    for (b, sch), g in prob[prob.cond == cond].groupby(["backend", "scheme"]):
        d = g.groupby("steps")[["PS+ZNE-lin", "SA-ZNE-loglin"]].mean()
        l, = ax.plot(d.index, d["PS+ZNE-lin"], "--o", ms=4, label=f"{b.replace('Fake','')}/{sch} PS+ZNE-lin")
        ax.plot(d.index, d["SA-ZNE-loglin"], "-s", ms=4, color=l.get_color(), label=f"{b.replace('Fake','')}/{sch} SA-ZNE-loglin")
    ax.set_yscale("log"); ax.set_xlabel("Trotter steps"); ax.set_ylabel("mean |error|"); ax.set_title(cond); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig("figures/fig4_error_vs_depth.png", dpi=130); plt.close(fig)

# Fig 5: drift penalty per cell
fig, ax = plt.subplots(figsize=(8, 4.5)); cells = sorted(prob.groupby(["backend", "scheme"]).groups); w = .38
for k, (mname, col) in enumerate([("PS+ZNE-lin", "#fc8d62"), ("SA-ZNE-loglin", "#d62728")]):
    vals = []
    for b, sch in cells:
        dm = prob[(prob.backend == b) & (prob.scheme == sch)].groupby("cond")[mname].mean()
        vals.append(dm["drift"] - dm["static"])
    ax.bar(np.arange(len(cells)) + (k - .5) * w, vals, w, color=col, label=mname)
ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(len(cells))); ax.set_xticklabels([f"{b.replace('Fake','')}/{s}" for b, s in cells])
ax.set_ylabel("MAE(drift) - MAE(static)"); ax.set_title("Accuracy lost to calibration drift between executions"); ax.legend()
fig.tight_layout(); fig.savefig("figures/fig5_drift_penalty.png", dpi=130); plt.close(fig)
print("figures written")
