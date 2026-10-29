"""Stage 2: finite-shot sampling, readout mitigation, estimators, statistics.

Usage: python analyze.py --probs results/probs.npz --shots 4000 --shot-seeds 20
Conditions:
  static : every noise level executed with the nominal backend noise (g = 1)
  drift  : each folded circuit executed at an independently drawn noise gain g
           (gate errors x g, T1/T2 / g; readout unchanged), emulating calibration drift
           between jobs.
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sazne.observables import make_observables
from sazne.sim import apply_readout
from sazne.estimators import baseline_estimates, sazne_v1, sazne_v2

FOLDS = {"global": [1, 3, 5], "local": [1, 2, 3]}
NOMINAL = ["ZNE-lin", "ZNE-exp", "PS+ZNE-lin", "PS+ZNE-rich", "PS+ZNE-exp", "GUESS-Pi", "GUESS-Pi+PS"]
SA = ["SA-ZNE", "SA-ZNE-loglin", "SA-ZNE-exp", "SA-v1-lin"]


def calibrate_readout(mats, shots, rng):
    """Tensored readout calibration from two circuits (all-0, all-1), finite shots."""
    est = []
    for m in mats:
        e01 = rng.binomial(shots, m[0, 1]) / shots
        e10 = rng.binomial(shots, m[1, 0]) / shots
        est.append(np.array([[1 - e01, e01], [e10, 1 - e10]]))
    return est


def mitigate(q, cal):
    n = len(cal); t = q.reshape([2] * n)
    for i, m in enumerate(cal):
        ax = n - 1 - i
        t = np.moveaxis(np.tensordot(t, np.linalg.inv(m), axes=([ax], [0])), -1, ax)
    return t.reshape(-1)


def stats_from(q, o, sector):
    tot = q.sum(); k = q[sector].sum()
    return q @ o / tot, q[sector] @ o[sector] / k, k / tot


def boot_ci(x, rng, B=4000):
    x = np.asarray(x); m = rng.choice(x, size=(B, len(x))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="results/probs.npz")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--cal-shots", type=int, default=4000)
    ap.add_argument("--shot-seeds", type=int, default=20)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    recs = np.load(a.probs, allow_pickle=True)["records"]
    groups = defaultdict(dict)
    for rec in recs:
        groups[(rec["backend"], int(rec["idx"]), int(rec["steps"]))][float(rec["g"])] = rec
    n = int(np.log2(len(recs[0]["ideal"])))
    obs, sector = make_observables(n)
    p_rand = sector.mean()
    rows, gains_log = [], []
    for (bname, idx, steps), byg in sorted(groups.items()):
        gs = sorted(byg); base = byg[1.0]; mats = base["readout"]
        bseed = sum(map(ord, bname))
        for cond in ("static", "drift"):
            for scheme, lams in FOLDS.items():
                for ss in range(a.shot_seeds):
                    rng = np.random.default_rng([bseed, idx, steps, ss, len(cond), len(scheme)])
                    gsel = [1.0] * len(lams) if cond == "static" else [gs[k] for k in rng.integers(0, len(gs), len(lams))]
                    cal = calibrate_readout(mats, a.cal_shots, rng)
                    counts = []
                    for j, g in enumerate(gsel):
                        p = apply_readout(byg[g][scheme][j], mats)
                        counts.append(rng.multinomial(a.shots, p) / a.shots)
                    for oname, o in obs.items():
                        ideal = float(base["ideal"] @ o)
                        um = [stats_from(c, o, sector) for c in counts]
                        mm = [stats_from(mitigate(c, cal), o, sector) for c in counts]
                        r, s, P = map(np.array, zip(*mm))
                        ru, su, Pu = map(np.array, zip(*um))
                        est = baseline_estimates(lams, r, s, P)
                        est["raw(noRM)"] = float(ru[0])
                        est.update(sazne_v1(ru, su, Pu))
                        est.update(sazne_v2(s, P, p_rand))
                        for m, e in est.items():
                            rows.append((bname, idx, steps, cond, scheme, oname, ss, m, e, ideal))
                        if oname == "stag_mag":
                            gains_log.append((bname, idx, steps, cond, scheme, ss, *gsel, *P))
    os.makedirs("results", exist_ok=True)
    path = f"results/estimates_{a.tag}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["backend", "inst", "steps", "cond", "scheme", "obs", "shot_seed", "method", "est", "ideal"])
        w.writerows(rows)
    with open(f"results/gains_{a.tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["backend", "inst", "steps", "cond", "scheme", "shot_seed", "g1", "g2", "g3", "P1", "P2", "P3"])
        w.writerows(gains_log)
    print("wrote", path, len(rows))
    summarize(path, a.tag)


def summarize(path, tag):
    import pandas as pd
    df = pd.read_csv(path)
    df["abserr"] = (df.est - df.ideal).abs()
    rng = np.random.default_rng(0)
    keys = ["backend", "cond", "scheme"]
    prob = df.groupby(keys + ["inst", "steps", "obs", "method"]).agg(mae=("abserr", "mean"), sd=("est", "std")).reset_index()
    out = {}
    for k, g in prob.groupby(keys):
        piv = g.pivot_table(index=["inst", "steps", "obs"], columns="method", values="mae")
        sdp = g.pivot_table(index=["inst", "steps", "obs"], columns="method", values="sd")
        st = {m: {"mean_abs_err": float(piv[m].mean()), "ci95": boot_ci(piv[m].values, rng),
                  "median_abs_err": float(piv[m].median()), "mean_shot_sd": float(sdp[m].mean())} for m in piv.columns}
        best = min(NOMINAL, key=lambda m: st[m]["mean_abs_err"])
        comps = {}
        for m in SA:
            d = piv[m].values - piv[best].values
            comps[m] = {"vs": best, "mean_diff": float(d.mean()), "ci95": boot_ci(d, rng),
                        "rel_change": float(d.mean() / piv[best].mean()), "win_rate": float((d < 0).mean()),
                        "wilcoxon_p": float(wilcoxon(piv[m].values, piv[best].values).pvalue)}
        # by depth
        bydepth = {int(s): {m: float(gg[m].mean()) for m in ["raw", "PS", best, "SA-ZNE"]}
                   for s, gg in piv.reset_index().groupby("steps")}
        out["|".join(k)] = {"n_problems": int(len(piv)), "methods": st, "best_nominal": best, "comparisons": comps, "by_steps": bydepth}
    json.dump(out, open(f"results/summary_{tag}.json", "w"), indent=1)
    for k, v in out.items():
        print(f"\n== {k}  (problems={v['n_problems']}, best nominal-lambda baseline={v['best_nominal']})")
        for m, s in sorted(v["methods"].items(), key=lambda t: t[1]["mean_abs_err"]):
            print(f"  {m:14s} MAE={s['mean_abs_err']:.4f} [{s['ci95'][0]:.4f},{s['ci95'][1]:.4f}] shot-sd={s['mean_shot_sd']:.4f}")
        for m, c in v["comparisons"].items():
            print(f"  {m} - {c['vs']}: diff={c['mean_diff']:+.4f} CI[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] rel={c['rel_change']:+.0%} win={c['win_rate']:.2f} p={c['wilcoxon_p']:.2g}")


if __name__ == "__main__":
    main()
