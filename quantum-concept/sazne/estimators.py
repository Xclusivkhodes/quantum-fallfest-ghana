"""Error-mitigation estimators.

Per noise level j we have: raw expectation r_j, symmetry-post-selected expectation s_j,
sector keep probability P_j and nominal scale factor lam_j.
"""
import numpy as np


def poly_extrap(x, y, deg, x0=0.0):
    c = np.polyfit(np.asarray(x, float), np.asarray(y, float), deg)
    return float(np.polyval(c, x0))


def exp_extrap(x, y, x0=0.0):
    """y = a*exp(-b x) fitted in log space; falls back to linear if the data change sign."""
    y = np.asarray(y, float)
    if np.all(y > 0) or np.all(y < 0):
        c = np.polyfit(np.asarray(x, float), np.log(np.abs(y)), 1)
        return float(np.sign(y[0]) * np.exp(np.polyval(c, x0)))
    return poly_extrap(x, y, 1, x0)


def guess_weights(P):
    """GUESS-style (arXiv:2603.13060) minimum-norm extrapolation weights learned from a
    symmetry observable with known ideal value. Here the symmetry observable is the sector
    projector Pi (ideal value 1): sum(w)=1, sum(w*P)=1."""
    P = np.asarray(P, float)
    A = np.vstack([np.ones_like(P), P])
    return A.T @ np.linalg.solve(A @ A.T, np.array([1.0, 1.0]))


def sym_fidelity(P, p_rand):
    """Detectable-fidelity proxy f = (P_keep - p_rand)/(1 - p_rand): equals 1 without noise
    and 0 for a fully mixed state (p_rand = sector dimension / 2^n)."""
    return np.clip((np.asarray(P, float) - p_rand) / (1 - p_rand), 1e-3, 1.0)


def baseline_estimates(lam, r, s, P):
    lam = np.asarray(lam, float)
    w = guess_weights(P)
    return {
        "raw": float(r[0]),
        "PS": float(s[0]),
        "ZNE-lin": poly_extrap(lam, r, 1),
        "ZNE-exp": exp_extrap(lam, r),
        "PS+ZNE-lin": poly_extrap(lam, s, 1),
        "PS+ZNE-rich": poly_extrap(lam, s, 2),
        "PS+ZNE-exp": exp_extrap(lam, s),
        "GUESS-Pi": float(w @ np.asarray(r, float)),
        "GUESS-Pi+PS": float(w @ np.asarray(s, float)),
    }


def sazne_v1(r, s, P):
    """Attempt 1 (failed): abscissa x = -ln P_keep on readout-UNmitigated data, extrapolate to x=0."""
    x = -np.log(np.clip(np.asarray(P, float), 1e-9, None))
    return {"SA-v1-lin": poly_extrap(x, s, 1), "SA-v1-exp": exp_extrap(x, s)}


def sazne_v2(s, P, p_rand):
    """Refined SA-ZNE: readout-mitigated data, abscissa from the in-sector detectable
    fidelity f (measured on the same shots). Primary: s linear in eps_sym = 1 - f -> 0."""
    f = sym_fidelity(P, p_rand)
    eps = 1 - f
    x = -np.log(f)
    return {"SA-ZNE": poly_extrap(eps, s, 1),
            "SA-ZNE-loglin": poly_extrap(x, s, 1),
            "SA-ZNE-exp": exp_extrap(x, s)}
