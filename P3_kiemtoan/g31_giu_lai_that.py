"""G31: disjoint hold-out of the calibrated part of the P3 certificate on real bench modules (28/10/2026).

G19 checked exchangeability with leave-one-module-out log-strengths of a model fitted on all modules, so the
held-out module had been in the fit. This script follows the protocol of the certificate itself: the modules are split
into three disjoint sets,
  fit set   -> the lifetime model N_mod (least squares of log N_f on the log swing, heating time and current),
  score set -> log-strengths S_i = log N_obs,i - log N_mod(x_i), whose lower conformal quantile is the bound,
  test set  -> modules never used before; a test module is covered when its log-strength is not below the bound.
The lower bound from n scores at level alpha is their floor(alpha (n + 1))-th smallest value; under exchangeability the
expected coverage is at least 1 - alpha (and at most 1 - alpha + 1/(n + 1) when ties have probability zero).
Experiments:
  A. Hernes et al. 2021 (34 modules, five conditions), random disjoint splits, 12/12/10 and 16/10/8 modules, 4000 splits.
  B. Hernes et al. 2021, one whole run (condition) held out as the test set, fit and score sets split at random from the
     other four runs (2000 splits per run): the test condition was never seen by the model.
  C. Choi et al. 2017 (39 IPMs, six heating times), model log N_f on log heating time, random disjoint splits
     13/13/13, 4000 splits.
Run: python -I g31_giu_lai_that.py   (numpy only). Writes g31_giu_lai_that.json.
"""
import csv
import json
import math
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
rng = np.random.default_rng(20261028)
ALPHAS = (0.10, 0.20)


def read(name):
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def lower_bound(scores, alpha):
    k = math.floor(alpha * (len(scores) + 1))
    return -np.inf if k < 1 else float(np.sort(scores)[k - 1])


def fit_scores(X_fit, y_fit, X, y):
    beta, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    return y - X @ beta


def run_split(X, y, fit, score, test):
    s_score = fit_scores(X[fit], y[fit], X[score], y[score])
    s_test = fit_scores(X[fit], y[fit], X[test], y[test])
    return {a: (float(np.mean(s_test >= lower_bound(s_score, a))), lower_bound(s_score, a)) for a in ALPHAS}


def summarize(covs, bounds):
    out = {}
    for a in ALPHAS:
        c = np.array(covs[a])
        b = np.array([v for v in bounds[a] if np.isfinite(v)])
        out[str(a)] = {"mean_coverage": round(float(c.mean()), 3), "p05_p95_of_split_coverage":
                       [round(float(np.percentile(c, 5)), 3), round(float(np.percentile(c, 95)), 3)],
                       "median_bound_logN": round(float(np.median(b)), 3) if b.size else None,
                       "share_finite": round(float(len(b) / len(bounds[a])), 3)}
    return out


def random_splits(X, y, sizes, reps):
    n = len(y)
    covs, bounds = {a: [] for a in ALPHAS}, {a: [] for a in ALPHAS}
    used = 0
    for _ in range(reps):
        p = rng.permutation(n)
        fit, score, test = p[:sizes[0]], p[sizes[0]:sizes[0] + sizes[1]], p[sizes[0] + sizes[1]:sum(sizes)]
        if np.linalg.matrix_rank(X[fit]) < X.shape[1] or np.linalg.cond(X[fit]) > 1e6:
            continue
        used += 1
        r = run_split(X, y, fit, score, test)
        for a in ALPHAS:
            covs[a].append(r[a][0])
            bounds[a].append(r[a][1])
    return summarize(covs, bounds), used


OUT = {}
H = read("G2_hernes2021_sohoa.csv")
XH = np.column_stack([np.ones(len(H)), np.log([float(r["dT_C"]) for r in H]), np.log([float(r["t_on_s"]) for r in H]),
                      np.log([float(r["I_load_A"]) for r in H])])
yH = np.log([float(r["Nf"]) for r in H])
runs = np.array([int(r["run"]) for r in H])

OUT["A_hernes_random_disjoint"] = {}
for sizes in ((12, 12, 10), (16, 10, 8)):
    res, used = random_splits(XH, yH, sizes, 4000)
    OUT["A_hernes_random_disjoint"][f"{sizes[0]}/{sizes[1]}/{sizes[2]}"] = {"splits_used": used, **res}

# B. one whole run held out: the model never saw the test condition
OUT["B_hernes_run_held_out"] = {}
tot = {a: [0, 0] for a in ALPHAS}
for r in range(1, 6):
    test = np.where(runs == r)[0]
    rest = np.where(runs != r)[0]
    covs, bounds, used = {a: [] for a in ALPHAS}, {a: [] for a in ALPHAS}, 0
    for _ in range(2000):
        p = rng.permutation(rest)
        half = len(p) // 2
        fit, score = p[:half], p[half:]
        if np.linalg.matrix_rank(XH[fit]) < XH.shape[1] or np.linalg.cond(XH[fit]) > 1e6:
            continue
        used += 1
        res = run_split(XH, yH, fit, score, test)
        for a in ALPHAS:
            covs[a].append(res[a][0])
            bounds[a].append(res[a][1])
    OUT["B_hernes_run_held_out"][f"run {r} (n={len(test)})"] = {"splits_used": used, **summarize(covs, bounds)}
    for a in ALPHAS:
        tot[a][0] += float(np.sum(covs[a])) * len(test) / max(len(covs[a]), 1)
        tot[a][1] += len(test)
OUT["B_hernes_run_held_out"]["pooled_over_runs_weighted_by_modules"] = {
    str(a): round(tot[a][0] / tot[a][1], 3) for a in ALPHAS}

# C. Choi et al. 2017
C = read("G2_choi2017_sohoa.csv")
XC = np.column_stack([np.ones(len(C)), np.log([float(r["t_dTj_s"]) for r in C])])
yC = np.log([float(r["Nf"]) for r in C])
res, used = random_splits(XC, yC, (13, 13, 13), 4000)
OUT["C_choi2017_random_disjoint"] = {"13/13/13": {"splits_used": used, **res}}

OUT["note"] = ("Coverage is the share of test modules whose log-strength is not below the conformal lower bound from "
               "the score set; the nominal level is 1 - alpha. Splits whose fit set has a rank-deficient or "
               "ill-conditioned design are skipped. The modules are the digitized lives of G2; suspended modules "
               "of Choi 2018 are not used.")
(HERE / "g31_giu_lai_that.json").write_text(json.dumps(OUT, indent=1), encoding="utf-8", newline="\n")
print(json.dumps(OUT, indent=1))
