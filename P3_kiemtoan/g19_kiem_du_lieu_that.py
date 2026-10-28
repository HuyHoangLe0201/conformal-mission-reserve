"""G19: real-data checks of the calibrated part of the P3 certificate (17/10/2026).

The calibrated part of the certificate is the conformal lower quantile of the log-strength. Its validity needs the
log-strength of the field device to be exchangeable with the bench scores (Assumption A1 of the manuscript: one law on the
reference conditions, one population per campaign and lot). This script checks that on the digitized per-module data:
  1. Hernes et al. 2021 (34 modules, five runs = five reference conditions): leave-one-module-out log-strengths S of the
     lifetime model fitted on all modules (as in g3_tref_bien.py); for each run, the conformal lower alpha-quantile of the
     S of the other four runs, and the share of the held-out run's modules above it.
  2. Choi et al. 2017 (39 IPMs, six swing durations): residuals of ln N_f about the mean of their condition; leave one
     condition out in the same way.
  3. Choi 2017 against Choi 2018 (same IPM type, other campaign; failures only, as in thongke_G2.py): the bound computed
     from all residuals of one campaign, and the share of the other campaign's residuals above it, in both directions.
     This is a lot shift: the within-condition variance differs by a factor of 9.9 between the campaigns.
The conformal lower bound from n calibration values at level alpha is their floor(alpha (n + 1))-th smallest value; a
test value is covered when it is not below the bound. Under exchangeability the expected coverage is at least 1 - alpha.

Run: python -I g19_kiem_du_lieu_that.py   (numpy and scipy from the user site). Writes g19_kiem_du_lieu_that.json.
"""
import csv
import json
import math
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())  # keeps the script runnable with python -I
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).parent
ALPHAS = (0.10, 0.20)
OUT = {"alphas": list(ALPHAS)}


def read(name):
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def lower_bound(cal, alpha):
    """Conformal lower alpha-quantile of the calibration values; -inf when floor(alpha (n + 1)) = 0."""
    k = math.floor(alpha * (len(cal) + 1))
    return -math.inf if k == 0 else float(np.sort(cal)[k - 1])


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def leave_one_group_out(values, groups, alpha):
    covered, total, per = 0, 0, {}
    for g in np.unique(groups):
        test, cal = values[groups == g], values[groups != g]
        q = lower_bound(cal, alpha)
        k = int(np.sum(test >= q))
        per[str(g)] = f"{k}/{test.size}"
        covered, total = covered + k, total + test.size
    return {"covered": covered, "n": total, "coverage": round(covered / total, 3), "wilson95": wilson(covered, total),
            "per_group": per}


# 1. Hernes: leave-one-module-out log-strengths of the lifetime model fitted on all 34 modules (g3_tref_bien.py)
H = read("G2_hernes2021_sohoa.csv")
runs = np.array([int(r["run"]) for r in H])
X = np.column_stack([np.ones(len(H)), np.log([float(r["dT_C"]) for r in H]), np.log([float(r["t_on_s"]) for r in H]),
                     np.log([float(r["I_load_A"]) for r in H])])
y = np.log([float(r["Nf"]) for r in H])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
hat = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
S = (y - X @ beta) / (1 - hat)
OUT["hernes_leave_one_run_out"] = {str(a): leave_one_group_out(S, runs, a) for a in ALPHAS}


# 2./3. Choi: residuals of ln N_f about the mean of their condition
def centered(rows, keep=lambda r: True):
    g = {}
    for r in rows:
        if keep(r):
            g.setdefault(r["condition"], []).append(math.log(float(r["Nf"])))
    res, lab = [], []
    for c, v in g.items():
        v = np.array(v)
        res.extend(v - v.mean())
        lab.extend([c] * v.size)
    return np.array(res), np.array(lab)


r17, c17 = centered(read("G2_choi2017_sohoa.csv"))
r18, _ = centered(read("G2_choi2018_bang2.csv"), keep=lambda r: r["failed"] == "1")
OUT["choi2017_leave_one_condition_out"] = {str(a): leave_one_group_out(r17, c17, a) for a in ALPHAS}
cross = {}
for a in ALPHAS:
    q17, q18 = lower_bound(r17, a), lower_bound(r18, a)
    k18, k17 = int(np.sum(r18 >= q17)), int(np.sum(r17 >= q18))
    cross[str(a)] = {
        "bound_from_2017_on_2018": {"covered": k18, "n": int(r18.size), "coverage": round(k18 / r18.size, 3),
                                    "wilson95": wilson(k18, r18.size)},
        "bound_from_2018_on_2017": {"covered": k17, "n": int(r17.size), "coverage": round(k17 / r17.size, 3),
                                    "wilson95": wilson(k17, r17.size)},
        "bound_ln": {"from_2017": round(q17, 3), "from_2018": round(q18, 3)}}
OUT["choi_cross_campaign"] = cross
OUT["sd_ln_residual"] = {"hernes_S": round(float(S.std(ddof=1)), 4), "choi2017": round(float(r17.std(ddof=1)), 4),
                         "choi2018": round(float(r18.std(ddof=1)), 4)}
OUT["note"] = ("Choi residuals are taken about the condition means, so the cross-campaign check covers a change of "
               "scatter only, not a shift of the mean life between lots.")

for k, v in OUT.items():
    print(k, json.dumps(v, ensure_ascii=False))
(HERE / "g19_kiem_du_lieu_that.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                                                  newline="\n")
