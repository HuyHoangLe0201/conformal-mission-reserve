"""G27 (19/10/2026, referee request): out-of-sample check of the stated heating-time rule on real lifetime data.

The stated margin carries lives beyond the longest tested heating time with the slope gamma_t = 0.5: the life at
t_on > t_max is the model life at t_max times (t_on / t_max)^-0.5. Here the rule is tested on lifetime tests that were
not used to fit the model, by fitting on the short heating times of a campaign and predicting its longest ones:

  A. Hernes et al. (2021), the bench of the case study: the five runs fitted without run 4 (t_on 4.5 s, the only run
     above 1.1 s); run 4 is predicted from the fit on runs 1-3 and 5 (t_on 0.15-1.1 s, swing, heating time and current
     as in the case study), (i) by extrapolating the fitted heating-time exponent and (ii) by the stated rule from
     t_max = 1.1 s;
  B. Choi et al. (2017), 39 modules at about 81 K: fitted on the 27 modules with t_on <= 2 s, the 12 modules at 5 and
     10 s predicted in the same two ways (log-life linear in log t_on and log swing);
  C. Junghaenel and Scheuermann (2017), Table 1 at 110 K: lives at 7-60 s predicted from 2 s by the stated rule
     (no scatter: the ratio of observed to predicted life).

For A and B, the conformal lower 10 % bound of the log-life is the prediction plus the floor(0.1 (n + 1))-th smallest
leave-one-out residual of the training modules; reported: the share of the predicted modules above the bound and the
median ratio of their life to the bound. A share near 1 with a large ratio means a conservative rule; a share well below
0.9, an unsafe one.

Run: python g27_ton_ngoai_mau.py  -> g27_ton_ngoai_mau.json (numpy, scipy; adds the user site-packages)
"""
import csv
import json
import math
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "P3_mophong"))
import numpy as np  # noqa: E402

import tuoi_tho as tt  # noqa: E402

GAMMA_T, ALPHA = 0.5, 0.10


def loo(X, y):
    """Least-squares fit and leave-one-out residuals."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    return b, (y - X @ b) / (1 - np.diag(H))


def lower(res, alpha=ALPHA):
    k = math.floor(alpha * (len(res) + 1))
    return -math.inf if k < 1 else float(np.sort(res)[k - 1])


def judge(y_obs, pred, q):
    bound = pred + q
    return dict(n=int(len(y_obs)), covered=float(np.mean(y_obs >= bound)),
                median_ratio_to_bound=float(np.exp(np.median(y_obs - bound))),
                min_ratio_to_bound=float(np.exp(np.min(y_obs - bound))),
                max_ratio_to_bound=float(np.exp(np.max(y_obs - bound))))


def hernes():
    h = tt.hernes()
    tr = h["run"] != 4
    te = ~tr
    X = tt.features(h["dT"], h["ton"], h["I"])
    y = np.log(h["Nf"])
    b, res = loo(X[tr], y[tr])
    q = lower(res)
    t_max = float(h["ton"][tr].max())
    pred_fit = X[te] @ b
    Xc = tt.features(h["dT"][te], np.full(te.sum(), t_max), h["I"][te])
    pred_rule = Xc @ b - GAMMA_T * np.log(h["ton"][te] / t_max)
    return dict(train_runs=[1, 2, 3, 5], test_run=4, t_max_train=t_max, t_test=float(h["ton"][te][0]),
                fitted_ton_exponent=float(b[2]), n_train=int(tr.sum()), q10=q,
                fitted_extrapolation=judge(y[te], pred_fit, q), stated_rule=judge(y[te], pred_rule, q))


def choi():
    rows = list(csv.DictReader(open(HERE / "G2_choi2017_sohoa.csv", encoding="utf-8")))
    t = np.array([float(r["t_dTj_s"]) for r in rows])
    dT = np.array([float(r["dTj_C"]) for r in rows])
    y = np.log(np.array([float(r["Nf"]) for r in rows]))
    X = np.column_stack([np.ones_like(t), np.log(dT), np.log(t)])
    tr, te = t <= 2.0, t > 2.0
    b, res = loo(X[tr], y[tr])
    q = lower(res)
    t_max = float(t[tr].max())
    pred_fit = X[te] @ b
    Xc = np.column_stack([np.ones(te.sum()), np.log(dT[te]), np.full(te.sum(), math.log(t_max))])
    pred_rule = Xc @ b - GAMMA_T * np.log(t[te] / t_max)
    out = dict(n_train=int(tr.sum()), t_max_train=t_max, t_test=sorted({float(v) for v in t[te]}),
               fitted_ton_exponent=float(b[2]), q10=q,
               fitted_extrapolation=judge(y[te], pred_fit, q), stated_rule=judge(y[te], pred_rule, q))
    for tt_ in out["t_test"]:
        k = te & (t == tt_)
        sub = k[te]
        out[f"stated_rule_at_{tt_:g}s"] = judge(y[k], pred_rule[sub], q)
        out[f"fitted_at_{tt_:g}s"] = judge(y[k], pred_fit[sub], q)
    return out


def junghaenel():
    rows = list(csv.DictReader(open(HERE / "G10_junghaenel2017_bang1.csv", encoding="utf-8")))
    out = {}
    for g, mech in ((1, "wire-bond lift-off"), (2, "chip-solder fatigue")):
        r = [x for x in rows if int(x["group"]) == g and float(x["dTj_K"]) == 110]
        t = np.array([float(x["t_on_s"]) for x in r])
        n = np.array([float(x["Nf"]) for x in r])
        ref = float(np.exp(np.mean(np.log(n[t == 2.0]))))
        ratios = {f"{tv:g}": float(np.exp(np.mean(np.log(n[t == tv]))) / (ref * (tv / 2.0) ** -GAMMA_T))
                  for tv in sorted(set(t)) if tv > 2.0}
        out[mech] = dict(ratio_observed_to_stated_rule=ratios)
    return out


def main():
    out = dict(gamma_t=GAMMA_T, alpha=ALPHA, hernes=hernes(), choi2017=choi(), junghaenel2017=junghaenel())
    (HERE / "g27_ton_ngoai_mau.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
