"""G3: reference set T_ref, log-strength residuals (draft of Figure F3) and the stated extrapolation margin of P3.

Inputs (this folder): G2_hernes2021_sohoa.csv, G2_hernes2021pcim_sohoa.csv, G2_choi2017_sohoa.csv,
G2_choi2018_bang2.csv, G10_junghaenel2017_bang1.csv (Table 1 of Junghaenel, Scheuermann 2017, the primary
source of the pulse-duration data; the re-plot in Scheuermann, Junghaenel 2018, digitized by
sohoa_scheuermann2018.py, is about 6 % lower and is kept for comparison only).

Sections:
  A. Lifetime model of the Hernes IHV modules, ln N_f = c + b1 ln dT + b3 ln t_on + b4 ln I (least squares on 34
     modules), against the published refits; lack of fit against free run means; leave-one-module-out log-strength
     residuals S; tests that the law of S does not depend on the run (k-sample Anderson-Darling with permutation
     p-value, Kruskal-Wallis, Fligner-Killeen) or on dT, t_on, I (Spearman); the same with run 4 (solder fatigue)
     left out; leave-one-run-out predictions; the authors' own damage at failure (PCIM Fig. 4) as an independent
     residual; the split design of the certificate (half of each run fits N_mod, the rest are scores).
  B. T_ref and the box of tested conditions.
  C. Stated margin: swing slope gamma below dT_T, heating-time slope gamma_t above the tested t_on, Arrhenius bound
     for T_min above the tested value, current clipped to the tested range; evidence for each value.
  D. Draft of Figure F3 (F3_nhap.pdf, F3_nhap.png).

Run: python g3_tref_bien.py   (numpy, scipy, matplotlib). Writes g3_tref_bien.json. Seed 20261016.
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
SEED = 20261016
rng = np.random.default_rng(SEED)
OUT = {"seed": SEED}
PERM = stats.PermutationMethod(n_resamples=9999, random_state=rng)
K_B = 8.617333e-5  # eV/K


def read(name):
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = float(res @ res / dof)
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta, res, s2, cov, dof


def design(rows):
    return np.column_stack([np.ones(len(rows)), np.log([r["dT"] for r in rows]), np.log([r["ton"] for r in rows]),
                            np.log([r["I"] for r in rows])])


def loo_residuals(X, y):
    """Leave-one-out (PRESS) residuals of least squares: e_i / (1 - h_ii)."""
    beta, res, *_ = ols(X, y)
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    return res / (1 - np.diag(H))


def run_tests(S, runs, label):
    groups = [S[runs == r] for r in np.unique(runs)]
    ad = stats.anderson_ksamp(groups, method=PERM)
    kw = stats.kruskal(*groups)
    fk = stats.fligner(*groups)
    print(f"{label}: k-sample Anderson-Darling p = {ad.pvalue:.3f}; Kruskal-Wallis p = {kw.pvalue:.3f}; "
          f"Fligner-Killeen p = {fk.pvalue:.3f}")
    return {"anderson_ksamp_p": round(float(ad.pvalue), 4), "kruskal_p": round(float(kw.pvalue), 4),
            "fligner_p": round(float(fk.pvalue), 4)}


# =================================================================================================================
print("A. Hernes IHV 3.3 kV modules: lifetime model and log-strength residuals")
H = [dict(run=int(r["run"]), I=float(r["I_load_A"]), ton=float(r["t_on_s"]), dT=float(r["dT_C"]), Nf=float(r["Nf"]))
     for r in read("G2_hernes2021_sohoa.csv")]
runs = np.array([r["run"] for r in H])
X, y = design(H), np.log([r["Nf"] for r in H])
beta, res, s2, cov, dof = ols(X, y)
se = np.sqrt(np.diag(cov))
corr = cov / np.outer(se, se)
print(f"fit on 34 modules: b1 (dT) = {beta[1]:.2f} +- {se[1]:.2f}, b3 (t_on) = {beta[2]:.2f} +- {se[2]:.2f}, "
      f"b4 (I) = {beta[3]:.2f} +- {se[3]:.2f}; residual sd {math.sqrt(s2):.3f} (ln)")
print("  published refit on all runs: -2.07, -1.42, -4.1; without run 3: -2.24, -1.35, -3.76")
print(f"  correlation of estimates: b1-b3 {corr[1, 2]:.2f}, b1-b4 {corr[1, 3]:.2f}, b3-b4 {corr[2, 3]:.2f}")
OUT["model"] = {"b1_dT": round(beta[1], 3), "b3_ton": round(beta[2], 3), "b4_I": round(beta[3], 3),
                "se": [round(v, 3) for v in se[1:]], "resid_sd_ln": round(math.sqrt(s2), 4),
                "corr_b1_b3": round(corr[1, 2], 3), "corr_b1_b4": round(corr[1, 3], 3),
                "corr_b3_b4": round(corr[2, 3], 3)}
# lack of fit: free run means plus a common within-run swing slope (6 parameters) against the model (4 parameters)
Xf = np.column_stack([(runs == r).astype(float) for r in range(1, 6)] + [np.log([r["dT"] for r in H])])
_, res_f, *_ = ols(Xf, y)
rss_r, rss_f = float(res @ res), float(res_f @ res_f)
F_lof = ((rss_r - rss_f) / 2) / (rss_f / (len(y) - 6))
p_lof = float(stats.f.sf(F_lof, 2, len(y) - 6))
print(f"lack of fit against free run means: F(2, {len(y) - 6}) = {F_lof:.2f}, p = {p_lof:.3f}")
OUT["lack_of_fit"] = {"F": round(F_lof, 3), "df": [2, len(y) - 6], "p": round(p_lof, 4)}

S = loo_residuals(X, y)
for r in range(1, 6):
    s = S[runs == r]
    print(f"  run {r}: n = {s.size}, mean S {s.mean():+.3f}, sd {s.std(ddof=1):.3f}")
OUT["loo_S"] = {"sd": round(float(S.std(ddof=1)), 4),
                "per_run_mean": {str(r): round(float(S[runs == r].mean()), 4) for r in range(1, 6)}}
OUT["tests_all_runs"] = run_tests(S, runs, "T_ref = runs 1-5, leave-one-out S")
sp = {}
for name, v in (("dT", X[:, 1]), ("t_on", X[:, 2]), ("I", X[:, 3])):
    rho, p = stats.spearmanr(v, S)
    sp[name] = {"rho": round(float(rho), 3), "p": round(float(p), 4)}
print("  Spearman S against ln dT, ln t_on, ln I: " + "; ".join(f"{k} rho {v['rho']:+.2f} (p {v['p']:.2f})"
                                                                for k, v in sp.items()))
OUT["spearman_all_runs"] = sp
# run 4 is the only run that failed mainly by solder fatigue (5 of 8 by the thermal-resistance criterion)
keep = runs != 4
H4 = [r for r, k in zip(H, keep) if k]
X4, y4 = design(H4), np.log([r["Nf"] for r in H4])
b4, *_ = ols(X4, y4)
S4 = loo_residuals(X4, y4)
print(f"without run 4: b1 = {b4[1]:.2f}, b3 = {b4[2]:.2f}, b4 = {b4[3]:.2f}")
OUT["tests_without_run4"] = run_tests(S4, runs[keep], "T_ref = runs 1, 2, 3, 5, leave-one-out S")
OUT["tests_without_run4"]["coefficients"] = [round(v, 3) for v in b4[1:]]

# leave one run out: prediction of a whole held-out condition
loro = {}
for r in range(1, 6):
    tr = runs != r
    bt, *_ = ols(X[tr], y[tr])
    pred = X[~tr] @ bt
    mean_res = float(np.mean(y[~tr] - pred))
    # the paper's deviation: model at the run's mean stress against the run's mean N_f
    xm = np.array([1, math.log(np.mean([h["dT"] for h, t in zip(H, tr) if not t])), X[~tr][0, 2], X[~tr][0, 3]])
    dev = (math.exp(xm @ bt) - np.mean(np.exp(y[~tr]))) / np.mean(np.exp(y[~tr]))
    # is the error consistent with the reduced fit's own prediction uncertainty for the run mean?
    _, _, s2t, covt, _ = ols(X[tr], y[tr])
    xbar = X[~tr].mean(axis=0)
    se_mean = math.sqrt(float(xbar @ covt @ xbar) + s2t / int((~tr).sum()))
    loro[str(r)] = {"mean_log_residual": round(mean_res, 4), "deviation_percent": round(100 * dev, 2),
                    "b1": round(float(bt[1]), 3), "z": round(mean_res / se_mean, 2)}
    print(f"  leave run {r} out: b1 = {bt[1]:.2f}; held-out mean log residual {mean_res:+.3f} "
          f"(z = {mean_res / se_mean:+.2f} against the reduced fit's prediction se); deviation at mean stress "
          f"{100 * dev:+.1f} %")
OUT["leave_one_run_out"] = loro
# model form between tested conditions: the largest non-conservative error when a whole run is held out
EPS0 = max(0.0, -min(v["mean_log_residual"] for v in loro.values()))
print(f"  largest non-conservative held-out error (model form between tested runs): {EPS0:.3f}")
OUT["model_form_margin"] = round(EPS0, 3)
# prediction standard error of ln N_mod: on the tested ridge against corners of the tested box
pse = {}
for name, (dt_, t_, i_) in {"run 4 condition (70 K, 4.5 s, 890 A)": (70, 4.5, 890),
                            "run 3 condition (28 K, 0.15 s, 1250 A)": (28, 0.15, 1250),
                            "box corner (26 K, 4.5 s, 890 A)": (26, 4.5, 890),
                            "box corner (70 K, 0.15 s, 1250 A)": (70, 0.15, 1250)}.items():
    xv = np.array([1, math.log(dt_), math.log(t_), math.log(i_)])
    pse[name] = round(float(math.sqrt(xv @ cov @ xv)), 3)
print("  prediction se of ln N_mod: " + "; ".join(f"{k}: {v:.2f}" for k, v in pse.items()))
OUT["prediction_se"] = pse
Z95 = 1.645
TON_LONG, I_LOW = 4.5, 890.0  # the field point after clipping heating time and current to the tested box


def se_line(cov_, dts_):
    """Prediction se of ln N_mod along the swing at the clipped heating time and current."""
    return np.array([math.sqrt(float(v @ cov_ @ v)) for v in
                     (np.array([1, math.log(d), math.log(TON_LONG), math.log(I_LOW)]) for d in dts_)])


DT_GRID = [70, 40, 26, 15, 10]
se_full = se_line(cov, DT_GRID)
print("  statistical margin 1.645 se along the swing at 4.5 s and 890 A (full fit): "
      + ", ".join(f"{d} K: {Z95 * s:.2f}" for d, s in zip(DT_GRID, se_full)))
OUT["stat_margin_full_fit"] = {str(d): round(Z95 * s, 3) for d, s in zip(DT_GRID, se_full)}
# a band valid simultaneously along the swing (Working-Hotelling, two coefficients on the line) instead of 1.645
WH = {f"dof {v}": round(math.sqrt(2 * stats.f.ppf(0.95, 2, v)), 3) for v in (30, 12, 6)}
print("  simultaneous factor along the swing line instead of 1.645 (Working-Hotelling, 95 %): "
      + ", ".join(f"{k}: {v:.2f}" for k, v in WH.items()) + " (full fit, half split, two per run)")
OUT["working_hotelling_factor"] = WH
# implied exponent error of the one extrapolation in swing (run 3, below every other run)
dT_fit_min = min(h["dT"] for h in H if h["run"] != 3)
dT_r3 = math.exp(np.mean(X[runs == 3, 1]))
gamma_obs = abs(loro["3"]["mean_log_residual"]) / math.log(dT_fit_min / dT_r3)
print(f"  run 3 extrapolation: from {dT_fit_min:.1f} K down to {dT_r3:.1f} K, implied exponent error "
      f"{gamma_obs:.2f}")
OUT["run3_implied_exponent_error"] = round(gamma_obs, 3)

# the authors' damage at failure (PCIM Fig. 4) for runs 1-5: an independent residual of their own fit
P = read("G2_hernes2021pcim_sohoa.csv")
lw = np.array([math.log(float(p["damage"])) for p in P if int(p["test"]) <= 5])
lw_runs = np.array([int(p["test"]) for p in P if int(p["test"]) <= 5])
OUT["tests_authors_W"] = run_tests(lw, lw_runs, "authors' damage at failure, runs 1-5 (in-sample fit)")
OUT["tests_authors_W"]["sd_lnW"] = round(float(lw.std(ddof=1)), 4)

# split design of the certificate: floor(n_r/2) modules of each run fit N_mod, the others are scores
n_fit = sum(int((runs == r).sum()) // 2 for r in range(1, 6))
m = len(H) - n_fit
thr = {}
for a in (0.10, 0.05):
    thr[str(a)] = {"paired_needs_m": math.ceil(1 / a - 1 - 1e-12), "union_needs_n1": math.ceil(2 / a - 1 - 1e-12),
                   "pac_needs_m_delta_eq_alpha": math.ceil(math.log(a) / math.log(1 - a) - 1e-12),
                   "paired_finite": m >= math.ceil(1 / a - 1 - 1e-12),
                   "union_finite": m >= math.ceil(2 / a - 1 - 1e-12),
                   "pac_finite": m >= math.ceil(math.log(a) / math.log(1 - a) - 1e-12)}
print(f"split design: {n_fit} modules fit N_mod, m = {m} scores; " + "; ".join(
    f"alpha {a}: paired {'finite' if t['paired_finite'] else 'infinite'}, union "
    f"{'finite' if t['union_finite'] else 'infinite'}, PAC {'finite' if t['pac_finite'] else 'infinite'}"
    for a, t in thr.items()))
B_SPLIT = 2000
smin, p_ad = np.empty(B_SPLIT), np.empty(B_SPLIT)
for b in range(B_SPLIT):
    fit_idx, sc_idx = [], []
    for r in range(1, 6):
        idx = rng.permutation(np.flatnonzero(runs == r))
        k = idx.size // 2
        fit_idx += list(idx[:k])
        sc_idx += list(idx[k:])
    bt, *_ = ols(X[fit_idx], y[fit_idx])
    sc = y[sc_idx] - X[sc_idx] @ bt
    smin[b] = sc.min()
    groups = [sc[runs[sc_idx] == r] for r in range(1, 6)]
    p_ad[b] = stats.kruskal(*groups).pvalue
print(f"  over {B_SPLIT} random splits: smallest score (the alpha = 0.10 paired index uses it) median "
      f"{np.median(smin):+.3f}, 5-95 % {np.quantile(smin, 0.05):+.3f} to {np.quantile(smin, 0.95):+.3f}; "
      f"Kruskal-Wallis across runs on the scores p < 0.05 in {np.mean(p_ad < 0.05):.3f} of splits")
OUT["split_design"] = {"n_fit": n_fit, "m_scores": m, "thresholds": thr,
                       "min_score_median": round(float(np.median(smin)), 4),
                       "min_score_q05_q95": [round(float(np.quantile(smin, q)), 4) for q in (0.05, 0.95)],
                       "fraction_kruskal_p_below_0.05": round(float(np.mean(p_ad < 0.05)), 4)}
# a leaner fit: two modules of each run fit N_mod (10), 24 scores; spread of the scores against the half split, and
# the statistical margin along the swing at the clipped heating time, which the fitting half pays for
sd_half, sd_two = np.empty(B_SPLIT), np.empty(B_SPLIT)
se_half, se_two = np.empty((B_SPLIT, len(DT_GRID))), np.empty((B_SPLIT, len(DT_GRID)))
for b in range(B_SPLIT):
    for nf_run, store, sstore in ((None, sd_half, se_half), (2, sd_two, se_two)):
        fit_idx, sc_idx = [], []
        for r in range(1, 6):
            idx = rng.permutation(np.flatnonzero(runs == r))
            k = idx.size // 2 if nf_run is None else nf_run
            fit_idx += list(idx[:k])
            sc_idx += list(idx[k:])
        bt, _, _, covb, _ = ols(X[fit_idx], y[fit_idx])
        store[b] = np.std(y[sc_idx] - X[sc_idx] @ bt, ddof=1)
        sstore[b] = se_line(covb, DT_GRID)
m_half = Z95 * np.median(se_half, axis=0)
m_two = Z95 * np.median(se_two, axis=0)
print(f"  score spread (sd of S): half split median {np.median(sd_half):.3f}; two per run fit (24 scores, union and "
      f"PAC finite at alpha 0.10) median {np.median(sd_two):.3f}")
print("  statistical margin along the swing at 4.5 s, 890 A (median over splits), half split: "
      + ", ".join(f"{d} K: {v:.2f}" for d, v in zip(DT_GRID, m_half)) + "; two per run: "
      + ", ".join(f"{d} K: {v:.2f}" for d, v in zip(DT_GRID, m_two)))
OUT["split_design"]["score_sd_half_split"] = round(float(np.median(sd_half)), 4)
OUT["split_design"]["score_sd_two_per_run_fit"] = round(float(np.median(sd_two)), 4)
OUT["split_design"]["stat_margin_half_split"] = {str(d): round(float(v), 3) for d, v in zip(DT_GRID, m_half)}
OUT["split_design"]["stat_margin_two_per_run"] = {str(d): round(float(v), 3) for d, v in zip(DT_GRID, m_two)}

# =================================================================================================================
print("\nB. T_ref and the box of tested conditions")
dT_T = min(h["dT"] for h in H)
box = {"dT_T_K": round(dT_T, 1), "dT_max_K": round(max(h["dT"] for h in H), 1),
       "t_on_s": [min(h["ton"] for h in H), max(h["ton"] for h in H)], "I_A": [min(h["I"] for h in H), max(h["I"] for h in H)],
       "Tj_min_C": 60}
print(f"T_ref = the five runs of Hernes et al. (2021); lowest recorded swing {dT_T:.1f} K (stated dT_T = 26 K); "
      f"t_on {box['t_on_s'][0]}-{box['t_on_s'][1]} s; I {box['I_A'][0]:.0f}-{box['I_A'][1]:.0f} A; T_j,min 60 C only")
OUT["T_ref_box"] = box
DT_T, TON_T, TMIN_T = 26.0, 4.5, 60.0

# =================================================================================================================
print("\nC. Stated margin")
b1 = -beta[1]
print(f"swing: fitted exponent beta = {b1:.2f}; one-sided 95 % statistical bound on its error 1.645 se = "
      f"{1.645 * se[1]:.2f} (charged separately, as the statistical margin at the clipped point); run-3 extrapolation "
      f"error {gamma_obs:.2f}; measurement-delay correction (paper) 0.13; mechanism switch (Scheuermann 2018) "
      f"4.42 - 3.21 = 1.21")
GAMMA = 1.2
print(f"  stated physical gamma = {GAMMA} below {DT_T:.0f} K: with the statistical part left out, margined damage "
      f"scales as dT^{b1 - GAMMA:.2f} (derating pays while gamma < beta = {b1:.2f})")
for dt in (20, 15, 10, 5):
    print(f"  dT = {dt:2d} K: margin {GAMMA * math.log(DT_T / dt):.2f} (factor {math.exp(GAMMA * math.log(DT_T / dt)):.2f}"
          f" on the damage of the cycle)")
OUT["swing"] = {"beta": round(b1, 3), "stat_bound_1.645se": round(1.645 * se[1], 3), "gamma": GAMMA,
                "beta_minus_gamma": round(b1 - GAMMA, 3), "dT_T": DT_T,
                "sensitivity": [0, 0.6, GAMMA, round(b1, 2)]}

# heating time: slopes of ln N_f against ln t_on in data with t_on as the only varied factor
# pulse-duration data, Table 1 of Junghaenel, Scheuermann (2017); "figure" keeps the panel of the 2018 re-plot
SC = [dict(r, figure="6" if r["dTj_K"] == "70" else "7") for r in read("G10_junghaenel2017_bang1.csv")]
slopes = {}
for fig in ("6", "7"):
    for grp, name in (("1", "wire bond"), ("2", "solder")):
        pts = [(math.log(float(s["t_on_s"])), math.log(float(s["Nf"]))) for s in SC if s["figure"] == fig and s["group"] == grp]
        xx, yy = np.array(pts).T
        sl = stats.linregress(xx, yy)
        key = f"junghaenel2017_{'70K' if fig == '6' else '110K'}_{name.replace(' ', '_')}"
        slopes[key] = {"slope": round(float(sl.slope), 3), "se": round(float(sl.stderr), 3),
                       "t_on_range_s": [round(float(np.exp(xx.min())), 2), round(float(np.exp(xx.max())), 1)],
                       "n": int(xx.size)}
C17 = read("G2_choi2017_sohoa.csv")
xc = np.log([float(c["t_dTj_s"]) for c in C17])
yc = np.log([float(c["Nf"]) for c in C17])
slc = stats.linregress(xc, yc)
slopes["choi2017_swing_duration"] = {"slope": round(float(slc.slope), 3), "se": round(float(slc.stderr), 3),
                                     "t_on_range_s": [0.59, 10.0], "n": int(xc.size)}
slopes["cips08"] = {"slope": -0.463, "t_on_range_s": [1, 15]}
slopes["hernes_refit"] = {"slope": round(float(beta[2]), 3), "t_on_range_s": [0.15, 4.5],
                          "note": "confounded with the current term"}
for k, v in slopes.items():
    print(f"  t_on slope {k:34s} {v['slope']:+.3f}" + (f" +- {v['se']:.3f}" if "se" in v else "")
          + f"  over {v['t_on_range_s'][0]}-{v['t_on_range_s'][1]} s")
GAMMA_T = 0.5
TON_MISSION = 90.0
print(f"  stated gamma_t = {GAMMA_T} above {TON_T} s: margin at the mission t_on of {TON_MISSION:.0f} s = "
      f"{GAMMA_T * math.log(TON_MISSION / TON_T):.2f} (factor {(TON_MISSION / TON_T) ** GAMMA_T:.1f}); "
      f"with 0.29: factor {(TON_MISSION / TON_T) ** 0.29:.1f}; with 1.0: factor {(TON_MISSION / TON_T) ** 1.0:.0f}; "
      f"model extrapolated with b3 = {beta[2]:.2f}: factor {(TON_MISSION / TON_T) ** (-beta[2]):.0f}")
OUT["heating_time"] = {"slopes": slopes, "gamma_t": GAMMA_T, "t_on_T": TON_T, "mission_t_on": TON_MISSION,
                       "margin_at_mission": round(GAMMA_T * math.log(TON_MISSION / TON_T), 3),
                       "sensitivity": [0.29, GAMMA_T, 1.0]}

# minimum temperature: Arrhenius bound above the single tested value
EA = {"wire bond, T_j,max (Scheuermann Fig. 4)": 0.069, "solder, T_jm (Scheuermann Fig. 5)": 0.097,
      "CIPS08 beta2 = 1285 K": 1285 * K_B, "SiC solder, Hoffmann 2023 beta2 = 1537 K": 1537 * K_B,
      "solder, T_j,max (Scheuermann Fig. 4)": 0.159}
EA_STATED = 0.159
tm = {}
for T in (65, 70, 80):
    tm[str(T)] = round(EA_STATED / K_B * (1 / (TMIN_T + 273.15) - 1 / (T + 273.15)), 4)
print("  T_min: activation energies (eV) " + "; ".join(f"{k}: {v:.3f}" for k, v in EA.items()))
print(f"  stated E_a = {EA_STATED} eV ({EA_STATED / K_B:.0f} K) above {TMIN_T:.0f} C: margin "
      + ", ".join(f"{k} C: {v:.3f}" for k, v in tm.items()) + "; no credit below 60 C")
OUT["min_temperature"] = {"Ea_eV_sources": {k: round(v, 4) for k, v in EA.items()}, "Ea_stated_eV": EA_STATED,
                          "margin": tm}
OUT["current"] = {"rule": "clip to the tested 890-1250 A; no credit below 890 A; above 1250 A the model exponent applies"}

# margin budget for a long, small cycle against the calibrated scatter (natural-log units of damage)
q10 = float(np.sort(S)[math.floor(0.10 * (S.size + 1)) - 1])
budget = {"calibrated scatter, -q_S at alpha 0.10": round(-q10, 3), "Miner factor, -ln 0.85": round(-math.log(0.85), 3),
          "model statistics at 10 K, 4.5 s, 890 A (half split)": round(float(m_half[DT_GRID.index(10)]), 3),
          "swing 10 K, gamma 1.2": round(GAMMA * math.log(DT_T / 10), 3),
          "heating time 90 s, gamma_t 0.5": round(GAMMA_T * math.log(TON_MISSION / TON_T), 3),
          "T_min 70 C, E_a 0.159 eV": tm["70"]}
stated = sum(v for k, v in budget.items() if not k.startswith("calibrated"))
print(f"  stated part {stated:.2f} against calibrated part {budget['calibrated scatter, -q_S at alpha 0.10']:.2f}: "
      f"the stated part is {100 * stated / (stated + budget['calibrated scatter, -q_S at alpha 0.10']):.0f} % of the "
      f"log budget for this cycle (damage factor {math.exp(stated):.0f})")
OUT["budget_stated_share"] = round(stated / (stated + budget["calibrated scatter, -q_S at alpha 0.10"]), 3)
print("  budget (ln of the damage factor): " + "; ".join(f"{k}: {v:.2f}" for k, v in budget.items()))
OUT["budget"] = budget

# =================================================================================================================
print("\nD. Draft of Figure F3")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.5})
COL = {1: "#0072BD", 2: "#D95319", 3: "#EDB120", 4: "#7E2F8E", 5: "#77AC30"}
fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.4))
dts = np.array([h["dT"] for h in H])
tons = np.array([h["ton"] for h in H])

ax = axs[0, 0]
for r in range(1, 6):
    ax.scatter(dts[runs == r], S[runs == r], s=14, color=COL[r], label=f"run {r}")
xx = np.linspace(8, DT_T, 50)
for g, ls in ((0.6, ":"), (GAMMA, "--")):
    ax.plot(xx, -g * np.log(DT_T / xx), "k", ls=ls, lw=0.9, label=fr"$-\gamma\,\ln(\Delta T_T/\Delta T)$, $\gamma$={g}")
grid = np.geomspace(8, 74, 60)
ax.plot(grid, -Z95 * se_line(cov, grid), color="0.45", ls="-.", lw=0.9, label="statistical (full fit), 4.5 s, 890 A")
ax.axvline(DT_T, color="grey", lw=0.6)
ax.axhline(0, color="grey", lw=0.4)
ax.set_xscale("log")
ax.set_xticks([8, 10, 15, 20, 26, 40, 70])
ax.set_xticklabels(["8", "10", "15", "20", "26", "40", "70"])
ax.set_xlabel(r"$\Delta T$ (K)")
ax.set_ylabel(r"log-strength $S$ (leave-one-out)")
ax.set_title(r"(a) Hernes 2021: $S$ against swing; margin below $\Delta T_T$")
ax.set_ylim(-1.6, 0.45)
h_, l_ = ax.get_legend_handles_labels()
ax.legend([h for h, l in zip(h_, l_) if not l.startswith("run")],
          [l for l in l_ if not l.startswith("run")], loc="lower right", fontsize=5.8,
          title="runs coloured as in (b)", title_fontsize=5.8)

ax = axs[0, 1]
for r in range(1, 6):
    ax.scatter(tons[runs == r], S[runs == r], s=14, color=COL[r], label=f"run {r}")
ax.axvspan(0.15, TON_T, color="0.92", zorder=0, label="tested $t_{on}$")
tt = np.linspace(TON_T, 120, 80)
for g, ls in ((0.29, ":"), (GAMMA_T, "--"), (1.0, "-.")):
    ax.plot(tt, -g * np.log(tt / TON_T), "k", ls=ls, lw=0.9, label=fr"$\gamma_t$={g}")
ax.axvline(TON_MISSION, color="firebrick", lw=0.8)
ax.text(TON_MISSION * 0.9, -2.35, "mission\n~90 s", color="firebrick", ha="right", fontsize=6.5)
ax.axhline(0, color="grey", lw=0.4)
ax.set_xscale("log")
ax.set_ylim(-2.5, 0.5)
ax.set_xlabel(r"$t_{on}$ (s)")
ax.set_title(r"(b) Hernes 2021: $S$ against $t_{on}$; margin above 4.5 s")
ax.legend(loc="lower left", ncol=2)

ax = axs[0, 2]
mk = {("6", "1"): ("s", "#C0392B", "wire bond, 70 K"), ("6", "2"): ("D", "#2C3E91", "solder, 70 K"),
      ("7", "1"): ("s", "#E67E73", "wire bond, 110 K"), ("7", "2"): ("D", "#7F8FD1", "solder, 110 K")}
for (fg, gr), (m_, c_, lab) in mk.items():
    pts = [(float(s["t_on_s"]), float(s["Nf"])) for s in SC if s["figure"] == fg and s["group"] == gr]
    ref = np.exp(np.mean([math.log(n) for t, n in pts if abs(t - 2) < 0.1]))
    ax.scatter([t for t, _ in pts], [n / ref for _, n in pts], marker=m_, s=14, color=c_, label=lab)
c17t = np.array([float(c["t_dTj_s"]) for c in C17])
c17n = np.array([float(c["Nf"]) for c in C17])
ref17 = math.exp(np.mean(np.log(c17n[np.isclose(c17t, 2.0)])))
ax.scatter(c17t, c17n / ref17, marker="o", s=10, facecolors="none", edgecolors="0.35", label=r"Choi 2017, $t_{\Delta T_j}$")
tt = np.logspace(math.log10(0.05), math.log10(120), 50)
for g, ls in ((0.29, ":"), (0.463, "--"), (1.42, "-.")):
    ax.plot(tt, (tt / 2) ** (-g), "k", ls=ls, lw=0.8, label=f"slope -{g}")
ax.axvline(TON_MISSION, color="firebrick", lw=0.8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(0.2, 30)
ax.set_xlabel(r"$t_{on}$ (s)")
ax.set_ylabel(r"$N_f / N_f(2\,\mathrm{s})$")
ax.set_title(r"(c) $t_{on}$ alone: Junghaenel 2017; Choi 2017")
ax.legend(loc="upper right", ncol=2, fontsize=5.8)

ax = axs[1, 0]
c18 = read("G2_choi2018_bang2.csv")
x18 = np.log([float(c["dTj_C"]) for c in c18])
y18 = np.log([float(c["Nf"]) for c in c18])
fail = np.array([c["failed"] == "1" for c in c18])
s18 = stats.linregress(x18[fail], y18[fail])
r18 = y18 - (s18.intercept + s18.slope * x18)
r17 = yc - (slc.intercept + slc.slope * xc)
cond17 = sorted(set(c17t), reverse=True)
for i, t in enumerate(cond17):
    v = r17[np.isclose(c17t, t)]
    ax.scatter(np.full(v.size, i) + rng.uniform(-0.15, 0.15, v.size), v, s=10, color="#2C3E91")
d18 = [70, 60, 40]
for j, d in enumerate(d18):
    sel = np.isclose(np.exp(x18), d)
    xs_ = np.full(sel.sum(), len(cond17) + 1 + j) + rng.uniform(-0.15, 0.15, sel.sum())
    ax.scatter(xs_[fail[sel]], r18[sel][fail[sel]], s=10, color="#C0392B")
    ax.scatter(xs_[~fail[sel]], r18[sel][~fail[sel]], s=14, facecolors="none", edgecolors="#C0392B", marker="^")
ax.set_xticks(list(range(len(cond17))) + [len(cond17) + 1 + j for j in range(3)])
ax.set_xticklabels([f"{t:g} s" for t in cond17] + [f"{d} K" for d in d18], rotation=45)
ax.axhline(0, color="grey", lw=0.4)
ax.set_ylabel(r"residual of $\ln N_f$")
ax.set_title("(d) one IPM type, two campaigns: Choi 2017 (blue), 2018 (red)")
ax.text(0.02, 0.04, "open triangles: suspended (lower bounds)", transform=ax.transAxes, fontsize=6.5)

ax = axs[1, 1]
Wt = {t: np.array([float(p["damage"]) for p in P if int(p["test"]) == t]) for t in range(1, 8)}
const = np.concatenate([Wt[t] for t in range(1, 6)])
qW = float(np.sort(const)[math.floor(0.10 * (const.size + 1)) - 1])
for t in range(1, 8):
    v = Wt[t]
    ax.scatter(np.full(v.size, t) + rng.uniform(-0.15, 0.15, v.size), v, s=12,
               color=COL.get(t, "#B8860B"), marker="o" if t <= 5 else ("D" if t == 6 else "^"))
ax.axhline(qW, color="k", lw=0.9, label=fr"lower 10 % bound, $\kappa_M$ = 1 ({qW:.2f})")
ax.axhline(0.85 * qW, color="k", ls="--", lw=0.9, label=fr"$\kappa_M$ = 0.85 ({0.85 * qW:.2f})")
ax.set_xticks(range(1, 8))
ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7"])
ax.set_xlabel("test (6: 40 then 70 K; 7: 70 then 40 K)")
ax.set_ylabel("damage at failure")
ax.set_ylim(0.65, 1.52)
ax.set_title("(e) Hernes PCIM 2021: Miner sum, mixed loading")
ax.legend(loc="upper right")

ax = axs[1, 2]
labels = list(budget)
vals = [budget[k] for k in labels]
ax.barh(range(len(vals)), vals, color=["#4C72B0", "#8172B2", "#937860", "#C44E52", "#C44E52", "#DD8452"])
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(["calibrated scatter\n" + r"$-q_S$, $\alpha$ = 0.10", "Miner factor\n" + r"$-\ln\kappa_M$",
                    "model statistics\n10 K at 4.5 s, 890 A", r"swing 10 K, $\gamma$ = 1.2",
                    r"$t_{on}$ 90 s, $\gamma_t$ = 0.5",
                    r"$T_{min}$ 70 °C"])
for i, v in enumerate(vals):
    ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=6.5)
ax.set_xlim(0, max(vals) * 1.2)
ax.invert_yaxis()
ax.set_xlabel("log damage factor")
ax.set_title("(f) margin budget, one long small cycle")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"F3_nhap.{ext}", dpi=200)
print("written F3_nhap.pdf, F3_nhap.png")

(HERE / "g3_tref_bien.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
print("written g3_tref_bien.json")
