"""G2/G3: statistics on the per-module power-cycling data collected for P3 (15/10/2026).

Inputs (this folder):
  G2_hernes2021_sohoa.csv      N_f of 34/36 IHV 3.3 kV modules, five constant-stress runs (IET Power Electron. 2021)
  G2_hernes2021pcim_sohoa.csv  Miner damage at end of life of 48/49 modules of the same type, five constant-stress and
                               two mixed-stress runs (PCIM Europe 2021, Fig. 4)
  G2_choi2017_sohoa.csv        N_f of 39 transfer-molded 600 V/30 A IPMs, six swing durations (IEEE TPEL 2017, Fig. 7)
  G2_choi2018_bang2.csv        N_f of 18 IPMs of the same type, three swings, two suspended units (Microelectron.
                               Reliab. 2018, Tables 1-2)
Published numbers used directly: Choi, Blaabjerg, Ma (APEC 2017), Table III and Section III.C.

Sections:
  1. scatter of log N_f within test conditions, by source; Fligner-Killeen tests of equal scatter;
  2. Choi 2018: Weibull with right censoring per condition; inverse-power-law Weibull with a common shape, compared
     with the published n = 5.6545 and beta = 6.864; likelihood-ratio test of a common shape, parametric-bootstrap
     p-value;
  3. Miner's rule under mixed swings (Hernes PCIM): strength W (damage at failure, constant runs) against the Miner sum
     at failure of the mixed runs; conformal check of the lower strength quantile, with and without a Miner factor;
  4. Miner's rule under mixed swing durations, t_dTj = 1/f_out (Choi APEC): observed lives against the predicted
     B-lives.

Run: python thongke_G2.py   (numpy, scipy). Writes thongke_G2.json. About 90 s (parametric bootstrap).
"""
import csv
import json
import math
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())  # keeps the script runnable with python -I
import numpy as np  # noqa: E402
from scipy import optimize, stats  # noqa: E402
from scipy.special import comb, logsumexp  # noqa: E402

HERE = Path(__file__).parent
SEED = 20261015
rng = np.random.default_rng(SEED)
OUT = {"seed": SEED}


def read(name):
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def groups(rows, key, val, cast=float, keep=lambda r: True):
    g = {}
    for r in rows:
        if keep(r):
            g.setdefault(r[key], []).append(cast(r[val]))
    return {k: np.array(v) for k, v in g.items()}


def scatter_block(name, g):
    """Within-condition standard deviation of log10 N_f and the Fligner-Killeen test of equal scatter."""
    sd = {k: float(np.std(np.log10(v), ddof=1)) for k, v in g.items()}
    resid = np.concatenate([np.log10(v) - np.log10(v).mean() for v in g.values()])
    pooled = float(np.sqrt(np.sum(resid ** 2) / (resid.size - len(g))))
    fk = stats.fligner(*[np.log10(v) for v in g.values()])
    print(f"{name}: n = {sum(len(v) for v in g.values())} in {len(g)} conditions; sd log10 N_f per condition "
          + ", ".join(f"{k}: {s:.3f}" for k, s in sd.items())
          + f"; pooled {pooled:.3f}; Fligner-Killeen {fk.statistic:.2f}, p = {fk.pvalue:.3f}")
    return {"n": {k: int(len(v)) for k, v in g.items()}, "sd_log10": {k: round(s, 4) for k, s in sd.items()},
            "pooled_sd_log10": round(pooled, 4), "fligner_stat": round(float(fk.statistic), 3),
            "fligner_p": round(float(fk.pvalue), 4)}, resid


# ---------------------------------------------------------------------------------------------------------------
print("1. Scatter of log N_f within test conditions")
hernes = read("G2_hernes2021_sohoa.csv")
choi17 = read("G2_choi2017_sohoa.csv")
choi18 = read("G2_choi2018_bang2.csv")
OUT["scatter"] = {}
OUT["scatter"]["hernes2021"], _ = scatter_block("Hernes 2021 (IHV 3.3 kV)", groups(hernes, "run", "Nf"))
OUT["scatter"]["choi2017"], r17 = scatter_block("Choi 2017 (IPM, dTj 81 C)", groups(choi17, "condition", "Nf"))
OUT["scatter"]["choi2018_failed_only"], r18 = scatter_block(
    "Choi 2018 (IPM, dTj 40-70 C, failures only)", groups(choi18, "condition", "Nf", keep=lambda r: r["failed"] == "1"))
# same module type, two campaigns: equal within-condition scatter?
df17, df18 = r17.size - 6, r18.size - 3
F = (np.sum(r18 ** 2) / df18) / (np.sum(r17 ** 2) / df17)
p_F = float(stats.f.sf(F, df18, df17))
fk2 = stats.fligner(r17, r18)
print(f"Choi 2018 against Choi 2017, pooled within-condition variance ratio {F:.2f} (F test p = {p_F:.4f}; "
      f"Fligner-Killeen on residuals p = {fk2.pvalue:.4f})")
OUT["scatter"]["choi2018_vs_choi2017"] = {"variance_ratio": round(float(F), 3), "F_test_p": round(p_F, 5),
                                          "fligner_p": round(float(fk2.pvalue), 5)}
# like for like in swing duration: both campaigns have a t_dTj = 1 s condition (dTj 70 C in 2018, 81 C in 2017)
a18 = np.log10(groups(choi18, "condition", "Nf", keep=lambda r: r["failed"] == "1")["1"])
a17 = np.log10(groups(choi17, "condition", "Nf")["4"])
F1 = np.var(a18, ddof=1) / np.var(a17, ddof=1)
p_F1 = float(stats.f.sf(F1, a18.size - 1, a17.size - 1))
print(f"same swing duration 1 s (Choi 2018 condition 1, n = {a18.size}; Choi 2017 condition 4, n = {a17.size}): "
      f"variance ratio {F1:.2f}, F test p = {p_F1:.4f}")
OUT["scatter"]["choi2018_vs_choi2017_1s"] = {"variance_ratio": round(float(F1), 3), "F_test_p": round(p_F1, 5)}


# ---------------------------------------------------------------------------------------------------------------
def weibull_profile(beta, data):
    """Profile log-likelihood of a common Weibull shape over groups with free scales, right censoring allowed.
    data: list of (times, failed) per group. Given beta, eta_g^beta = sum(t^beta) / r_g."""
    ll = 0.0
    for t, d in data:
        r = d.sum()
        lt = np.log(t)
        ll += r * math.log(beta) - r * (logsumexp(beta * lt) - math.log(r)) + (beta - 1) * lt[d].sum() - r
    return ll


def fit_shape(data):
    res = optimize.minimize_scalar(lambda lb: -weibull_profile(math.exp(lb), data), bounds=(-1, 5), method="bounded",
                                   options={"xatol": 1e-8})
    return math.exp(res.x), -res.fun


def scales(beta, data):
    return [math.exp((logsumexp(beta * np.log(t)) - math.log(d.sum())) / beta) for t, d in data]


def lr_common_shape(data):
    b0, ll0 = fit_shape(data)
    fits = [fit_shape([g]) for g in data]
    return 2 * (sum(ll for _, ll in fits) - ll0), b0, [b for b, _ in fits]


print("\n2. Choi 2018: Weibull with right censoring")
c18 = {}
for r in choi18:
    c18.setdefault(int(r["condition"]), []).append((float(r["Nf"]), r["failed"] == "1", float(r["dTj_C"])))
data18 = [(np.array([t for t, _, _ in v]), np.array([d for _, d, _ in v])) for _, v in sorted(c18.items())]
dT18 = [v[0][2] for _, v in sorted(c18.items())]
lr, b_common, b_sep = lr_common_shape(data18)
eta_common = scales(b_common, data18)
print("per-condition shape (MLE, censoring included): " + ", ".join(f"dTj {d:.0f} C: {b:.2f}" for d, b in
                                                                       zip(dT18, b_sep)))
print(f"common shape with free scales: {b_common:.3f}; scales " + ", ".join(f"{e:,.0f}" for e in eta_common))


def ipl_negll(p):
    """Inverse-power-law Weibull, eta = exp(a) dT^-n, common shape; theta = exp(-a beta) profiled out."""
    beta, n = math.exp(p[0]), p[1]
    r = sum(d.sum() for _, d in data18)
    lq = logsumexp(np.concatenate([beta * np.log(t) + n * beta * math.log(dt) for (t, _), dt in zip(data18, dT18)]))
    ll = r * math.log(beta) + r * (math.log(r) - lq) - r
    for (t, d), dt in zip(data18, dT18):
        ll += n * beta * math.log(dt) * d.sum() + (beta - 1) * np.log(t[d]).sum()
    return -ll


res = optimize.minimize(ipl_negll, x0=[math.log(7.0), 5.6], method="Nelder-Mead",
                        options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 4000})
beta_ipl, n_ipl = math.exp(res.x[0]), res.x[1]
r_tot = sum(d.sum() for _, d in data18)
lq = logsumexp(np.concatenate([beta_ipl * np.log(t) + n_ipl * beta_ipl * math.log(dt) for (t, _), dt in
                               zip(data18, dT18)]))
a_ipl = (lq - math.log(r_tot)) / beta_ipl  # exp(a) = scale coefficient: eta = exp(a) dT^-n
A_B10 = math.exp(a_ipl) * (-math.log(0.9)) ** (1 / beta_ipl)
print(f"inverse-power-law Weibull: n = {n_ipl:.4f} (paper 5.6545), shape = {beta_ipl:.3f} (paper 6.864), "
      f"A for B10 = {A_B10:.4e} (paper 7.09526e15)")
# parametric bootstrap of the likelihood-ratio statistic under a common shape, same censoring scheme
cens = [[(t, d) for t, d in zip(*g)] for g in data18]
B = 2000
lr_boot = np.empty(B)
for b in range(B):
    sim = []
    for g, eta in zip(cens, eta_common):
        t = eta * rng.weibull(b_common, size=len(g))
        d = np.ones(len(g), dtype=bool)
        for i, (c, failed) in enumerate(g):
            if not failed and t[i] > c:  # the suspended unit: censored at its stop time if still alive
                t[i], d[i] = c, False
        sim.append((t, d))
    lr_boot[b] = lr_common_shape(sim)[0]
p_lr = float(np.mean(lr_boot >= lr))
print(f"likelihood ratio, common against separate shapes: {lr:.2f}; chi2(2) p = {stats.chi2.sf(lr, 2):.3f}; "
      f"parametric bootstrap p = {p_lr:.3f} ({B} replicates)")
OUT["choi2018"] = {"shape_per_condition": {f"{d:.0f}": round(b, 3) for d, b in zip(dT18, b_sep)},
                   "shape_common": round(b_common, 3), "ipl_n": round(n_ipl, 4), "ipl_shape": round(beta_ipl, 3),
                   "ipl_A_B10": float(f"{A_B10:.5e}"), "lr_common_shape": round(float(lr), 3),
                   "lr_p_chi2": round(float(stats.chi2.sf(lr, 2)), 4), "lr_p_bootstrap": round(p_lr, 4),
                   "bootstrap_replicates": B}
# same test on the Choi 2017 and Hernes 2021 runs (complete data)
for key, rows, k in (("choi2017", choi17, "condition"), ("hernes2021", hernes, "run")):
    g = groups(rows, k, "Nf")
    data = [(v, np.ones(len(v), dtype=bool)) for _, v in sorted(g.items())]
    lr_k, b0, bs = lr_common_shape(data)
    eta0 = scales(b0, data)
    boot = np.empty(B)
    for b in range(B):
        boot[b] = lr_common_shape([(e * rng.weibull(b0, size=len(v)), np.ones(len(v), dtype=bool))
                                   for (v, _), e in zip(data, eta0)])[0]
    pk = float(np.mean(boot >= lr_k))
    print(f"{key}: shapes per condition " + ", ".join(f"{b:.1f}" for b in bs) + f"; common {b0:.2f}; "
          f"likelihood ratio {lr_k:.2f}, bootstrap p = {pk:.3f}")
    OUT[key + "_shape"] = {"per_condition": [round(b, 2) for b in bs], "common": round(b0, 3),
                           "lr": round(float(lr_k), 3), "lr_p_bootstrap": round(pk, 4)}


# ---------------------------------------------------------------------------------------------------------------
def neg_hypergeom_tail(x, n_cal, n_test, k):
    """P(at least x of n_test exchangeable test points fall below the k-th smallest of n_cal calibration points),
    when all n_cal + n_test values are exchangeable and continuous."""
    total = comb(n_cal + n_test, n_test, exact=True)
    p = 0
    for j in range(x, n_test + 1):
        # j test points before the k-th calibration point: the first j + k - 1 positions hold j tests and k - 1
        # calibration points, position j + k holds a calibration point
        p += comb(j + k - 1, j, exact=True) * comb(n_cal + n_test - j - k, n_test - j, exact=True)
    return p / total


print("\n3. Miner's rule under mixed swings (Hernes PCIM 2021, Fig. 4)")
pcim = read("G2_hernes2021pcim_sohoa.csv")
W = groups(pcim, "test", "damage")
const = np.concatenate([W[str(t)] for t in range(1, 6)])
mix6, mix7 = W["6"], W["7"]
mix = np.concatenate([mix6, mix7])
lc = np.log(const)
sd_runs = {t: float(np.std(np.log(W[str(t)]), ddof=1)) for t in range(1, 6)}
fk_w = stats.fligner(*[np.log(W[str(t)]) for t in range(1, 6)])
print(f"constant runs: n = {const.size}, mean W {const.mean():.3f}, sd ln W {lc.std(ddof=1):.3f}; sd ln W per run "
      + ", ".join(f"{t}: {s:.3f}" for t, s in sd_runs.items()) + f"; Fligner-Killeen p = {fk_w.pvalue:.3f}")
res3 = {"n_constant": int(const.size), "mean_W_constant": round(float(const.mean()), 4),
        "sd_lnW_constant": round(float(lc.std(ddof=1)), 4), "fligner_p_runs": round(float(fk_w.pvalue), 4)}
for name, m in (("test6_40K_then_70K", mix6), ("test7_70K_then_40K", mix7), ("tests6_7", mix)):
    kappa = math.exp(np.log(m).mean() - lc.mean())
    welch = stats.ttest_ind(np.log(m), lc, equal_var=False)
    mw = stats.mannwhitneyu(m, const, alternative="less")
    print(f"{name}: n = {m.size}, mean {m.mean():.3f}, Miner factor exp(mean ln W_mix - mean ln W_const) = "
          f"{kappa:.3f}; Welch t p = {welch.pvalue:.4f}; Mann-Whitney (lower) p = {mw.pvalue:.4f}")
    res3[name] = {"n": int(m.size), "mean": round(float(m.mean()), 4), "min": round(float(m.min()), 3),
                  "max": round(float(m.max()), 3), "kappa": round(kappa, 4),
                  "welch_p_two_sided": round(float(welch.pvalue), 5), "mann_whitney_p_less": round(float(mw.pvalue), 5)}
# conformal check: lower alpha-quantile of the constant-run strengths, applied to the mixed modules
res3["conformal_check"] = []
srt = np.sort(const)
for alpha in (0.10, 0.20):
    k = math.floor(alpha * (const.size + 1))
    q = float(srt[k - 1])
    for kappa_m in (1.0, 0.85):
        x = int(np.sum(mix < kappa_m * q))
        p = neg_hypergeom_tail(x, const.size, mix.size, k)
        print(f"alpha {alpha:.2f}: q = W_({k}) = {q:.3f}; Miner factor {kappa_m:.2f}: {x}/{mix.size} mixed modules "
              f"below {kappa_m * q:.3f} (nominal at most {alpha:.2f}); P(at least {x} | exchangeable) = {p:.4f}")
        res3["conformal_check"].append({"alpha": alpha, "k": k, "q": round(q, 4), "kappa_M": kappa_m,
                                        "below": x, "n_mixed": int(mix.size), "p_exchangeable": round(p, 5)})
OUT["miner_hernes_pcim"] = res3

# ---------------------------------------------------------------------------------------------------------------
print("\n4. Miner's rule under mixed swing durations (Choi, Blaabjerg, Ma, APEC 2017)")
PCT = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80]) / 100
B_SPC = {1: [4.72, 5.36, 5.58, 5.73, 5.83, 5.93, 6.02, 6.11, 6.20],  # Table III, periods of 24 h
         2: [8.42, 9.64, 10.06, 10.33, 10.54, 10.74, 10.92, 11.09, 11.28]}
OBS = {1: [7450 / 1440, 7100 / 1440, 7050 / 1440], 2: [14380 / 1440]}  # Section III.C, minutes to end of life
res4, Fs = {}, []
for spc, bl in B_SPC.items():
    bl = np.array(bl)
    slope, icpt = np.polyfit(np.log(bl), np.log(-np.log(1 - PCT)), 1)  # Weibull through the predicted B-lives
    for L in OBS[spc]:
        F_ = 1 - math.exp(-math.exp(icpt + slope * math.log(L)))
        Fs.append(F_)
        print(f"SPC {spc}: observed {L:.2f} periods; ratio to predicted B50 {L / bl[5]:.3f}; "
              f"quantile level under the prediction {F_:.3f}")
    res4[f"spc{spc}"] = {"observed_periods": [round(L, 3) for L in OBS[spc]],
                         "ratio_to_B50": [round(L / bl[5], 4) for L in OBS[spc]],
                         "quantile_level": [round(1 - math.exp(-math.exp(icpt + slope * math.log(L))), 4)
                                            for L in OBS[spc]], "weibull_shape_of_prediction": round(slope, 2)}
fisher = -2 * np.sum(np.log(Fs))
p_fisher = float(stats.chi2.sf(fisher, 2 * len(Fs)))
print(f"all three SPC-1 lives below the predicted B10: probability 0.1^3 = 0.001 if the prediction holds; "
      f"Fisher combination of the four quantile levels: {fisher:.2f}, chi2({2 * len(Fs)}) p = {p_fisher:.4f}")
res4["p_all_spc1_below_B10"] = 0.001
res4["fisher_stat"], res4["fisher_p"] = round(float(fisher), 3), round(p_fisher, 5)
OUT["miner_choi_apec"] = res4

(HERE / "thongke_G2.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
print("\nwritten thongke_G2.json")
