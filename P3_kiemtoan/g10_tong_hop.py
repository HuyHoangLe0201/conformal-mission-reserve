"""G10: evidence from the seven papers read on 16/10/2026 for the stated margin of P3.

Inputs (this folder):
  G10_junghaenel2017_bang1.csv  Table 1 of Junghaenel, Scheuermann (2017): N_f against load-pulse duration t_on for
                                wire-bond lift-off (group 1) and chip-solder fatigue (group 2), SKiM63 without
                                baseplate, T_j,max 150 C, dTj 70 K (t_on <= 2 s) and 110 K (t_on >= 2 s)
  G10_zeng2019_bang.csv         Tables II-VIII of Zeng et al. (2019): cycles to end of life of each module, in % of the
                                mean life of the constant test, for three constant and two block-mixed tests (1200 V /
                                300 A modules with Cu baseplate; all failures by bond-wire lift-off)
  G3_scheuermann2018_ton.csv    the same pulse-duration data as re-plotted in Scheuermann, Junghaenel (2018), digitized
                                in G3 (for comparison only)
  thongke_G2.json               Miner factors from Hernes PCIM 2021 and Choi APEC 2017 (G2)

Sections:
  A. heating time: slopes of ln N_f against ln t_on; margin factors from 4.5 s to the mission 90 s under each rule;
  B. Miner's rule: per-module Miner sums of the block tests of Zeng et al., raw and corrected for the realised swing
     and minimum temperature with the CIPS08 exponents; the Miner factor across three studies;
  C. low swings: the swing exponent of Lutz et al. (2020) below 30 K.

Run: python g10_tong_hop.py   (numpy, scipy). Writes g10_tong_hop.json.
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
OUT = {}
TON_T, TON_MISSION = 4.5, 90.0


def read(name):
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# =================================================================================================================
print("A. Heating time (Junghaenel, Scheuermann 2017, Table 1)")
J = read("G10_junghaenel2017_bang1.csv")
slopes = {}
for dT, lo, hi in ((110, 2, 60), (110, 10, 60), (70, 0.07, 2)):
    for g, name in (("1", "wire bond"), ("2", "solder")):
        pts = [(math.log(float(r["t_on_s"])), math.log(float(r["Nf"]))) for r in J
               if int(r["dTj_K"]) == dT and r["group"] == g and lo <= float(r["t_on_s"]) <= hi]
        x, y = np.array(pts).T
        s = stats.linregress(x, y)
        key = f"{name} {dT} K {lo}-{hi} s"
        # one-sided 95 % upper confidence bound of the magnitude of the slope (t with n - 2 degrees of freedom); added
        # 17/10/2026 to compare the stated slope 0.5 with the data
        ub = float(-s.slope + stats.t.ppf(0.95, x.size - 2) * s.stderr)
        slopes[key] = {"slope": round(float(s.slope), 3), "se": round(float(s.stderr), 3), "n": int(x.size),
                       "upper95_abs": round(ub, 3)}
        print(f"  {key:26s} slope {s.slope:+.3f} +- {s.stderr:.3f} (n = {x.size}), one-sided 95 % bound of |slope| "
              f"{ub:.3f}")
OUT["ton_slopes_table"] = slopes
# the same data as re-plotted in Scheuermann 2018 and digitized in G3
G3 = read("G3_scheuermann2018_ton.csv")
ratios = []
for r in G3:
    t, grp, dT = float(r["t_on_s"]), r["group"], int(r["dTj_K"])
    tab = [float(j["Nf"]) for j in J if j["group"] == grp and int(j["dTj_K"]) == dT
           and abs(math.log(float(j["t_on_s"]) / t)) < 0.05]
    if tab:
        ratios.append(float(r["Nf"]) / float(np.exp(np.mean(np.log(tab)))))
ratios = np.array(ratios)
print(f"  re-plot in Scheuermann 2018 against Table 1 (geometric mean per condition): ratio median {np.median(ratios):.3f}, "
      f"range {ratios.min():.3f}-{ratios.max():.3f} over {ratios.size} points")
OUT["replot_vs_table_ratio"] = {"median": round(float(np.median(ratios)), 3),
                                "range": [round(float(ratios.min()), 3), round(float(ratios.max()), 3)]}


def frac(t, C, g):
    """Fractional power law of the SKiM63 model and of the solder fit: (C + t^-g) / (C + 1), t_on in s."""
    return (C + t ** (-g)) / (C + 1)


rules = {
    "solder fractional fit (C 0.363, gamma 0.741)": frac(TON_T, 0.363, 0.741) / frac(TON_MISSION, 0.363, 0.741),
    "SKiM63 wire-bond model (C 1.434, gamma 1.208)": frac(TON_T, 1.434, 1.208) / frac(TON_MISSION, 1.434, 1.208),
    "CIPS08 exponent 0.463, t_on clipped at 15 s (Lutz 2020)": (15 / TON_T) ** 0.463,
    "CIPS08 0.463 to 15 s, then 0.29": (15 / TON_T) ** 0.463 * (TON_MISSION / 15) ** 0.29,
    "single slope 0.29": (TON_MISSION / TON_T) ** 0.29,
    "single slope 0.5 (stated)": (TON_MISSION / TON_T) ** 0.5,
    "single slope 1.0": (TON_MISSION / TON_T) ** 1.0,
}
for k, v in rules.items():
    print(f"  damage factor from {TON_T} s to {TON_MISSION:.0f} s, {k:56s} {v:5.2f} (log {math.log(v):.2f})")
OUT["ton_margin_rules"] = {k: {"factor": round(v, 3), "log": round(math.log(v), 3)} for k, v in rules.items()}

# =================================================================================================================
print("\nB. Miner's rule (Zeng et al. 2019)")
Z = read("G10_zeng2019_bang.csv")
B1, B2 = -4.416, 1285.0  # CIPS08 swing exponent and Arrhenius constant, used by the authors for the correction
ref = {}
for t, name in ((1, "N1"), (2, "N2"), (3, "N3")):
    rows = [r for r in Z if int(r["test"]) == t]
    ref[name] = (np.mean([float(r["dTj_K"]) for r in rows]), np.mean([float(r["Tjmin_C"]) for r in rows]))
const = np.array([float(r["cycles_pct"]) / 100 for r in Z if int(r["test"]) <= 3])
lc = np.log(const)
groups = [np.log([float(r["cycles_pct"]) / 100 for r in Z if int(r["test"]) == t]) for t in (1, 2, 3)]
pooled_sd = float(np.sqrt(sum(((g - g.mean()) ** 2).sum() for g in groups) / (lc.size - 3)))
print(f"  constant tests: 18 modules, pooled within-test sd of ln(life) {pooled_sd:.3f}")


def fraction(r, corrected):
    f = float(r["cycles_pct"]) / 100
    if corrected:
        dT0, Tm0 = ref[r["reference"]]
        # expected life at the module's realised conditions, relative to the mean conditions of the constant test
        f *= (float(r["dTj_K"]) / dT0) ** (-B1) * math.exp(-B2 * (1 / (float(r["Tjmin_C"]) + 273.15) - 1 / (Tm0 + 273.15)))
    return f


res = {}
for t in (4, 5):
    labels = sorted({r["label"] for r in Z if int(r["test"]) == t})
    for corrected in (False, True):
        sums = np.array([sum(fraction(r, corrected) for r in Z if int(r["test"]) == t and r["label"] == lab)
                         for lab in labels])
        key = f"test {t}, {'corrected' if corrected else 'raw'}"
        w = stats.ttest_ind(np.log(sums), lc, equal_var=False)
        res[key] = {"miner_sums": [round(float(s), 3) for s in sums], "mean": round(float(sums.mean()), 3),
                    "kappa": round(float(math.exp(np.log(sums).mean())), 3), "welch_p": round(float(w.pvalue), 4)}
        print(f"  {key:22s} Miner sums " + ", ".join(f"{s:.2f}" for s in sums) + f"; mean {sums.mean():.3f}; "
              f"kappa {math.exp(np.log(sums).mean()):.3f}; Welch p {w.pvalue:.3f}")
res["test 6 (alternating 20 % steps)"] = {"mean": 1.006, "note": "mean only, no per-module table"}
print("  test 6, alternating test 1 and test 2 in 20 % steps: mean 1.006 (text only)")
OUT["zeng2019"] = {"pooled_sd_ln": round(pooled_sd, 4), "results": res}
G2 = json.loads((HERE / "thongke_G2.json").read_text(encoding="utf-8"))
kappa = {
    "Hernes PCIM 2021, 40 then 70 K (IHV 3.3 kV)": G2["miner_hernes_pcim"]["test6_40K_then_70K"]["kappa"],
    "Hernes PCIM 2021, 70 then 40 K": G2["miner_hernes_pcim"]["test7_70K_then_40K"]["kappa"],
    "Choi APEC 2017, SPC 1 (IPM, mixed swing durations)": round(float(np.exp(np.mean(np.log(
        G2["miner_choi_apec"]["spc1"]["ratio_to_B50"])))), 3),
    "Choi APEC 2017, SPC 2": G2["miner_choi_apec"]["spc2"]["ratio_to_B50"][0],
    "Zeng 2019, test 4 (100 K/50 C then 50 K/100 C), corrected": res["test 4, corrected"]["kappa"],
    "Zeng 2019, test 5 (100 K, 15 s then 50 K/100 C), corrected": res["test 5, corrected"]["kappa"],
    "Zeng 2019, test 6 (alternating)": 1.006,
}
print("  Miner factor across the three studies:")
for k, v in kappa.items():
    print(f"    {k:60s} {v:.3f}")
OUT["kappa_across_studies"] = kappa

# =================================================================================================================
print("\nC. Low swings (Lutz et al. 2020, DAB high-power modules, t_on 10 ms)")


def beta1p(dT, b1=-4.1, th1=27.1, th2=2.08):
    """Swing exponent of Lutz et al. (2020), Eq. (3): exp(-(dT - theta1)/theta2) + beta1."""
    return math.exp(-(dT - th1) / th2) + b1


lut = {str(d): round(beta1p(d), 2) for d in (43, 35, 30, 28, 27.1, 26, 25)}
print("  exponent beta1' at dT (K): " + ", ".join(f"{k}: {v}" for k, v in lut.items()))
print("  three modules at corrected swings 26.7-29.6 K passed 1.17e9 cycles without failure; expected end of life "
      "1.6-2.3e9 cycles; CIPS08 underestimates such lives by more than a factor 10 below 30 K (paper)")
OUT["lutz2020"] = {"beta1_prime": lut, "unfailed_cycles": 1.169e9, "expected_eol_gigacycles": [1.61, 2.26],
                   "corrected_swings_K": [26.7, 29.6]}

(HERE / "g10_tong_hop.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
print("\nwritten g10_tong_hop.json")
