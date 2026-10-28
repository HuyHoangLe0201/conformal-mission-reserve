"""P3 case study (G6): the full case-study runs, resumable and parallel over repetitions.

Every repetition draws its own world, bench, recorded intervals and field leg from a seed fixed by (experiment seed,
repetition), so results do not depend on scheduling, and every finished repetition is appended at once as one line of
g6_<experiment>_<tag>.jsonl. Running the same command again skips the repetitions already written.

Devices (3.3 kV converter, 18/10/2026; the 1.7 kV runs are kept in luu_1700V_20261018/). The leg is built of
single-switch modules, and the certified devices are its outer module (IGBT T1 with diode D1) and its inner module (T2
with D2; thu_nghiem_g5.MODULES; T4 and T3 have the same cycles by symmetry). They are separate devices: the field leg
draws a log-strength, a Miner factor and a thermal scale per module, and each module carries its own certificate from
the same bench scores and recorded intervals; the chips of one module share its strength, so its certificate takes its
largest chip. Static experiments (A, D, E, N, X, T, L, Y, R) return one record per module under 'modules', with the
structure that the single-device records had, and the summaries pool the modules. In the closed loop (C) the life
belongs to the leg: a derating level passes when the certificates of both modules pass, the reserve is the smaller one,
the modules of the leg are replaced together when no level passes or at the end of a static age, and the leg fails when
either module fails; coverage is recorded per decision and module.

Experiments (stated margins of G5: short pulses out of scope, gamma_t = 0.5 uncapped, statistical part 'delta'):
  A  coverage and tightness. Per repetition and bench size n1: certificates U1, U2, U3, U5, U6, U9 at several levels
     alpha, their coverage of the true end-of-interval damage and of the calibrated part, the oracle U10 (quantiles of
     the true damage of fresh field modules in the same world), and U6 with the stated lifetime margins replaced by the
     world's true values ('anchored': gamma, gamma_t above 4.5 s and 60 s, E_a measured exactly, as ideal anchor tests
     would; Miner factor, thermal bound and statistical part kept).
  C  closed loop over leg lives: one configuration per truth and driving strategy runs every policy and method on
     common draws, 300 lives (since 18/10/2026; it replaces the separate C, F4 and V runs, which differed in their
     numbers of lives). Certificate policies: 'reserve' (smallest passing level whose reserve reaches the planned
     replacement after t_plan intervals, else the passing level with the largest reserve), 'pass' (smallest passing
     level), 'fixed20' (20 % whenever it passes), 'reserve_cap' (only levels with at most late_max late sections),
     'reserve_budget' (running mean of late sections at most late_max), 'replace' (no derating: the certificate only
     triggers replacement). Methods: 'union', 'paired', 'split0' (the recorded intervals of the base policy at every
     level, no re-simulation), 'reuse' (policy and certificate on the same data), 'aci' (adaptive level, driven by the
     miss indicator of the truth), 'mc' and 'bayes' (strength law fitted to all bench scores). Baselines without a
     certificate in the loop: 'tbr60' (replacement at the planned life), 'rtf' (run to failure, censored at T_MAX),
     'fix20_tbr60' (20 % derating throughout and replacement at the planned life), 'cert_age' (replacement at the
     reserve of the paired certificate at commissioning, on all bench scores, without derating), 'mc_age' (at the
     B-life of the leg under the Monte Carlo model at commissioning, the alpha-quantile of its life), 'mc_opt10' and
     'mc_opt50' (at the age that minimizes the cost per interval of age replacement under the Monte Carlo life law,
     Barlow and Hunter 1960, for a failure costing 10 or 50 planned replacements). Per life and run: served intervals,
     failure, true damage at replacement, coverage per decision and module (bits), derated and false-derating shares
     (a derating the leg did not need: at full load no module would reach its budget before the planned replacement),
     mean level, late sections (mean and sum, for a cost of lateness), ACI level, static age. The drift
     configurations add a true thermal scale that rises with the true damage of each module (rho) and, optionally, a
     fixed allowance in the upper scale of the certificate (s_up).
  D  margin sensitivity: (1) deterministic, on the real Hernes data (half split, 200 splits): margined damage of a
     30-day interval and certified life of a new leg for every margin variant, including short pulses brought into the
     scope; (2) coverage lost when the truth sits at the base margin while the certificate uses a narrower variant
     (boundary truth, n1 = 40).
  E  season shift: the field interval runs at ambient + delta (summer), the recorded intervals are the winter data;
     paired certificate with the recorded intervals as they are, against re-simulated at the field ambient.
  N  anchor scenario: n_a anchor tests at each heating time of an anchor set (300 s, or 30 s and 300 s; swing 80 K,
     T_min 60 C) give upper confidence bounds (joint 0.95) on the heating-time effect, which replace the stated
     heating-time margin; below the last anchor only monotonicity of life in the heating time is assumed, above it the
     stated slope continues. Paired certificate with it against the stated one in the same world.
  X  exceedance sweep: boundary truth with both heating-time slopes at g = 0.5-1.0 (stated 0.5); coverage of the
     stated certificate and of certificates with anchor tests, against g.
  T  temporal dependence: recorded and coming 30-day intervals built four ways: 'resampled', 'blocks' (one vehicle
     day on each of 30 consecutive calendar dates), 'forward' (recorded blocks end before the coming block, which covers
     the last 30 dates) and 'backward' (recorded blocks start after the coming block, the first 30 dates).
  L  lot shift: field modules from a lot whose log-strength law is moved (-0.1, -0.2) or widened (x1.5, x2 about the
     median) against the bench lot; paired, union, Monte Carlo and Bayes certificates of the bench lot, and the paired
     certificate recalibrated on 10, 20 or 40 modules of the field lot.
  Y  several stated components exceeded at once from the truth at the margins, singly and cumulatively.
  R  re-simulation necessity (added 18/10/2026): the coming interval runs under a changed policy while the recorded
     intervals ran under the base policy (cruising, base pattern): a balancing pattern ('in' relieves the inner IGBTs,
     'out' the outer ones) or a switch to the energy-saving 'coast' driving strategy; paired certificate on the recorded
     intervals as they are, against re-simulated under the changed policy.

Run: python thu_nghiem_g6.py <tag> [workers]  (configurations in CONFIGS; chay_tatca.sh runs them all)
     python thu_nghiem_g6.py D_det  (deterministic part of D, single process)
     python thu_nghiem_g6.py wait_memory <GB>  (blocks until that much virtual memory is free; the machine is shared)
"""
import json
import math
import os
import platform
import site
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
if hasattr(platform, "_wmi_query"):
    # Python 3.12: importing scipy.stats queries WMI through platform, and the WMI client (fastprox.dll) crashed a run
    # under load on 16/10/2026 (0xc0000409); without WMI, platform falls back to sys.getwindowsversion()
    def _no_wmi(*_a):
        raise OSError("WMI queries disabled")
    platform._wmi_query = _no_wmi

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import chung_chi as cc  # noqa: E402
import thu_nghiem_g5 as g5  # noqa: E402
import tuoi_tho as tt  # noqa: E402

BASE = tt.Margin(stat="delta")
CERTS = ("U1 box", "U2 Monte Carlo", "U3 Bayes", "U5 union", "U6 paired", "U9 PAC")
T_MAX = 240                      # closed loop: horizon of 240 intervals (20 years); run to failure is censored there
CERT_RUNS = (tuple(("reserve", m) for m in ("union", "paired", "split0", "reuse", "aci", "mc", "bayes"))
             + tuple((p, m) for p in ("pass", "fixed20", "reserve_cap", "reserve_budget", "replace")
                     for m in ("union", "paired")))
BASELINES = ("tbr60", "rtf", "fix20_tbr60", "cert_age", "mc_age", "mc_opt10", "mc_opt50")
RUNS_ALL = CERT_RUNS + tuple(("none", b) for b in BASELINES)
RUNS_DRIFT = tuple(("reserve", m) for m in ("union", "paired", "mc", "bayes")) + tuple(("none", b) for b in BASELINES)
SEEDS_C = {"evidence": 611, "boundary": 612, "adverse_strict": 613, "beyond": 614}
RUNS_A05 = tuple((p, m) for p in ("reserve", "replace") for m in ("union", "paired")) + \
    tuple(("none", b) for b in BASELINES)
RUNS_N40 = tuple((p, m) for p in ("reserve", "replace") for m in ("union", "paired", "pac")) + \
    tuple(("none", b) for b in BASELINES)
RUNS_WEAK = tuple(("replace", m) for m in ("paired", "union", "mc", "bayes")) + tuple(("none", b) for b in BASELINES)
RUNS_ANCHOR = tuple(("replace", m) for m in ("paired", "union")) + \
    tuple(("none", b) for b in ("tbr60", "rtf", "cert_age", "mc_age", "mc_opt10"))

CONFIGS = {
    # A: coverage and tightness
    "A_evidence_k0": dict(exp="A", scen="evidence", k_past=0, n1=(10, 20, 40, 80), n2=100, alphas=(0.1, 0.05, 0.01),
                          n_oracle=400, reps=200, anchored=True, seed=601),
    "A_boundary_k0": dict(exp="A", scen="boundary", k_past=0, n1=(10, 20, 40, 80), n2=100, alphas=(0.1, 0.05, 0.01),
                          n_oracle=400, reps=200, anchored=False, seed=602),
    "A_adverse_strict_k0": dict(exp="A", scen="adverse_strict", k_past=0, n1=(10, 20, 40, 80), n2=100,
                                alphas=(0.1, 0.05, 0.01), n_oracle=400, reps=200, anchored=False, seed=603),
    "A_beyond_k0": dict(exp="A", scen="beyond", k_past=0, n1=(10, 20, 40, 80), n2=100, alphas=(0.1, 0.05, 0.01),
                        n_oracle=400, reps=200, anchored=False, seed=607),
    "A_evidence_k20": dict(exp="A", scen="evidence", k_past=20, n1=(20, 40), n2=100, alphas=(0.1, 0.05),
                           n_oracle=100, reps=60, anchored=False, seed=604),
    # n2 = 500 (the 1.7 kV file g6_A_evidence_k0_n500.jsonl stays open in a stalled process, hence the new tag)
    "A_evidence_n500": dict(exp="A", scen="evidence", k_past=0, n1=(20, 40), n2=500, alphas=(0.1, 0.05, 0.01),
                            n_oracle=400, reps=60, anchored=False, seed=605),
    # C: closed loop over leg lives. All runs of one truth share its seed, for both strategies (the cycle pools of the
    # two strategies hold the same 4215 days), so every comparison between policies, methods and strategies is paired.
    **{f"C_{sc}_{st}": dict(exp="C", scen=sc, strategy=st, runs=RUNS_ALL, t_plan=60, n1=20, n2=50, alpha=0.1,
                            reps=300, seed=SEEDS_C[sc], late_max=0.05)
       for sc in SEEDS_C for st in ("cruise", "coast")},
    # drift of the true thermal scale (rho per unit of true damage of the module) against a fixed allowance s_up in the
    # upper scale of the certificate: 1.32 = 1.10 (1 + 0.2) covers rho = 0.2 up to the damage budget, not rho = 0.4
    **{f"C_{name}": dict(exp="C", scen="boundary", strategy="cruise", runs=RUNS_DRIFT, t_plan=60, n1=20, n2=50,
                         alpha=0.1, reps=300, seed=612, late_max=0.05, rho=rho, s_up=su)
       for name, rho, su in (("drift", 0.2, 1.10), ("drift_allow", 0.2, 1.32), ("drift04", 0.4, 1.10),
                             ("drift04_allow", 0.4, 1.32))},
    # D: coverage lost with a narrower margin than the truth (boundary truth at the base margin)
    "D_cov": dict(exp="D", n1=40, n2=100, alpha=0.1, n_oracle=0, reps=100, seed=621),
    # E: season shift
    "E_shift": dict(exp="E", scen="boundary", n1=40, n2=100, alpha=0.1, deltas=(0.0, 10.0, 20.0), reps=100, seed=631),
    "E_shift_evidence": dict(exp="E", scen="evidence", n1=40, n2=100, alpha=0.1, deltas=(0.0, 10.0, 20.0), reps=100,
                             seed=632),
    # N: anchor tests at the field heating time in place of the stated heating-time margin
    "N_evidence": dict(exp="N", scen="evidence", n1=40, n2=100, alphas=(0.1, 0.05), n_a=(3, 5, 10),
                       sets=((300.0,), (30.0, 300.0)), dT_a=80.0, reps=100, seed=644),
    "N_boundary": dict(exp="N", scen="boundary", n1=40, n2=100, alphas=(0.1, 0.05), n_a=(3, 5, 10),
                       sets=((300.0,), (30.0, 300.0)), dT_a=80.0, reps=100, seed=645),
    "N_adverse_strict": dict(exp="N", scen="adverse_strict", n1=40, n2=100, alphas=(0.1, 0.05), n_a=(3, 5, 10),
                             sets=((300.0,), (30.0, 300.0)), dT_a=80.0, reps=100, seed=646),
    # X: exceedance of the stated heating-time margin (boundary truth with both heating-time slopes at g >= 0.5)
    "X_ton": dict(exp="X", n1=40, n2=100, alphas=(0.1,), g=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), n_a=(5,),
                  sets=((300.0,), (30.0, 300.0)), dT_a=80.0, reps=100, seed=651),
    # T: temporal dependence of the missions
    "T_boundary": dict(exp="T", scen="boundary", n1=40, n2=100, alphas=(0.1,), reps=200, seed=661),
    "T_evidence": dict(exp="T", scen="evidence", n1=40, n2=100, alphas=(0.1,), reps=200, seed=662),
    # L: bench-to-field lot shift
    "L_boundary": dict(exp="L", scen="boundary", n1=40, n2=100, alphas=(0.1, 0.05), n_lot=(10, 20, 40), n_field=200,
                       reps=200, seed=671),
    "L_evidence": dict(exp="L", scen="evidence", n1=40, n2=100, alphas=(0.1, 0.05), n_lot=(10, 20, 40), n_field=200,
                       reps=200, seed=672),
    # Y: several stated components exceeded at once, from the truth at the margins
    "Y_stack": dict(exp="Y", n1=40, n2=100, alphas=(0.1, 0.05), n_field=50, reps=200, seed=681),
    # R: re-simulation necessity under a balancing pattern
    "R_boundary": dict(exp="R", scen="boundary", n1=40, n2=100, alphas=(0.1, 0.05), patterns=("in", "out", "coast"),
                       reps=200, seed=691),
    "R_evidence": dict(exp="R", scen="evidence", n1=40, n2=100, alphas=(0.1, 0.05), patterns=("in", "out", "coast"),
                       reps=200, seed=692),
    # Third review (19/10/2026). T_evidence repeated with another seed: its calibrated part, 0.863 at n1 = 40, lay 2.2
    # cluster standard errors below the exact level 37/41, against 0.915 in A_evidence_k0 at the same n1.
    "T_evidence_b": dict(exp="T", scen="evidence", n1=40, n2=100, alphas=(0.1,), reps=200, seed=663),
    # Q: truths outside the family of the stated margin (static coverage as A): a swing-heating-time interaction that no
    # slope of the margin follows (adverse), and lives rising faster than a power law at small swings (favorable)
    **{f"Q_{sc.replace('+', '_')}": dict(exp="A", scen=sc, k_past=0, n1=(20, 40), n2=100, alphas=(0.1, 0.05),
                                         n_oracle=400, reps=200, anchored=False, seed=seed)
       for sc, seed in (("boundary+int", 701), ("evidence+int", 702), ("boundary+lutz", 703),
                        ("boundary+nc", 704), ("evidence+nc", 705), ("boundary+th", 706),
                        ("evidence+th", 707), ("boundary+seq", 708), ("evidence+seq", 709))},
    # U: paired certificate against union certificates whose levels are allocated in advance with a small mission level
    **{f"U_{sc}": dict(exp="U", scen=sc, n1=(10, 20, 40, 80), n2=100, alphas=(0.1, 0.05), a_V=(0.005, 0.01, 0.02),
                       reps=200, seed=seed) for sc, seed in (("boundary", 711), ("evidence", 712))},
    # M: mission change within a life (driving strategy switched at an interval drawn in t_sw), without derating:
    # certificates re-simulate the recorded days under the new strategy ('stale' keeps the old one); static ages are
    # set at commissioning for the first mission, and 'replanned' ages are recomputed at the change
    **{f"M_{sc}_{a}_{b}": dict(exp="M", scen=sc, first=a, second=b, t_sw=(10, 30), n1=20, n2=50, alpha=0.1, t_plan=60,
                               reps=300, seed=seed)
       for (sc, a, b), seed in zip((("boundary", "cruise", "coast"), ("boundary", "coast", "cruise"),
                                    ("evidence", "cruise", "coast"), ("evidence", "coast", "cruise")),
                                   (721, 722, 723, 724))},
    # Editor re-review (19/10/2026): the closed loop at n1 = 40 with the PAC certificate (eta = alpha; 30 scores per
    # side make it finite), and fleets of 20 legs sharing one bench, to see how the failures vary between benches
    **{f"C_boundary_{st}_n40": dict(exp="C", scen="boundary", strategy=st, runs=RUNS_N40, t_plan=60, n1=40, n2=50,
                                    alpha=0.1, eta=0.1, reps=300, seed=seed, late_max=0.05)
       for st, seed in (("cruise", 617), ("coast", 618))},
    **{f"F_{sc}_{st}": dict(exp="F", scen=sc, strategy=st, n1=40, n2=50, alpha=0.1, eta=0.1, n_legs=20, t_plan=60,
                            reps=100, seed=seed)
       for (sc, st), seed in zip((("boundary", "cruise"), ("boundary", "coast"), ("beyond", "cruise")), (731, 732, 733))},
    # C at alpha = 0.05 (a stricter level; n1 = 40 so that both certificates are finite)
    **{f"C_boundary_{st}_a05": dict(exp="C", scen="boundary", strategy=st, runs=RUNS_A05, t_plan=60, n1=40, n2=50,
                                    alpha=0.05, reps=300, seed=seed, late_max=0.05)
       for st, seed in (("cruise", 615), ("coast", 616))},
    # S: a second failure mechanism in the truth, fatigue of the baseplate solder under the case-temperature cycles
    # (han_de.py, with its dwell rule), which the certificates do not count. When cruising, the same configurations and
    # seeds as A_*_k0, whose junction results they reproduce, since the solder draws use a separate random stream; when
    # coasting, the same configurations with new seeds (the A runs cruise only). Solder factor per world: swing under
    # the hottest chip 1-1.3 times the lumped case node (to the power 4.4), times the life level of the lot, 1-2 times
    # shorter than the curve; at most 1.3^4.4 x 2 = 6.35.
    **{f"S_{sc}_{st}": dict(exp="A", scen=sc, strategy=st, k_past=0, n1=(10, 20, 40, 80), n2=100,
                            alphas=(0.1, 0.05, 0.01), n_oracle=400, reps=200, anchored=anch, seed=seed,
                            solder=dict(sw=(1.0, 1.3), life=(1.0, 2.0)))
       for sc, st, anch, seed in (("evidence", "cruise", True, 601), ("boundary", "cruise", False, 602),
                                  ("evidence", "coast", False, 743), ("boundary", "coast", False, 744))},
    # W: coverage of the statistical part alone, first-order ('delta'), Monte Carlo ('mc') and Scheffe ellipsoid bounds
    "W_boundary": dict(exp="W", scen="boundary", n1=(20, 40), n_int=10, stats=("delta", "mc", "ellipsoid"), reps=200,
                       seed=751),
    # K: multi-split paired certificate, B = 15 splits of the same bench (the first is that of A), at alpha and at
    # alpha/2; the same configurations and seeds as A_*_k0, whose results they reproduce (separate stream)
    **{f"K_{sc}": dict(exp="A", scen=sc, k_past=0, n1=(10, 20, 40, 80), n2=100, alphas=(0.1, 0.05, 0.01),
                       n_oracle=400, reps=200, anchored=anch, seed=seed,
                       derand=dict(B=15, n1=(20, 40), alphas=(0.1, 0.05)))
       for sc, anch, seed in (("evidence", True, 601), ("boundary", False, 602))},
    # Fourth round (20/10/2026), experiments that test the remaining weaknesses directly.
    # P: a weak subpopulation (2-10 % of the devices, lives 1.6-2.7 times shorter), a strength law outside the Weibull
    # family of the parametric certificates: static coverage of the calibrated part and of the true damage
    **{f"P_{sc}_weak": dict(exp="A", scen=f"{sc}+weak", k_past=0, n1=(20, 40, 80), n2=100, alphas=(0.1, 0.05),
                            n_oracle=400, reps=200, anchored=anch, seed=seed)
       for sc, anch, seed in (("boundary", False, 761), ("evidence", True, 762))},
    # small miscoverage levels with large benches (the paired certificate needs 99 scores at alpha = 0.01)
    **{f"P_{name}_a001": dict(exp="A", scen=sc, k_past=0, n1=(100, 200), n2=200, alphas=(0.01,), n_oracle=2000,
                              reps=200, anchored=False, seed=seed)
       for name, sc, seed in (("boundary", "boundary", 763), ("boundary_weak", "boundary+weak", 764))},
    # closed loop, replacement only, under the weak subpopulation: conformal against parametric certificates
    **{f"C_boundary_{st}_weak": dict(exp="C", scen="boundary+weak", strategy=st, runs=RUNS_WEAK, t_plan=60, n1=20,
                                     n2=50, alpha=0.1, reps=300, seed=seed, late_max=0.05)
       for st, seed in (("cruise", 766), ("coast", 767))},
    # mission change with the weak subpopulation, and with the replanned ages computed only after a detection delay
    # (the base configuration again, same seed, so its other runs are reproduced)
    "M_boundary_cruise_coast_weak": dict(exp="M", scen="boundary+weak", first="cruise", second="coast", t_sw=(10, 30),
                                         n1=20, n2=50, alpha=0.1, t_plan=60, reps=300, seed=765, delays=(2, 6, 12)),
    "M_boundary_cruise_coast_dl": dict(exp="M", scen="boundary", first="cruise", second="coast", t_sw=(10, 30), n1=20,
                                       n2=50, alpha=0.1, t_plan=60, reps=300, seed=721, delays=(2, 6, 12)),
    # value of anchor tests in the closed loop: the stated lifetime margins replaced by ideal anchor tests (the world's
    # true values) or by five realistic anchor tests at 30 and 300 s (heating-time rule only), replacement only; the
    # seeds of C, so every life is paired with the stated-margin run of the same truth and strategy
    **{f"V_{sc}_{st}_{an}": dict(exp="C", scen=sc, strategy=st, runs=RUNS_ANCHOR, t_plan=60, n1=20, n2=50, alpha=0.1,
                                 reps=300, seed=SEEDS_C[sc], late_max=0.05, anchor=an)
       for sc in ("evidence", "boundary") for st in ("cruise", "coast") for an in ("ideal", "real")},
    # electro-thermal truth (dien_nhiet.py): static coverage when the true junction response has losses at the
    # instantaneous junction temperature and a temperature-dependent chip impedance, the certificate keeping the
    # linear model with 150 C losses and its stated bound 1.10
    **{f"E_{sc}_et_{st}": dict(exp="A", scen=sc, strategy=st, k_past=0, n1=(20, 40), n2=100, alphas=(0.1, 0.05),
                               n_oracle=400, reps=200, anchored=False, seed=seed, truth_db="et")
       for (sc, st), seed in zip((("boundary", "cruise"), ("boundary", "coast"), ("evidence", "cruise"),
                                  ("evidence", "coast")), (771, 772, 773, 774))},
}
ANCHOR_SET, ANCHOR_N, ANCHOR_DT = (30.0, 300.0), 5, 80.0   # realistic anchor tests of the V configurations
SOL_SD = 0.5          # S: log-life scatter of the solder between modules (lognormal), wider than the 0.14 of the strengths
SOL_CU = 10.0         # S: a copper baseplate has ten times fewer cycles than AlSiC (AN2019-05)
SOL_K = 4.4           # S: exponent of the solder law on the case swing (han_de.K_EXP)
SOL_NB = 40           # S: thermal-cycling tests of the field lot for the certificate of the solder
SOL_STATED = (1.3 * 1.10) ** SOL_K   # S: stated part of that certificate: swing under the hottest chip, thermal bound
SOL_ALPHAS = (0.1, 0.05)             # S: levels of the solder certificate (alone, and alpha/2 beside the junction)

# margin variants of D (name -> Margin); all keep the base scope (short pulses excluded)
VARIANTS = {
    "base": BASE,
    "gamma 0": replace(BASE, gamma=0.0),
    "gamma 0.6": replace(BASE, gamma=0.6),
    "gamma 2.08": replace(BASE, gamma=2.08),
    "dT_T 68 K": replace(BASE, dT_T=68.0),
    "gamma_t 0.29": replace(BASE, gamma_t=0.29),
    "gamma_t 1.0": replace(BASE, gamma_t=1.0),
    "t_on cap 60 s": replace(BASE, ton_rule="cap"),
    "t_on slope 0.27 above 60 s": replace(BASE, ton_rule="slope"),
    "E_a 0.111 eV": replace(BASE, Ea=0.111),
    "kappa_M 1": replace(BASE, kappa_M=1.0),
    "stat ellipsoid": replace(BASE, stat="ellipsoid"),
    "stat none": replace(BASE, stat="none"),
}
# The short-pulse variants need the cycles outside the base scope, which the pools drop; their effect on the whole
# mission is in g5_ngan_sach.json.
NARROW = ("gamma 0.6", "gamma_t 0.29", "t_on cap 60 s", "kappa_M 1", "stat none")
# Short pulses (t_on < 0.15 s) inside the scope, for the deterministic part of D only (pool with those cycles kept)
SHORT = {"short pulses counted at 0.15 s": replace(BASE, short_rule="clip"),
         "short pulses credited, 0.59": replace(BASE, short_rule="credit")}

_W = {}


def scenario(name):
    """Truth by name; a suffix '+int' or '+lutz' adds a structural departure (tt.Scenario.structural, 19/10/2026)."""
    base, _, kind = name.partition("+")
    sc = {"evidence": tt.Scenario(), "adverse": tt.Scenario.adverse(), "boundary": tt.Scenario.boundary(),
          "adverse_strict": tt.Scenario.adverse_strict(), "beyond": tt.Scenario.beyond()}[base]
    return sc.structural(kind) if kind else sc


def _init(exp, strategy="cruise", s_up=g5.S_UP, truth_db=None):
    # A failure here (memory, missing database) is kept and raised by every repetition: a raising initializer would
    # make multiprocessing restart the worker without end.
    try:
        if truth_db:
            # fourth round (20/10/2026): the truth of a static run evaluated on another cycle database of the same
            # days (dien_nhiet.py: losses at the instantaneous junction temperature, temperature-dependent chip
            # impedance), while the certificate keeps the case-study database
            _W["tpool"] = g5.DayPool(f"chu_ky_{strategy}_d00_base_{truth_db}.npz", BASE)
        h = tt.hernes()
        _W["b_full"] = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])["b"]
        if exp == "C":
            # s_up: upper thermal scale of the certificate (stated 1.10; a larger value is a fixed drift allowance)
            _W["pools"] = {d: g5.DayPool(f"chu_ky_{strategy}_d{int(round(100 * d)):02d}_base.npz", BASE, s_up=s_up)
                           for d in g5.LEVELS}
            late = json.loads((HERE / "giam_tai_kha_thi.json").read_text(encoding="utf-8"))
            _W["late"] = {d: 1 - late[f"{d:.1f}"]["on_time_share"] for d in g5.LEVELS}
        elif exp == "R":
            _W["pools"] = {p: g5.DayPool(f"chu_ky_cruise_d00_{p}.npz", BASE) for p in ("base", "in", "out")}
            _W["pools"]["coast"] = g5.DayPool("chu_ky_coast_d00_base.npz", BASE)
        elif exp == "F":
            _W["pool"] = g5.DayPool(f"chu_ky_{strategy}_d00_base.npz", BASE)
        elif exp == "M":
            # both driving strategies on the same 4215 days, without derating (mission change within a life)
            _W["pools"] = {st: g5.DayPool(f"chu_ky_{st}_d00_base.npz", BASE) for st in ("cruise", "coast")}
        else:
            # the static experiments use the cruising mission; S also runs on the coasting one (19/10/2026)
            _W["pool"] = g5.DayPool(f"chu_ky_{strategy}_d00_base.npz", BASE)
    except Exception as e:  # noqa: BLE001
        _W.clear()
        _W["error"] = f"worker initialization failed: {e!r}"


def _check():
    if "error" in _W:
        raise RuntimeError(_W["error"])


def free_virtual_gb():
    """Available commit charge (free virtual memory) in GB on Windows; None elsewhere."""
    if os.name != "nt":
        return None
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                   [(k, ctypes.c_ulonglong) for k in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.apf / 1e9


def wait_memory(need_gb, label="", poll_s=60):
    """Block until the free virtual memory reaches need_gb; returns it (or None where it cannot be read)."""
    free, t0, said = free_virtual_gb(), time.time(), -1
    while free is not None and free < need_gb:
        k = int((time.time() - t0) // 1800)
        if k > said:
            print(f"{label}: waiting for {need_gb:.1f} GB of free virtual memory (now {free:.1f} GB), "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)
            said = k
        time.sleep(poll_s)
        free = free_virtual_gb()
    return free


def model_raw(pool, idx, fitd, mg, s=g5.S_UP, shift=0.0):
    """Margined model damage per chip from the raw cycles (for margins or ambient shifts other than the pool's)."""
    out = []
    for rw in pool.raw:
        cyc = {k: np.concatenate([rw[i][k] for i in idx]) for k in ("dTheta", "theta_min", "ton", "n", "T_amb")}
        V, la = tt.prep(cyc, mg, s, shift)
        out.append(tt.damage_from_prep(*tt.compress(V, la), fitd, mg))
    return np.array(out)


def anchored_margin(world, fitd):
    """Stated lifetime margins replaced by the world's true values, as ideal anchor tests would measure them."""
    return replace(BASE, gamma=max(0.0, abs(fitd["b"][1]) - world.beta_low), gamma_t=world.gamma_t, ton_rule="slope",
                   gamma_h=world.gamma_h, Ea=world.Ea)


def field_leg(world, sc, rng, K):
    """Log-strength, Miner factor and thermal scale of each of the K modules of a field leg (separate devices)."""
    return world.strengths(rng, K), rng.uniform(*sc.kappa, size=K), rng.uniform(*sc.s_th, size=K)


def world_params(world, b_full):
    """Components of the truth that are drawn per world, for the breakdown by exceeded components."""
    out = dict(beta_shift=float(world.beta_low - abs(b_full[1])), gamma_t=float(world.gamma_t),
               gamma_h=float(world.gamma_h), Ea=float(world.Ea))
    if world.c_int or world.c_lutz or world.nc_sd or world.c_th or world.c_seq:
        out.update(c_int=float(world.c_int), c_lutz=float(world.c_lutz), nc_sd=float(world.nc_sd),
                   c_th=float(world.c_th), c_seq=float(world.c_seq))
    if world.weak_p:
        out.update(weak_p=float(world.weak_p), weak_d=float(world.weak_d))
    return out


def bench_scores(world, rng, n_total):
    """Synthetic bench on the Hernes design, half split within each run: fit, scores and design matrix."""
    bn = g5.bench(world, rng, n_total)
    fi, si = tt.split(bn["run"], "half", rng)
    fitd = tt.fit(bn["dT"][fi], bn["ton"][fi], bn["I"][fi], bn["Nf"][fi], bn["run"][fi])
    X = tt.features(bn["dT"], bn["ton"], bn["I"])
    return bn, fi, si, fitd, X, np.log(bn["Nf"][si]) - X[si] @ fitd["b"]


def flist(x):
    return [float(v) for v in np.atleast_1d(x)]


# ------------------------------------------------------------------------------------------------------------- A
def rep_A(args):
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    sc = scenario(cfg["scen"])
    pool, b_full = _W["pool"], _W["b_full"]
    K, groups = pool.K, pool.groups
    world = tt.World.draw(rng, b_full, sc)
    k = cfg["k_past"]
    nc = len(pool.chips)
    sol = cfg.get("solder")
    if sol:
        # S (19/10/2026): baseplate-solder channel, drawn from a separate stream so that the junction results are those
        # of A: a factor per world on the nominal AlSiC law with the dwell rule (1 to 6.35: case swing under the hottest
        # chip up to 1.3 times the lumped node and lives halved, 1.3^4.4 x 2), the module's thermal scale on the case
        # swing and a lognormal solder strength per module; the copper variant multiplies the solder damage by SOL_CU
        # (second version, 19/10/2026: the factor is drawn as the product of the swing factor under the hottest chip,
        # stated, and the life level of the lot against the curve, which a solder bench measures)
        if "solder" not in _W:
            import han_de
            _W["solder"] = han_de.daily(cfg.get("strategy", "cruise"))
        rng_s = np.random.default_rng([cfg["seed"], r, 7])
        f_sw, f_life = float(rng_s.uniform(*sol["sw"])), float(rng_s.uniform(*sol["life"]))
        c_sol = f_sw ** SOL_K * f_life
    sol_out = []

    def field():
        S, kap, s_th = field_leg(world, sc, rng, K)
        past = [pool.sample(rng) for _ in range(k)]
        new = pool.sample(rng)
        # strength not common (19/10/2026): the true damage uses the log-strength at field conditions, S + delta, while
        # the score keeps the reference strength S; delta is drawn only in that truth, so other streams are unchanged
        St = S + world.nc_sd * rng.standard_normal(K) if world.nc_sd else S
        tp = _W.get("tpool", pool)
        D = sum((tp.true(p, world, St, kap, s_th) for p in past), np.zeros(nc)) + tp.true(new, world, St, kap, s_th)
        if sol:
            days = np.concatenate(past + [new])
            sol_out.append(c_sol * np.asarray(s_th) ** SOL_K * np.exp(-SOL_SD * rng_s.standard_normal(K))
                           * _W["solder"][:, days].sum(1))
        return S, kap, s_th, past, new, pool.mod(D)

    orc = np.array([field()[5] for _ in range(cfg["n_oracle"])])                    # [n_oracle, K]
    if sol:
        orc_s = np.array(sol_out)
        sol_out.clear()
    rec = [pool.sample(rng) for _ in range(cfg["n2"])]
    S_new, kap, s_th, past, new, D_star = field()
    mods = [dict(D_star=float(D_star[j]), oracle={str(a): float(np.quantile(orc[:, j], 1 - a)) for a in cfg["alphas"]},
                 n1={}) for j in range(K)]
    if sol:
        # true damage of the module under each baseplate: the larger of the junction (bond-wire) and solder damages
        D_s = sol_out[0]
        for j in range(K):
            mods[j]["solder"] = dict(
                D_sol=float(D_s[j]),
                D_star={b: float(max(D_star[j], f * D_s[j])) for b, f in (("AlSiC", 1.0), ("Cu", SOL_CU))},
                oracle={b: {str(a): float(np.quantile(np.maximum(orc[:, j], f * orc_s[:, j]), 1 - a))
                            for a in cfg["alphas"]} for b, f in (("AlSiC", 1.0), ("Cu", SOL_CU))})
        # a certificate of the solder itself: SOL_NB thermal-cycling tests of the field lot at the reference swing of
        # the curve give log-strength scores -log f_life + SOL_SD z (no fit: the curve is given); the margined solder
        # damage of a recorded interval takes the stated swing factor 1.3 and the upper thermal scale 1.10; paired
        # certificate as for the junction. A copper bench would measure lives ten times shorter, so the copper
        # certificate is SOL_CU times the AlSiC one, for the same scores
        Sb = -math.log(f_life) + SOL_SD * rng_s.standard_normal(SOL_NB)
        V_s = SOL_STATED * np.array([_W["solder"][:, q].sum(1) for q in rec[:SOL_NB]])      # [SOL_NB, K]
        for j in range(K):
            mods[j]["solder"]["U_sol"] = {str(a): cc.u6_paired(Sb, V_s[:, j], 0.0, a) for a in SOL_ALPHAS}
    for n1 in cfg["n1"]:
        bn, fi, si, fitd, X, S = bench_scores(world, rng, 2 * n1)
        V = np.array([pool.model(q, fitd, BASE) for q in rec])
        D0 = sum((pool.model(p, fitd, BASE) for p in past), np.zeros(nc))
        V_new = pool.model(new, fitd, BASE)
        # score of each field module at a tested condition drawn from the bench design (calibrated part)
        shift_ref = X[rng.integers(0, len(bn["dT"]), K)] @ (world.b - fitd["b"])
        f_true = dict(b=world.b, cov=fitd["cov"], refs=fitd["refs"], dof=fitd["dof"])
        V_rel0 = pool.model(new, f_true, replace(BASE, stat="none"))
        m = min(len(S), cfg["n2"])
        if cfg.get("anchored"):
            mga = anchored_margin(world, fitd)
            Va = np.array([model_raw(pool, q, fitd, mga) for q in rec[:m]])
            D0a = sum((model_raw(pool, p, fitd, mga) for p in past), np.zeros(nc))
        for j, g in enumerate(groups):
            Vg, D0g = V[:, g], D0[g]
            certs = {}
            for a in cfg["alphas"]:
                certs[str(a)] = {"U1 box": cc.u1_box(S, Vg, D0g), "U2 Monte Carlo": cc.u2_montecarlo(S, Vg, D0g, a, rng),
                                 "U3 Bayes": cc.u3_bayes(S, Vg, D0g, a, rng), "U5 union": cc.u5_union(S, Vg, D0g, a),
                                 "U6 paired": cc.u6_paired(S[:m], Vg[:m], D0g, a),
                                 "U9 PAC": cc.u9_pac(S[:m], Vg[:m], D0g, a, a)}
            U_new = math.exp(-(S_new[j] + shift_ref[j])) * float((D0g + V_new[g]).max())
            V_rel = V_rel0[g] * math.exp(shift_ref[j])
            out = dict(U_new=U_new, stat_ok=bool(np.all(V_rel <= V_new[g] * (1 + 1e-9))), n_scores=int(len(S)),
                       certs=certs)
            if cfg.get("anchored"):
                out["anchored"] = {str(a): cc.u6_paired(S[:m], Va[:, g], D0a[g], a) for a in cfg["alphas"]}
            mods[j]["n1"][str(n1)] = out
        dr = cfg.get("derand")
        if dr and n1 in dr["n1"]:
            # multi-split (19/10/2026): the same bench split again B - 1 times, from a separate stream so that the
            # results above are those of A; the paired certificate of every split at each level, the first being the
            # split above. Their median at alpha/2 covers with probability at least 1 - alpha (Markov's inequality)
            rng_d = np.random.default_rng([cfg["seed"], r, 11, n1])
            ms = [{str(a): [mods[j]["n1"][str(n1)]["certs"][str(a)]["U6 paired"]] for a in dr["alphas"]}
                  for j in range(K)]
            for _ in range(dr["B"] - 1):
                fi2, si2 = tt.split(bn["run"], "half", rng_d)
                fit2 = tt.fit(bn["dT"][fi2], bn["ton"][fi2], bn["I"][fi2], bn["Nf"][fi2], bn["run"][fi2])
                S2 = np.log(bn["Nf"][si2]) - X[si2] @ fit2["b"]
                m2 = min(len(S2), cfg["n2"])
                V2 = np.array([pool.model(q, fit2, BASE) for q in rec[:m2]])
                D02 = sum((pool.model(p, fit2, BASE) for p in past), np.zeros(nc))
                for j, g in enumerate(groups):
                    for a in dr["alphas"]:
                        ms[j][str(a)].append(cc.u6_paired(S2[:m2], V2[:, g], D02[g], a))
            for j in range(K):
                mods[j]["n1"][str(n1)]["multi"] = ms[j]
    out = dict(rep=r, world=world_params(world, b_full), field=dict(S=flist(S_new), kappa=flist(kap),
                                                                     s_th=flist(s_th)), modules=mods)
    if sol:
        out["c_sol"] = c_sol
    return out


# ------------------------------------------------------------------------------------------------------------- C
def strength_draws(S, kind, rng, nmc=4000):
    """Log-strength draws of the parametric certificates: 'mc' from the Gumbel (minimum) law fitted by maximum
    likelihood, as U2; 'bayes' from the posterior predictive of U3 (grid prior 1/sigma). nmc may be a shape."""
    S = np.asarray(S, float)
    if kind == "mc":
        loc, scale = stats.gumbel_l.fit(S)
        return stats.gumbel_l.rvs(loc, scale, size=nmc, random_state=rng)
    mu0, sd, ngrid = S.mean(), S.std(ddof=1), 60
    M, G = np.meshgrid(np.linspace(mu0 - 4 * sd, mu0 + 4 * sd, ngrid), np.linspace(0.2 * sd, 3 * sd, ngrid),
                       indexing="ij")
    ll = stats.gumbel_l.logpdf(S[None, None, :], M[..., None], G[..., None]).sum(-1) - np.log(G)
    w = np.exp(ll - ll.max()).ravel()
    n = int(np.prod(nmc))
    idx = rng.choice(w.size, n, p=w / w.sum())
    return stats.gumbel_l.rvs(M.ravel()[idx], G.ravel()[idx], random_state=rng).reshape(nmc)


def pac_rank(m, alpha, eta):
    """Rank j of the training-conditional (PAC) certificate: the smallest j with P(Bin(m, 1 - alpha) >= j) <= eta, or
    m + 1 (infinite certificate) if none qualifies (Section 3.2, Eq. (5))."""
    for j in range(1, m + 1):
        if stats.binom.sf(j - 1, m, 1 - alpha) <= eta:
            return j
    return m + 1


def reserve_pac(S, V, D0, alpha, eta):
    """Certificate U_(j) and reserve of the PAC version, as g5.reserve('paired') with the rank j of pac_rank (added
    19/10/2026 for the closed loop)."""
    V = np.asarray(V, float)
    m = min(len(S), len(V))
    U = cc.paired_scores(S[:m], V[:m], D0)
    j = pac_rank(m, alpha, eta)
    C = cc.kth(U, j)
    if j > m:
        return C, 0.0
    k = np.min((np.exp(S[:m])[:, None] - D0[None, :]) / V[:m], axis=1)
    return C, float(max(0.0, np.sort(k)[m - j]))


def param_reserve(Sd, V, rows, D0, alpha):
    """Certificate and reserve of a parametric certificate from strength draws Sd and recorded intervals V[rows]: the
    (1 - alpha) quantile of exp(-S) max_chip (D0 + V) and the alpha quantile of min_chip (exp(S) - D0) / V, so that
    C <= 1 exactly when R >= 1 (up to the Monte Carlo error of the quantiles)."""
    Vr = np.asarray(V, float)[rows]
    U = np.exp(-Sd) * (D0[None, :] + Vr).max(1)
    k = np.min((np.exp(Sd)[:, None] - D0[None, :]) / Vr, axis=1)
    return float(np.quantile(U, 1 - alpha)), float(max(0.0, np.quantile(k, alpha)))


def mc_leg_life(S_all, V0, groups, rng, nmc=4000):
    """Life of the leg in intervals under the Monte Carlo model at commissioning: independent Gumbel strengths of the
    modules (fitted to all bench scores) and a recorded interval drawn for each life, its margined damage repeated:
    L = min over modules of exp(S_k) / max_chip V. Returns the draws."""
    loc, scale = stats.gumbel_l.fit(np.asarray(S_all, float))
    Sd = stats.gumbel_l.rvs(loc, scale, size=(nmc, len(groups)), random_state=rng)
    Vr = np.asarray(V0, float)[rng.integers(0, len(V0), nmc)]
    return np.min(np.stack([np.exp(Sd[:, j]) / Vr[:, g].max(1) for j, g in enumerate(groups)], 1), 1)


def best_age(L, r, t_max=T_MAX):
    """Age replacement (Barlow and Hunter 1960) under the life draws L (intervals): the integer age that minimizes
    (1 + (r - 1) P(L <= a)) / E[min(L, a)]."""
    a = np.arange(1, t_max + 1)
    Ls = np.sort(L)
    F = np.searchsorted(Ls, a, side="right") / L.size
    cs = np.r_[0.0, np.cumsum(Ls)]
    n_le = np.searchsorted(Ls, a, side="right")
    Emin = (cs[n_le] + a * (L.size - n_le)) / L.size
    return int(a[np.argmin((1 + (r - 1) * F) / Emin)])


def rep_C(args):
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    sc = scenario(cfg["scen"])
    pools, late, b_full = _W["pools"], _W["late"], _W["b_full"]
    p0 = pools[0.0]
    K, groups, nc = p0.K, p0.groups, len(p0.chips)
    levels = sorted(pools)
    alpha, n1, n2, t_plan = cfg["alpha"], cfg["n1"], cfg["n2"], cfg["t_plan"]
    world = tt.World.draw(rng, b_full, sc)
    bn, fi, si, fitd, X, S_all = bench_scores(world, rng, 3 * n1)
    perm = rng.permutation(len(si))
    S_pol, S_cer = S_all[perm[:len(si) // 2]], S_all[perm[len(si) // 2:]]
    J1 = [p0.sample(rng) for _ in range(n2)]
    J2 = [p0.sample(rng) for _ in range(n2)]
    S_new, kap, s_th = field_leg(world, sc, rng, K)
    life = [p0.sample(rng) for _ in range(T_MAX)]
    V1 = {d: np.array([pools[d].model(q, fitd, BASE) for q in J1]) for d in levels}
    V2 = {d: np.array([pools[d].model(q, fitd, BASE) for q in J2]) for d in levels}
    an = cfg.get("anchor")
    if an:
        # fourth round (20/10/2026): the stated lifetime margins replaced by anchor tests, at full load only (the V runs
        # replace without derating); the anchors draw from their own stream, so the rest of each life is that of the C
        # run with the same seed, and the static ages at commissioning use the anchored margin too
        rng_a = np.random.default_rng([cfg["seed"], r, 55])
        mg_a = anchored_margin(world, fitd) if an == "ideal" else \
            anchored(world, fitd, rng_a, ANCHOR_N, ANCHOR_SET, ANCHOR_DT)[0]
        V1[0.0] = np.array([model_raw(p0, q, fitd, mg_a) for q in J1])
        V2[0.0] = np.array([model_raw(p0, q, fitd, mg_a) for q in J2])
    rho = cfg.get("rho", 0.0)
    # parametric draws on their own stream: strength law fitted to all bench scores, policy on J1, certificate on J2
    rng_p = np.random.default_rng([cfg["seed"], r, 99])
    nmc = 4000
    rows_pol, rows_cer = rng_p.integers(0, n2, nmc), rng_p.integers(0, n2, nmc)
    par = {kind: strength_draws(S_all, kind, rng_p, nmc) for kind in ("mc", "bayes")}
    # static ages at commissioning (no feedback): reserve of the paired certificate on all bench scores and J2, the
    # B-life (alpha-quantile) of the leg under the Monte Carlo model and its cost-optimal ages
    R_cert = min(g5.reserve("paired", S_all, V2[0.0][:, g], np.zeros(len(g)), alpha)[1] for g in groups)
    L_mc = mc_leg_life(S_all, V2[0.0], groups, rng_p, nmc)
    ages = dict(cert_age=int(math.floor(R_cert)), mc_age=int(math.floor(np.quantile(L_mc, alpha))),
                mc_opt10=best_age(L_mc, 10), mc_opt50=best_age(L_mc, 50))
    cache_V, cache_D = {}, {}

    def model_t(t, d):
        if (t, d) not in cache_V:
            cache_V[(t, d)] = model_raw(p0, life[t], fitd, mg_a) if an and d == 0.0 else \
                pools[d].model(life[t], fitd, BASE)
        return cache_V[(t, d)]

    def true_t(t, d, Dt):
        """True damage per chip of the field leg in interval t at level d; with rho > 0 the thermal scale of each module
        rises with its true damage (then nothing is cached)."""
        if rho == 0.0:
            if (t, d) not in cache_D:
                cache_D[(t, d)] = pools[d].true(life[t], world, S_new, kap, s_th)
            return cache_D[(t, d)]
        return pools[d].true(life[t], world, S_new, kap, s_th * (1 + rho * p0.mod(Dt)))

    def level_ok(meth, d, D0, aci):
        """Whether a level passes on the policy data (both modules) and its reserve (the smaller one)."""
        Cs, Rs = [], []
        for j, g in enumerate(groups):
            if meth == "union":
                C, R = g5.reserve("union", S_all, V1[d][:, g], D0[g], alpha)
            elif meth == "reuse":
                C, R = g5.reserve("paired", S_cer, V2[d][:, g], D0[g], alpha)
            elif meth == "aci":
                C, R = g5.reserve("paired", S_pol, V1[d][:, g], D0[g], alpha, level=aci[j].a)
            elif meth == "split0":
                # without re-simulation the recorded intervals carry the base damage at every level, so the policy
                # cannot see what derating buys
                C, R = g5.reserve("paired", S_pol, V1[0.0][:, g], D0[g], alpha)
            elif meth in ("mc", "bayes"):
                C, R = param_reserve(par[meth], V1[d][:, g], rows_pol, D0[g], alpha)
            elif meth == "pac":
                C, R = reserve_pac(S_pol, V1[d][:, g], D0[g], alpha, cfg.get("eta", alpha))
            else:
                C, R = g5.reserve("paired", S_pol, V1[d][:, g], D0[g], alpha)
            Cs.append(C)
            Rs.append(R)
        return max(Cs) <= 1, min(Rs)

    def certify(meth, d, D0, aci):
        """Certificate of each module for the coming interval at the chosen level, on the certification data."""
        m = min(len(S_cer), n2)
        out = []
        for j, g in enumerate(groups):
            if meth == "union":
                out.append(cc.u5_union(S_all, V2[d][:, g], D0[g], alpha))
            elif meth == "aci":
                out.append(aci[j].certificate(S_cer[:m], V2[d][:m][:, g], D0[g]))
            elif meth in ("mc", "bayes"):
                out.append(param_reserve(par[meth], V2[d][:, g], rows_cer, D0[g], alpha)[0])
            elif meth == "pac":
                out.append(cc.u9_pac(S_cer[:m], V2[d][:m][:, g], D0[g], alpha, cfg.get("eta", alpha)))
            else:
                Vc = V2[0.0] if meth == "split0" else V2[d]
                out.append(cc.u6_paired(S_cer[:m], Vc[:m][:, g], D0[g], alpha))
        return np.array(out)

    def record(hist, bits, Dt, failed, aci=None, age=None):
        n = len(hist)
        ds = np.array([x[0] for x in hist]) if n else np.zeros(0)
        cov = np.array([[(b >> j) & 1 for j in range(K)] for b in bits], float) if bits else None
        return dict(intervals=n, harvest=float(Dt.max()), harvest_mod=flist(p0.mod(Dt)), failed=bool(failed),
                    collapse=n == 0,
                    coverage=float(cov.mean()) if cov is not None and n else None,
                    cov_all=float(cov.min(1).mean()) if cov is not None and n else None,
                    bits=list(bits) if bits is not None else None,
                    derated_share=float(np.mean(ds > 0)) if n else None,
                    false_derating=float(np.mean([x[1] for x in hist])) if n and hist[0][1] is not None else None,
                    false_share_of_derated=(float(np.mean([x[1] for x in hist if x[0] > 0]))
                                            if n and hist[0][1] is not None and any(x[0] > 0 for x in hist) else None),
                    mean_level=float(ds.mean()) if n else None,
                    late_share=float(np.mean([late[d] for d in ds])) if n else None,
                    late_sum=float(sum(late[d] for d in ds)),
                    aci_level=float(np.mean([x.a for x in aci])) if aci is not None else None, age=age)

    def run_cert(policy, meth):
        D0, Dt = np.zeros(nc), np.zeros(nc)
        # adaptive conformal inference: one level per module, each driven by the miss indicator of its own module, so
        # that each module, like the other certificates, aims at 1 - alpha
        aci = [cc.ACI(alpha, eta=0.005) for _ in groups] if meth == "aci" else None
        hist, bits, failed = [], [], False
        lev = [0.0] if policy == "replace" else levels
        for t in range(T_MAX):
            cand = []
            for d in lev:
                ok, R = level_ok(meth, d, D0, aci)
                if ok:
                    cand.append((d, R))
            if policy == "reserve_cap":
                cand = [(d, R) for d, R in cand if late[d] <= cfg["late_max"]]
            elif policy == "reserve_budget":
                spent = sum(late[x[0]] for x in hist)
                cand = [(d, R) for d, R in cand if (spent + late[d]) / (len(hist) + 1) <= cfg["late_max"]]
            if not cand:
                break
            if policy in ("pass", "replace"):
                d_sel = cand[0][0]
            elif policy == "fixed20":
                d_sel = 0.2 if any(abs(d - 0.2) < 1e-9 for d, _ in cand) else cand[0][0]
            else:
                ok = [d for d, R in cand if R >= max(t_plan - t, 1)]
                d_sel = ok[0] if ok else max(cand, key=lambda x: x[1])[0]
            Cc = certify(meth, d_sel, D0, aci)
            Vt = model_t(t, d_sel)
            Dn = true_t(t, d_sel, Dt)
            Dn0 = Dn if d_sel == 0.0 else true_t(t, 0.0, Dt)
            need = max(t_plan - t, 1)
            # a derating the leg did not need: at full load no chip would reach its budget before the planned replacement
            false_der = bool(d_sel > 0 and np.all((1 - Dt) / np.maximum(Dn0, 1e-12) >= need))
            cov = p0.mod(Dt + Dn) <= Cc
            hist.append((d_sel, false_der))
            bits.append(int(sum(1 << j for j in range(K) if cov[j])))
            if aci is not None:
                for j in range(K):
                    aci[j].update(not bool(cov[j]))
            D0, Dt = D0 + Vt, Dt + Dn
            if Dt.max() >= 1:
                failed = True
                break
        return record(hist, bits, Dt, failed, aci)

    def run_static(d, age):
        """No certificate in the loop: level d throughout, replacement at the age (None: run to failure)."""
        Dt = np.zeros(nc)
        hist, failed = [], False
        for t in range(T_MAX if age is None else max(0, min(int(age), T_MAX))):
            Dt = Dt + true_t(t, d, Dt)
            hist.append((d, None))
            if Dt.max() >= 1:
                failed = True
                break
        return record(hist, None, Dt, failed, age=None if age is None else max(0, min(int(age), T_MAX)))

    out = {}
    for policy, meth in cfg["runs"]:
        if policy != "none":
            out[f"{policy}|{meth}"] = run_cert(policy, meth)
        elif meth == "tbr60":
            out["none|tbr60"] = run_static(0.0, t_plan)
        elif meth == "rtf":
            out["none|rtf"] = run_static(0.0, None)
        elif meth == "fix20_tbr60":
            out["none|fix20_tbr60"] = run_static(0.2, t_plan)
        else:
            out[f"none|{meth}"] = run_static(0.0, ages[meth])
    return dict(rep=r, world=world_params(world, b_full), field=dict(S=flist(S_new), kappa=flist(kap),
                                                                      s_th=flist(s_th)),
                ages=ages, R_cert=float(R_cert), n_scores=[len(S_pol), len(S_cer), len(S_all)], runs=out)


# ------------------------------------------------------------------------------------------------------------- D
def rep_D(args):
    """Boundary truth at the base margin; certificates with the base margin and with narrower variants."""
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    pool, b_full = _W["pool"], _W["b_full"]
    sc = tt.Scenario.boundary()
    world = tt.World.draw(rng, b_full, sc)
    bn, fi, si, fitd, X, S = bench_scores(world, rng, 2 * cfg["n1"])
    rec = [pool.sample(rng) for _ in range(cfg["n2"])]
    S_new = world.strengths(rng, pool.K)
    new = pool.sample(rng)
    D_star = pool.mod(pool.true(new, world, S_new, 0.85, 1.10))
    m = min(len(S), cfg["n2"])
    Vv = {name: np.array([model_raw(pool, q, fitd, VARIANTS[name]) for q in rec[:m]]) for name in ("base",) + NARROW}
    mods = [dict(D_star=float(D_star[j]), variants={name: cc.u6_paired(S[:m], V[:, g], 0.0, cfg["alpha"])
                                                    for name, V in Vv.items()}) for j, g in enumerate(pool.groups)]
    return dict(rep=r, modules=mods)


# ------------------------------------------------------------------------------------------------------------- E
def rep_E(args):
    """Season shift: the field interval runs at ambient + delta; recorded intervals as recorded or re-simulated."""
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    pool, b_full = _W["pool"], _W["b_full"]
    sc = scenario(cfg["scen"])
    world = tt.World.draw(rng, b_full, sc)
    bn, fi, si, fitd, X, S = bench_scores(world, rng, 2 * cfg["n1"])
    rec = [pool.sample(rng) for _ in range(cfg["n2"])]
    S_new, kap, s_th = field_leg(world, sc, rng, pool.K)
    new = pool.sample(rng)
    m = min(len(S), cfg["n2"])
    V0 = np.array([pool.model(q, fitd, BASE) for q in rec[:m]])
    mods = [dict(deltas={}) for _ in range(pool.K)]
    for dlt in cfg["deltas"]:
        D_star = pool.mod(pool.true(new, world, S_new, kap, s_th, shift=dlt))
        Vs = V0 if dlt == 0 else np.array([model_raw(pool, q, fitd, BASE, shift=dlt) for q in rec[:m]])
        for j, g in enumerate(pool.groups):
            mods[j]["deltas"][str(dlt)] = dict(D_star=float(D_star[j]),
                                               C_recorded=cc.u6_paired(S[:m], V0[:, g], 0.0, cfg["alpha"]),
                                               C_resim=cc.u6_paired(S[:m], Vs[:, g], 0.0, cfg["alpha"]),
                                               V_ratio=float(np.median(Vs[:, g].max(1) / V0[:, g].max(1))))
    return dict(rep=r, modules=mods)


# ------------------------------------------------------------------------------------------------------------- N
def anchor_bound(world, fitd, rng, n_a, t_a, dT_a, conf=0.95):
    """Upper confidence bound on the heating-time effect at t_a (log-life lost against the clipped model) from n_a
    anchor tests at swing dT_a, T_min 60 C and the current of the field model (890 A). The intercept of the fit centres
    the strength law, so the mean anchor residual estimates minus the effect; its error is the anchor scatter plus the
    fit error at the anchor condition. Uses the fit half and the anchors only, not the calibration scores."""
    S_a = world.strengths(rng, n_a)
    dT, ton = np.full(n_a, dT_a), np.full(n_a, t_a)
    x_a = tt.features(*tt.clip_x(dT[:1], ton[:1]))[0]
    r = world.log_life(dT, np.full(n_a, tt.TMIN_T), ton, S_a) - tt.features(*tt.clip_x(dT, ton)) @ fitd["b"]
    se = math.sqrt(r.var(ddof=1) / n_a + float(x_a @ fitd["cov"] @ x_a))
    d_hat = -float(r.mean())
    return max(0.0, d_hat + float(stats.t.ppf(conf, n_a - 1)) * se), d_hat


def anchored(world, fitd, rng, n_a, t_set, dT_a, conf=0.95):
    """Margin with the heating-time rule replaced by n_a anchor tests at each heating time of t_set (joint level conf,
    Bonferroni over the anchor times); returns the margin and the bounds."""
    b = [anchor_bound(world, fitd, rng, n_a, t, dT_a, 1 - (1 - conf) / len(t_set))[0] for t in t_set]
    return replace(BASE, ton_rule="anchor", t_a=tuple(t_set), delta_a=tuple(b)), b


def true_effect(world, t):
    """True heating-time effect at t (log-life lost against the clipped model)."""
    return world.gamma_t * math.log(min(max(t, tt.TON_HI), tt.TON_CAP) / tt.TON_HI) + \
        world.gamma_h * math.log(max(t / tt.TON_CAP, 1.0))


def set_name(t_set):
    return "+".join(f"{t:g}" for t in t_set)


def rep_N(args):
    """Anchor scenario: the stated heating-time margin replaced by the bounds of n_a anchor tests at each heating time of
    an anchor set (Margin ton_rule 'anchor', monotonicity only below the last anchor); the other stated margins kept.
    Paired certificate against the stated one in the same world; the anchors belong to the world, the certificates and
    the true damage to each module."""
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    sc = scenario(cfg["scen"])
    pool, b_full = _W["pool"], _W["b_full"]
    world = tt.World.draw(rng, b_full, sc)
    bn, fi, si, fitd, X, S = bench_scores(world, rng, 2 * cfg["n1"])
    rec = [pool.sample(rng) for _ in range(cfg["n2"])]
    S_new, kap, s_th = field_leg(world, sc, rng, pool.K)
    new = pool.sample(rng)
    D_star = pool.mod(pool.true(new, world, S_new, kap, s_th))
    m = min(len(S), cfg["n2"])
    V = np.array([pool.model(q, fitd, BASE) for q in rec[:m]])
    times = sorted({t for ts in cfg["sets"] for t in ts})
    te = {f"{t:g}": true_effect(world, t) for t in times}
    mods = [dict(D_star=float(D_star[j]), true_effect=te,
                 stated={str(a): cc.u6_paired(S[:m], V[:, g], 0.0, a) for a in cfg["alphas"]}, anchors={})
            for j, g in enumerate(pool.groups)]
    for ts in cfg["sets"]:
        for n_a in cfg["n_a"]:
            mg, b = anchored(world, fitd, rng, n_a, ts, cfg["dT_a"])
            Va = np.array([model_raw(pool, q, fitd, mg) for q in rec[:m]])
            for j, g in enumerate(pool.groups):
                mods[j]["anchors"][f"{set_name(ts)}|{n_a}"] = dict(
                    bounds=b, certs={str(a): cc.u6_paired(S[:m], Va[:, g], 0.0, a) for a in cfg["alphas"]})
    return dict(rep=r, world=world_params(world, b_full), modules=mods)


# ------------------------------------------------------------------------------------------------------------- X
def rep_X(args):
    """Exceedance sweep: the boundary truth with both heating-time slopes (4.5-60 s and above 60 s) set to g, from the
    stated 0.5 upwards; the same bench, fit, recorded intervals and field leg for every g. Stated certificate against
    certificates with n_a anchor tests at t_a drawn in each world."""
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    pool, b_full = _W["pool"], _W["b_full"]
    sc = tt.Scenario.boundary()
    world0 = tt.World.draw(rng, b_full, sc)
    bn, fi, si, fitd, X, S = bench_scores(world0, rng, 2 * cfg["n1"])
    rec = [pool.sample(rng) for _ in range(cfg["n2"])]
    S_new, kap, s_th = field_leg(world0, sc, rng, pool.K)
    new = pool.sample(rng)
    m = min(len(S), cfg["n2"])
    V = np.array([pool.model(q, fitd, BASE) for q in rec[:m]])
    mods = [dict(stated={str(a): cc.u6_paired(S[:m], V[:, g], 0.0, a) for a in cfg["alphas"]}, g={})
            for g in pool.groups]
    for gg in cfg["g"]:
        w = replace(world0, gamma_t=gg, gamma_h=gg)
        D_star = pool.mod(pool.true(new, w, S_new, kap, s_th))
        outs = [dict(D_star=float(D_star[j]), anchors={}) for j in range(pool.K)]
        for ts in cfg["sets"]:
            for n_a in cfg["n_a"]:
                mg, b = anchored(w, fitd, rng, n_a, ts, cfg["dT_a"])
                Va = np.array([model_raw(pool, q, fitd, mg) for q in rec[:m]])
                for j, g in enumerate(pool.groups):
                    outs[j]["anchors"][f"{set_name(ts)}|{n_a}"] = dict(
                        bounds=b, certs={str(a): cc.u6_paired(S[:m], Va[:, g], 0.0, a) for a in cfg["alphas"]})
        for j in range(pool.K):
            mods[j]["g"][str(gg)] = outs[j]
    return dict(rep=r, modules=mods)


# ------------------------------------------------------------------------------------------------------------- T
def date_positions(pool):
    """Positions (in the pool) of the vehicle days of each calendar date, dates in increasing order."""
    if "dates" not in _W:
        _W["dates"] = [np.flatnonzero(pool.date == x) for x in np.unique(pool.date)]
    return _W["dates"]


def block(pool, rng, start, n_days=g5.INTERVAL_DAYS):
    """An interval of one vehicle day on each of n_days consecutive calendar dates, from date number start."""
    dates = date_positions(pool)
    return np.array([rng.choice(dates[k]) for k in range(start, start + n_days)])


def rep_T(args):
    """Temporal dependence: one world, bench and new field leg; the recorded and coming intervals built from resampled
    days, from consecutive dates anywhere, and from consecutive dates split forward and backward in time."""
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    sc = scenario(cfg["scen"])
    pool, b_full = _W["pool"], _W["b_full"]
    world = tt.World.draw(rng, b_full, sc)
    nd, L = len(date_positions(pool)), g5.INTERVAL_DAYS
    S_new, kap, s_th = field_leg(world, sc, rng, pool.K)
    n1, n2 = cfg["n1"], cfg["n2"]
    bn, fi, si, fitd, X, S = bench_scores(world, rng, 2 * n1)
    shift_ref = X[rng.integers(0, len(bn["dT"]), pool.K)] @ (world.b - fitd["b"])
    m = min(len(S), n2)
    designs = {
        "resampled": (lambda: pool.sample(rng), lambda: pool.sample(rng)),
        "blocks": (lambda: block(pool, rng, int(rng.integers(0, nd - L + 1))),
                   lambda: block(pool, rng, int(rng.integers(0, nd - L + 1)))),
        "forward": (lambda: block(pool, rng, int(rng.integers(0, nd - 2 * L + 1))), lambda: block(pool, rng, nd - L)),
        "backward": (lambda: block(pool, rng, int(rng.integers(L, nd - L + 1))), lambda: block(pool, rng, 0)),
    }
    mods = [dict(designs={}) for _ in range(pool.K)]
    for name, (recorded, coming) in designs.items():
        rec = [recorded() for _ in range(n2)]
        new = coming()
        V = np.array([pool.model(q, fitd, BASE) for q in rec])
        V_new = pool.model(new, fitd, BASE)
        D_star = pool.mod(pool.true(new, world, S_new, kap, s_th))
        for j, g in enumerate(pool.groups):
            mods[j]["designs"][name] = dict(
                D_star=float(D_star[j]), U_new=math.exp(-(S_new[j] + shift_ref[j])) * float(V_new[g].max()),
                V_new=float(V_new[g].max()), V_rec_median=float(np.median(V[:, g].max(axis=1))),
                paired={str(a): cc.u6_paired(S[:m], V[:m][:, g], 0.0, a) for a in cfg["alphas"]},
                union={str(a): cc.u5_union(S, V[:, g], np.zeros(len(g)), a) for a in cfg["alphas"]})
    return dict(rep=r, modules=mods)


# ------------------------------------------------------------------------------------------------------------- L
# field lot against the bench lot: shift of the log-strength and factor on its scatter about the median of the bench law
LOT = {"same": (0.0, 1.0), "mean -0.1": (-0.1, 1.0), "mean -0.2": (-0.2, 1.0), "scatter x1.5": (0.0, 1.5),
       "scatter x2": (0.0, 2.0)}


def rep_L(args):
    """Bench-to-field lot shift: the field modules come from a lot whose strength law is moved or widened against that
    of the bench (LOT); the world, bench, fit, recorded intervals and field legs are the same for every variant (common
    draws of the base law, transformed). Paired, union, Monte Carlo and Bayes certificates on the bench lot, and the
    paired certificate recalibrated on n_lot modules of the field lot tested on the bench design and scored against the
    same fit."""
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    sc = scenario(cfg["scen"])
    pool, b_full = _W["pool"], _W["b_full"]
    K = pool.K
    world = tt.World.draw(rng, b_full, sc)
    n1, n2 = cfg["n1"], cfg["n2"]
    bn, fi, si, fitd, X, S = bench_scores(world, rng, 2 * n1)
    rec = [pool.sample(rng) for _ in range(n2)]
    V = np.array([pool.model(q, fitd, BASE) for q in rec])
    m = min(len(S), n2)
    # field legs at log-strength 0: the true damage of a module of log-strength S is exp(-S) times its value there, and
    # so is its score at a reference condition drawn with the bench proportions (calibrated part; the conditions come
    # from their own stream, so the true damages do not depend on them)
    D_unit, V_new = [], []
    for _ in range(cfg["n_field"]):
        kap, s_th = rng.uniform(*sc.kappa, size=K), rng.uniform(*sc.s_th, size=K)
        q = pool.sample(rng)
        D_unit.append(pool.mod(pool.true(q, world, 0.0, kap, s_th)))
        V_new.append(pool.mod(pool.model(q, fitd, BASE)))
    D_unit, V_new = np.array(D_unit), np.array(V_new)                               # [n_field, K]
    jj = np.random.default_rng([cfg["seed"], r, 11]).integers(0, len(bn["dT"]), (cfg["n_field"], K))
    U_unit = np.exp(-(X[jj] @ (world.b - fitd["b"]))) * V_new
    S_field = world.strengths(rng, cfg["n_field"] * K).reshape(cfg["n_field"], K)
    h = tt.hernes()
    nl = max(cfg["n_lot"])
    rows = rng.integers(0, len(h["dT"]), nl)
    shift_lot = tt.features(h["dT"][rows], h["ton"][rows], h["I"][rows]) @ (world.b - fitd["b"])
    S_lot = world.strengths(rng, nl)
    med = float(np.median(world.strengths(np.random.default_rng([cfg["seed"], r, 7]), 200000)))
    mods = []
    for j, g in enumerate(pool.groups):
        Vg = V[:, g]
        D0 = np.zeros(len(g))
        certs = {str(a): {"U6 paired": cc.u6_paired(S[:m], Vg[:m], D0, a), "U5 union": cc.u5_union(S, Vg, D0, a),
                          "U2 Monte Carlo": cc.u2_montecarlo(S, Vg, D0, a, rng),
                          "U3 Bayes": cc.u3_bayes(S, Vg, D0, a, rng)} for a in cfg["alphas"]}
        res = dict(certs=certs, variants={})
        for name, (dm, fs) in LOT.items():
            def law(s):
                return med + fs * (s - med) + dm
            D = np.exp(-law(S_field[:, j])) * D_unit[:, j]
            U = np.exp(-law(S_field[:, j])) * U_unit[:, j]
            S_l = law(S_lot) + shift_lot
            recal = {f"{n}|{a}": cc.u6_paired(S_l[:n], Vg[:n], D0, a) for n in cfg["n_lot"] for a in cfg["alphas"]}
            cov = {a: {k: float(np.mean(D <= c)) for k, c in certs[a].items()} for a in certs}
            cal = {a: {k: float(np.mean(U <= c)) for k, c in certs[a].items()} for a in certs}
            for k, c in recal.items():
                cov[k.split("|")[1]][f"recal {k.split('|')[0]}"] = float(np.mean(D <= c))
                cal[k.split("|")[1]][f"recal {k.split('|')[0]}"] = float(np.mean(U <= c))
            res["variants"][name] = dict(D_median=float(np.median(D)), recal=recal, coverage=cov, calibrated=cal)
        mods.append(res)
    return dict(rep=r, modules=mods)


# ------------------------------------------------------------------------------------------------------------- Y
# stated components exceeded together: the truth at the margins with components moved beyond their stated values, one
# at a time and cumulatively (Miner factor 0.80 against 0.85, thermal scale 1.15 against 1.10, both heating-time slopes
# 0.6 against 0.5, swing exponent below 26 K lower by 2.0 against 1.2)
Y_STEPS = {"kappa": dict(kappa=0.80), "thermal": dict(s_th=1.15), "heating": dict(gamma=0.6),
           "swing": dict(beta_shift=-2.0)}
Y_VARIANTS = {"none": (), "kappa": ("kappa",), "thermal": ("thermal",), "heating": ("heating",), "swing": ("swing",),
              "kappa+thermal": ("kappa", "thermal"), "kappa+thermal+heating": ("kappa", "thermal", "heating"),
              "all four": ("kappa", "thermal", "heating", "swing")}


def rep_Y(args):
    """Several stated components exceeded at once, from the truth at the margins (Y_VARIANTS); the same world, bench,
    fit, recorded intervals and field legs for every variant. Paired, union, Monte Carlo and Bayes certificates."""
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    pool, b_full = _W["pool"], _W["b_full"]
    K = pool.K
    sc = tt.Scenario.boundary()
    world0 = tt.World.draw(rng, b_full, sc)
    bn, fi, si, fitd, X, S = bench_scores(world0, rng, 2 * cfg["n1"])
    rec = [pool.sample(rng) for _ in range(cfg["n2"])]
    V = np.array([pool.model(q, fitd, BASE) for q in rec])
    m = min(len(S), cfg["n2"])
    certs = []
    for g in pool.groups:
        Vg, D0 = V[:, g], np.zeros(len(g))
        certs.append({str(a): {"U6 paired": cc.u6_paired(S[:m], Vg[:m], D0, a), "U5 union": cc.u5_union(S, Vg, D0, a),
                               "U2 Monte Carlo": cc.u2_montecarlo(S, Vg, D0, a, rng),
                               "U3 Bayes": cc.u3_bayes(S, Vg, D0, a, rng)} for a in cfg["alphas"]})
    S_field = world0.strengths(rng, cfg["n_field"] * K).reshape(cfg["n_field"], K)
    days = [pool.sample(rng) for _ in range(cfg["n_field"])]
    cache = {}

    def unit_damage(steps):
        """True damage at log-strength 0 of every field module, for the exceeded components in steps: [n_field, K]."""
        p = {}
        for s in steps:
            p.update(Y_STEPS[s])
        key = (p.get("s_th", 1.10), p.get("gamma", 0.5), p.get("beta_shift", -1.2))
        if key not in cache:
            w = replace(world0, gamma_t=key[1], gamma_h=key[1], beta_low=abs(b_full[1]) + key[2])
            cache[key] = np.array([pool.mod(pool.true(q, w, 0.0, 0.85, key[0])) for q in days])
        return cache[key] * 0.85 / p.get("kappa", 0.85)

    mods = [dict(certs=certs[j], variants={}) for j in range(K)]
    for name, steps in Y_VARIANTS.items():
        D = np.exp(-S_field) * unit_damage(steps)
        for j in range(K):
            mods[j]["variants"][name] = dict(D_median=float(np.median(D[:, j])),
                                             coverage={a: {k: float(np.mean(D[:, j] <= c)) for k, c in
                                                           certs[j][a].items()} for a in certs[j]})
    return dict(rep=r, modules=mods)


# ------------------------------------------------------------------------------------------------------------- R
def rep_R(args):
    """Re-simulation necessity (added 18/10/2026): the field leg runs its coming interval under a changed policy (a
    balancing pattern, or coasting instead of cruising) while the recorded intervals ran under the base policy. Paired
    certificate of each module on the recorded intervals as they are (recorded) and re-simulated under the changed
    policy (resim), against the true damage under it; the base policy itself is the control. The pools hold the same
    vehicle days, so the same day indices give the same timetable days under each policy."""
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    sc = scenario(cfg["scen"])
    pools, b_full = _W["pools"], _W["b_full"]
    pb = pools["base"]
    world = tt.World.draw(rng, b_full, sc)
    bn, fi, si, fitd, X, S = bench_scores(world, rng, 2 * cfg["n1"])
    rec = [pb.sample(rng) for _ in range(cfg["n2"])]
    S_new, kap, s_th = field_leg(world, sc, rng, pb.K)
    new = pb.sample(rng)
    m = min(len(S), cfg["n2"])
    Vb = np.array([pb.model(q, fitd, BASE) for q in rec[:m]])
    mods = [dict(patterns={}) for _ in range(pb.K)]
    for p in ("base",) + tuple(cfg["patterns"]):
        Vp = Vb if p == "base" else np.array([pools[p].model(q, fitd, BASE) for q in rec[:m]])
        D_star = pb.mod(pools[p].true(new, world, S_new, kap, s_th))
        for j, g in enumerate(pb.groups):
            mods[j]["patterns"][p] = dict(
                D_star=float(D_star[j]), V_ratio=float(np.median(Vp[:, g].max(1) / Vb[:, g].max(1))),
                recorded={str(a): cc.u6_paired(S[:m], Vb[:, g], 0.0, a) for a in cfg["alphas"]},
                resim={str(a): cc.u6_paired(S[:m], Vp[:, g], 0.0, a) for a in cfg["alphas"]})
    return dict(rep=r, modules=mods)


# ------------------------------------------------------------------------------------------------------------- U
def rep_U(args):
    """Paired certificate against union certificates with the levels allocated in advance, alpha_S + alpha_V = alpha
    with a small alpha_V ('union a/2' is the equal split of cc.u5_union): coverage of the true damage, of the paired
    score U_new, and size. Added 19/10/2026 (third review)."""
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    sc = scenario(cfg["scen"])
    pool, b_full = _W["pool"], _W["b_full"]
    K, groups = pool.K, pool.groups
    world = tt.World.draw(rng, b_full, sc)
    S_new, kap, s_th = field_leg(world, sc, rng, K)
    new = pool.sample(rng)
    D_star = pool.mod(pool.true(new, world, S_new, kap, s_th))
    rec = [pool.sample(rng) for _ in range(cfg["n2"])]
    mods = [dict(D_star=float(D_star[j]), n1={}) for j in range(K)]
    for n1 in cfg["n1"]:
        bn, fi, si, fitd, X, S = bench_scores(world, rng, 2 * n1)
        V = np.array([pool.model(q, fitd, BASE) for q in rec])
        V_new = pool.model(new, fitd, BASE)
        shift_ref = X[rng.integers(0, len(bn["dT"]), K)] @ (world.b - fitd["b"])
        m = min(len(S), cfg["n2"])
        for j, g in enumerate(groups):
            Vg, D0g = V[:, g], np.zeros(len(g))
            certs = {}
            for a in cfg["alphas"]:
                c = {"paired": cc.u6_paired(S[:m], Vg[:m], D0g, a), "union a/2": cc.u5_union(S, Vg, D0g, a)}
                for aV in cfg["a_V"]:
                    if aV < a:
                        c[f"union aV={aV}"] = cc.u5_union_split(S, Vg, D0g, a - aV, aV)
                certs[str(a)] = c
            mods[j]["n1"][str(n1)] = dict(U_new=math.exp(-(S_new[j] + shift_ref[j])) * float(V_new[g].max()),
                                          n_scores=int(len(S)), certs=certs)
    return dict(rep=r, world=world_params(world, b_full), modules=mods)


# ------------------------------------------------------------------------------------------------------------- M
def rep_M(args):
    """Mission change within a leg life (third review, 19/10/2026): the intervals before t_sw, drawn uniformly in
    cfg['t_sw'], run under the first driving strategy and the rest under the second, on the same exogenous days and
    without derating. The certificates (replacement when either module fails on the policy data) re-simulate the
    recorded days under the strategy in force; 'paired stale' keeps them under the first strategy. Static ages are set
    at commissioning for the first strategy; 'replanned' ages are recomputed at the change, for the second strategy,
    from the margined damage accrued so far: the paired reserve on all bench scores, and the B-life and cost-optimal
    ages of the Monte Carlo law conditioned on survival. A life ends at replacement or failure."""
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    sc = scenario(cfg["scen"])
    pools, b_full = _W["pools"], _W["b_full"]
    pa, pb = pools[cfg["first"]], pools[cfg["second"]]
    K, groups, nc = pa.K, pa.groups, len(pa.chips)
    alpha, n1, n2 = cfg["alpha"], cfg["n1"], cfg["n2"]
    world = tt.World.draw(rng, b_full, sc)
    bn, fi, si, fitd, X, S_all = bench_scores(world, rng, 3 * n1)
    perm = rng.permutation(len(si))
    S_pol, S_cer = S_all[perm[:len(si) // 2]], S_all[perm[len(si) // 2:]]
    J1 = [pa.sample(rng) for _ in range(n2)]
    J2 = [pa.sample(rng) for _ in range(n2)]
    S_new, kap, s_th = field_leg(world, sc, rng, K)
    life = [pa.sample(rng) for _ in range(T_MAX)]
    t_sw = int(rng.integers(cfg["t_sw"][0], cfg["t_sw"][1] + 1))
    VA1, VB1 = (np.array([p.model(q, fitd, BASE) for q in J1]) for p in (pa, pb))
    VA2, VB2 = (np.array([p.model(q, fitd, BASE) for q in J2]) for p in (pa, pb))
    rng_p = np.random.default_rng([cfg["seed"], r, 99])
    nmc = 4000
    rows_pol, rows_cer = rng_p.integers(0, n2, nmc), rng_p.integers(0, n2, nmc)
    par_mc = strength_draws(S_all, "mc", rng_p, nmc)
    R_cert = min(g5.reserve("paired", S_all, VA2[:, g], np.zeros(len(g)), alpha)[1] for g in groups)
    L_mc = mc_leg_life(S_all, VA2, groups, rng_p, nmc)
    ages = dict(cert_age=int(math.floor(R_cert)), mc_age=int(math.floor(np.quantile(L_mc, alpha))),
                mc_opt10=best_age(L_mc, 10), mc_opt50=best_age(L_mc, 50))
    cache_V, cache_D = {}, {}

    def pool_t(t):
        return pa if t < t_sw else pb

    def model_t(t):
        if t not in cache_V:
            cache_V[t] = pool_t(t).model(life[t], fitd, BASE)
        return cache_V[t]

    def true_t(t):
        if t not in cache_D:
            cache_D[t] = pool_t(t).true(life[t], world, S_new, kap, s_th)
        return cache_D[t]

    # replanned ages at the change, from the margined damage accrued under the first strategy
    D0_sw = sum((model_t(t) for t in range(t_sw)), np.zeros(nc))
    rng_r = np.random.default_rng([cfg["seed"], r, 98])
    loc, scale = stats.gumbel_l.fit(S_all)
    Sd = stats.gumbel_l.rvs(loc, scale, size=(nmc, K), random_state=rng_r)
    Vr = VB2[rng_r.integers(0, n2, nmc)]
    L_rem, alive = np.full(nmc, np.inf), np.ones(nmc, bool)
    for j, g in enumerate(groups):
        rem = np.exp(Sd[:, j])[:, None] - D0_sw[g][None, :]
        alive &= np.all(rem > 0, axis=1)
        L_rem = np.minimum(L_rem, np.min(rem / Vr[:, g], axis=1))
    L_rem = L_rem[alive]
    R_rp = min(g5.reserve("paired", S_all, VB2[:, g], D0_sw[g], alpha)[1] for g in groups)
    replanned = dict(cert_age=t_sw + int(math.floor(R_rp)),
                     mc_age=t_sw + (int(math.floor(np.quantile(L_rem, alpha))) if L_rem.size else 0),
                     mc_opt10=t_sw + (best_age(L_rem, 10) if L_rem.size else 0),
                     mc_opt50=t_sw + (best_age(L_rem, 50) if L_rem.size else 0))

    def certificates(meth, V, D0, rows, S_sc):
        out = []
        for g in groups:
            if meth == "union":
                out.append(cc.u5_union(S_all, V[:, g], D0[g], alpha))
            elif meth == "mc":
                out.append(param_reserve(par_mc, V[:, g], rows, D0[g], alpha)[0])
            else:
                m = min(len(S_sc), len(V))
                out.append(cc.u6_paired(S_sc[:m], V[:m][:, g], D0[g], alpha))
        return np.array(out)

    def run_cert(meth):
        D0, Dt = np.zeros(nc), np.zeros(nc)
        bits, failed, n = [], False, 0
        for t in range(T_MAX):
            after = t >= t_sw and meth != "paired stale"
            V1, V2 = (VB1, VB2) if after else (VA1, VA2)
            base = "paired" if meth == "paired stale" else meth
            if np.max(certificates(base, V1, D0, rows_pol, S_pol)) > 1:
                break
            Cc = certificates(base, V2, D0, rows_cer, S_cer)
            Dn = true_t(t)
            cov = pa.mod(Dt + Dn) <= Cc
            bits.append(int(sum(1 << j for j in range(K) if cov[j])))
            D0, Dt, n = D0 + model_t(t), Dt + Dn, n + 1
            if Dt.max() >= 1:
                failed = True
                break
        return dict(intervals=n, failed=failed, harvest=float(Dt.max()), bits=bits)

    def run_static(age):
        Dt, failed, n = np.zeros(nc), False, 0
        for t in range(T_MAX if age is None else max(0, min(int(age), T_MAX))):
            Dt, n = Dt + true_t(t), n + 1
            if Dt.max() >= 1:
                failed = True
                break
        return dict(intervals=n, failed=failed, harvest=float(Dt.max()))

    out = {f"cert|{m}": run_cert(m) for m in ("paired", "paired stale", "union", "mc")}
    out["none|tbr60"] = run_static(cfg["t_plan"])
    out["none|rtf"] = run_static(None)
    for k, a in ages.items():
        out[f"none|{k}"] = run_static(a)
        # replanning acts only on legs still in service at the change
        out[f"replan|{k}"] = run_static(a if a <= t_sw else max(t_sw, replanned[k]))
    # fourth round (20/10/2026): the change is detected only after a delay; the ages are then recomputed from the
    # margined damage accrued up to detection, part of it under the second strategy (own stream per delay, so the runs
    # above are unchanged); a leg replaced by its static age before detection is not replanned
    delayed = {}
    for dl in cfg.get("delays", ()):
        t_d = min(t_sw + dl, T_MAX)
        D0_d = sum((model_t(t) for t in range(t_d)), np.zeros(nc))
        rng_d = np.random.default_rng([cfg["seed"], r, 97, dl])
        Sd_d = stats.gumbel_l.rvs(loc, scale, size=(nmc, K), random_state=rng_d)
        Vr_d = VB2[rng_d.integers(0, n2, nmc)]
        L_d, alive_d = np.full(nmc, np.inf), np.ones(nmc, bool)
        for j, g in enumerate(groups):
            rem = np.exp(Sd_d[:, j])[:, None] - D0_d[g][None, :]
            alive_d &= np.all(rem > 0, axis=1)
            L_d = np.minimum(L_d, np.min(rem / Vr_d[:, g], axis=1))
        L_d = L_d[alive_d]
        R_d = min(g5.reserve("paired", S_all, VB2[:, g], D0_d[g], alpha)[1] for g in groups)
        rp = dict(cert_age=t_d + int(math.floor(R_d)),
                  mc_age=t_d + (int(math.floor(np.quantile(L_d, alpha))) if L_d.size else 0),
                  mc_opt10=t_d + (best_age(L_d, 10) if L_d.size else 0),
                  mc_opt50=t_d + (best_age(L_d, 50) if L_d.size else 0))
        delayed[str(dl)] = rp
        for k, a in ages.items():
            out[f"replan{dl}|{k}"] = run_static(a if a <= t_d else max(t_d, rp[k]))
    res = dict(rep=r, world=world_params(world, b_full), t_sw=t_sw, ages=ages, replanned=replanned,
               field=dict(S=flist(S_new), kappa=flist(kap), s_th=flist(s_th)), runs=out)
    if delayed:
        res["replanned_delayed"] = delayed
    return res


# ------------------------------------------------------------------------------------------------------------- F
def rep_F(args):
    """Fleets (editor re-review, 19/10/2026): n_legs field legs share one world, one bench and one set of recorded
    intervals, as the legs of a fleet calibrated on one lot would. Replacement-only certificates (paired, union, and PAC
    with eta) and static ages set at commissioning, per leg, without derating; the share of failed legs per fleet shows
    how far the outcome of one bench can differ from the average over benches."""
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    sc = scenario(cfg["scen"])
    pool, b_full = _W["pool"], _W["b_full"]
    K, groups, nc = pool.K, pool.groups, len(pool.chips)
    alpha, n1, n2, eta = cfg["alpha"], cfg["n1"], cfg["n2"], cfg["eta"]
    world = tt.World.draw(rng, b_full, sc)
    bn, fi, si, fitd, X, S_all = bench_scores(world, rng, 3 * n1)
    perm = rng.permutation(len(si))
    S_pol, S_cer = S_all[perm[:len(si) // 2]], S_all[perm[len(si) // 2:]]
    J1 = [pool.sample(rng) for _ in range(n2)]
    J2 = [pool.sample(rng) for _ in range(n2)]
    V1 = np.array([pool.model(q, fitd, BASE) for q in J1])
    V2 = np.array([pool.model(q, fitd, BASE) for q in J2])
    rng_p = np.random.default_rng([cfg["seed"], r, 99])
    R_cert = min(g5.reserve("paired", S_all, V2[:, g], np.zeros(len(g)), alpha)[1] for g in groups)
    L_mc = mc_leg_life(S_all, V2, groups, rng_p, 4000)
    ages = dict(tbr60=cfg["t_plan"], cert_age=int(math.floor(R_cert)),
                mc_age=int(math.floor(np.quantile(L_mc, alpha))), mc_opt10=best_age(L_mc, 10))

    def certs(meth, V, D0, S_sc):
        out = []
        m = min(len(S_sc), len(V))
        for g in groups:
            if meth == "union":
                out.append(cc.u5_union(S_all, V[:, g], D0[g], alpha))
            elif meth == "pac":
                out.append(cc.u9_pac(S_sc[:m], V[:m][:, g], D0[g], alpha, eta))
            else:
                out.append(cc.u6_paired(S_sc[:m], V[:m][:, g], D0[g], alpha))
        return np.array(out)

    res = {k: dict(failed=0, intervals=0, covered=0, decisions=0) for k in ("paired", "union", "pac", *ages)}
    for _ in range(cfg["n_legs"]):
        S_new, kap, s_th = field_leg(world, sc, rng, K)
        life = [pool.sample(rng) for _ in range(T_MAX)]
        cache_V, cache_D = {}, {}

        def model_t(t):
            if t not in cache_V:
                cache_V[t] = pool.model(life[t], fitd, BASE)
            return cache_V[t]

        def true_t(t):
            if t not in cache_D:
                cache_D[t] = pool.true(life[t], world, S_new, kap, s_th)
            return cache_D[t]

        for meth in ("paired", "union", "pac"):
            D0, Dt = np.zeros(nc), np.zeros(nc)
            for t in range(T_MAX):
                if np.max(certs(meth, V1, D0, S_pol)) > 1:
                    break
                Cc = certs(meth, V2, D0, S_cer)
                Dn = true_t(t)
                cov = pool.mod(Dt + Dn) <= Cc
                res[meth]["covered"] += int(np.sum(cov))
                res[meth]["decisions"] += K
                D0, Dt = D0 + model_t(t), Dt + Dn
                res[meth]["intervals"] += 1
                if Dt.max() >= 1:
                    res[meth]["failed"] += 1
                    break
        for k, a in ages.items():
            Dt = np.zeros(nc)
            for t in range(max(0, min(int(a), T_MAX))):
                Dt = Dt + true_t(t)
                res[k]["intervals"] += 1
                if Dt.max() >= 1:
                    res[k]["failed"] += 1
                    break
    return dict(rep=r, world=world_params(world, b_full), ages=ages, n_legs=cfg["n_legs"],
                pac_rank=pac_rank(min(len(S_cer), n2), alpha, eta), runs=res)


def rep_W(args):
    """W (19/10/2026, referee request): coverage of the statistical part of the margin alone, for the first-order
    ('delta', the case study), Monte Carlo ('mc') and Scheffe ellipsoid bounds. For each bench size and several coming
    intervals: the damage under the true coefficients, relative to a tested condition drawn from the bench design as
    in A (shift_ref), against the margined damage of each bound (max over the chips of a module)."""
    cfg, r = args
    _check()
    rng = np.random.default_rng([cfg["seed"], r])
    sc = scenario(cfg["scen"])
    pool, b_full = _W["pool"], _W["b_full"]
    K, groups = pool.K, pool.groups
    world = tt.World.draw(rng, b_full, sc)
    out = {}
    for n1 in cfg["n1"]:
        bn, fi, si, fitd, X, S = bench_scores(world, rng, 2 * n1)
        f_true = dict(b=world.b, cov=fitd["cov"], refs=fitd["refs"], dof=fitd["dof"])
        recs = []
        for _ in range(cfg["n_int"]):
            q = pool.sample(rng)
            shift = X[rng.integers(0, len(bn["dT"]), K)] @ (world.b - fitd["b"])
            V_rel0 = pool.model(q, f_true, replace(BASE, stat="none"))
            Vs = {st: pool.model(q, fitd, replace(BASE, stat=st)) for st in cfg["stats"]}
            for j, g in enumerate(groups):
                V_rel = V_rel0[g] * math.exp(shift[j])
                recs.append({st: [bool(np.all(V_rel <= Vs[st][g] * (1 + 1e-9))), float(Vs[st][g].max())]
                             for st in cfg["stats"]})
        out[str(n1)] = recs
    return dict(rep=r, n1=out)


REP = {"A": rep_A, "C": rep_C, "D": rep_D, "E": rep_E, "N": rep_N, "X": rep_X, "T": rep_T, "L": rep_L, "Y": rep_Y,
       "R": rep_R, "U": rep_U, "M": rep_M, "F": rep_F, "W": rep_W}


def run(tag, workers):
    cfg = CONFIGS[tag]
    out = HERE / f"g6_{tag}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["rep"])
    todo = [r for r in range(cfg["reps"]) if r not in done]
    if not todo:
        print(f"{tag}: {len(done)} done", flush=True)
        return
    if free_virtual_gb() is not None:
        # memory per worker: six cycle pools of four chips for C (about 1.6 GB at its peak), four for R, one otherwise;
        # the machine is shared, so wait for room and lower the number of workers if needed
        per = {"C": 1.6, "R": 1.1, "M": 0.9}.get(cfg["exp"], 0.6) * (2 if cfg.get("truth_db") else 1)
        free = wait_memory(per + 0.7, tag)
        cap = max(1, int((free - 0.7) / per))
        if cap < workers:
            print(f"{tag}: {free:.1f} GB of free virtual memory, {workers} -> {cap} workers", flush=True)
            workers = cap
    print(f"{tag}: {len(done)} done, {len(todo)} to run with {workers} workers, {time.strftime('%d/%m %H:%M')}",
          flush=True)
    t0 = time.time()
    # ProcessPoolExecutor, not multiprocessing.Pool: a worker that dies (at start or in a task) breaks the executor
    # and stops the run with an error, where Pool restarts dead workers without end (seen on 16/10/2026: about 190
    # leaked handles per second and no progress)
    with ProcessPoolExecutor(workers, initializer=_init,
                             initargs=(cfg["exp"], cfg.get("strategy", "cruise"), cfg.get("s_up", g5.S_UP),
                                       cfg.get("truth_db"))) as ex, \
            open(out, "a", encoding="utf-8", newline="\n") as f:
        futs = [ex.submit(REP[cfg["exp"]], (cfg, r)) for r in todo]
        for k, fu in enumerate(as_completed(futs)):
            f.write(json.dumps(fu.result()) + "\n")
            f.flush()
            if (k + 1) % 10 == 0 or k + 1 == len(todo):
                print(f"  {tag}: {k + 1}/{len(todo)}, {time.time() - t0:.0f} s", flush=True)


def deterministic_D(n_splits=200, n_int=200, seed=640):
    """Margined damage of an interval and certified life of a new leg on the real Hernes data, per margin variant: the
    interval damage of the critical module (the larger one) and the reserve of the leg (the smaller one)."""
    rng = np.random.default_rng(seed)
    h = tt.hernes()
    X = tt.features(h["dT"], h["ton"], h["I"])
    pool_base = g5.DayPool("chu_ky_cruise_d00_base.npz", BASE)
    pool_short = g5.DayPool("chu_ky_cruise_d00_base.npz", SHORT["short pulses counted at 0.15 s"])
    days = [pool_base.sample(rng) for _ in range(n_int)]
    splits = [tt.split(h["run"], "half", rng) for _ in range(n_splits)]
    fits = [(tt.fit(h["dT"][fi], h["ton"][fi], h["I"][fi], h["Nf"][fi], h["run"][fi]), si) for fi, si in splits]
    f_full = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])
    out = {}
    for name, mg in list(VARIANTS.items()) + list(SHORT.items()):
        pool = pool_short if mg.short_rule != "exclude" else pool_base
        t0 = time.time()
        V_full = np.array([pool.mod(model_raw(pool, q, f_full, mg)) for q in days])         # [n_int, K]
        life, v_half = [], []
        for fitd, si in fits[:40]:
            S = np.log(h["Nf"][si]) - X[si] @ fitd["b"]
            V = np.array([model_raw(pool, q, fitd, mg) for q in days[:50]])
            life.append(min(g5.reserve("paired", S, V[:, g], np.zeros(len(g)), 0.1)[1] for g in pool.groups))
            v_half.append(float(np.median(pool.mod(V).max(1))))
        out[name] = dict(V_interval_median=float(np.median(V_full.max(1))),
                         V_interval_q90=float(np.quantile(V_full.max(1), 0.9)),
                         V_interval_median_modules=[float(np.median(V_full[:, j])) for j in range(pool.K)],
                         # the same with the half splits of the certified life (median over the 40 splits of the
                         # median interval), the basis of the single cycles of Table 5b and of the worked example
                         V_interval_median_half=float(np.median(v_half)),
                         certified_life_intervals_median=float(np.median(life)),
                         certified_life_intervals_q10_q90=[float(np.quantile(life, q)) for q in (0.1, 0.9)])
        print(f"{name:30s} V {out[name]['V_interval_median']:.4f}, certified life {np.median(life):6.1f} intervals "
              f"({time.time() - t0:.0f} s)", flush=True)
    (HERE / "g6_D_det.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "D_det":
        wait_memory(2.0, "D_det")
        deterministic_D()
    elif a[1] == "wait_memory":
        wait_memory(float(a[2]), a[3] if len(a) > 3 else "wait")
    else:
        run(a[1], int(a[2]) if len(a) > 2 else 4)
