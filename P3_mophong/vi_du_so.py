"""P3 worked example (readability package of 17/10/2026; per module since 18/10/2026): the paired certificate and its
reserve for a new leg on the real bench data of Hernes et al. (2021) and the cruising mission, step by step, with the
certificates of the real data over random splits that Section 6.3 quotes.

50 recorded 30-day intervals of the cruising mission are drawn once; 200 random splits of the 34-module design are
drawn in each of two schemes, 'half' (each run split in half: 16 modules fit the model, 18 give scores) and 'two' (two
modules of each run fit, 24 scores). For every split: the paired certificate of each module (outer, inner) of a new leg
for the coming interval at alpha = 0.10, the certificate of the leg (the larger one, of the critical module), and, in
the 'two' scheme, the union certificate (it needs at least 19 scores). The split with the median leg certificate of the
'half' scheme is reported in detail: fitted exponents, the scores, the margined damage of the paired intervals for the
critical module and its build-up from the bare model (upper thermal scale, stated margin, statistical part), the paired
scores, the certificate and the reserve, and the reserve of the leg at the derating levels 10-50 % (cycle databases
chu_ky_cruise_d10-d50_base.npz, the same vehicle days), with the choice of the reserve rule for a planned replacement
after 60 intervals. Reserves are reported as computed, not rounded up.

Run: python -I vi_du_so.py  -> vi_du_so.json
"""
import gc
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import thu_nghiem_g6 as g6  # noqa: E402  (also disables the WMI query of scipy on Python 3.12)
import numpy as np  # noqa: E402

import chung_chi as cc  # noqa: E402
import thu_nghiem_g5 as g5  # noqa: E402
import tuoi_tho as tt  # noqa: E402

ALPHA, N2, N_SPLITS, T_PLAN, SEED = 0.10, 50, 200, 60, 20261018
BASE = g6.BASE
BARE = replace(BASE, gamma=0.0, gamma_t=0.0, Ea=0.0, kappa_M=1.0, stat="none")
STATED = replace(BASE, stat="none")

g6.wait_memory(1.5, "vi_du_so")
h = tt.hernes()
X = tt.features(h["dT"], h["ton"], h["I"])
pool = g5.DayPool("chu_ky_cruise_d00_base.npz", BASE)
groups = pool.groups
rng = np.random.default_rng(SEED)
rec = [pool.sample(rng) for _ in range(N2)]
splits = {"half": [], "two": []}
for _ in range(N_SPLITS):
    for scheme in ("half", "two"):
        fi, si = tt.split(h["run"], scheme, rng)
        fitd = tt.fit(h["dT"][fi], h["ton"][fi], h["I"][fi], h["Nf"][fi], h["run"][fi])
        S = np.log(h["Nf"][si]) - X[si] @ fitd["b"]
        m = min(len(S), N2)
        V = np.array([pool.model(q, fitd, BASE) for q in rec])          # all recorded intervals, for the union
        Cp = [cc.u6_paired(S[:m], V[:m][:, g], np.zeros(len(g)), ALPHA) for g in groups]
        Cu = [cc.u5_union(S, V[:, g], np.zeros(len(g)), ALPHA) for g in groups] if scheme == "two" else None
        splits[scheme].append(dict(Cp=Cp, Cu=Cu, fi=fi, si=si, fitd=fitd, S=S, V=V[:m]))
real = {}
for scheme, ss in splits.items():
    leg = np.array([max(x["Cp"]) for x in ss])
    mods = np.array([x["Cp"] for x in ss])
    real[scheme] = dict(n_scores=int(len(ss[0]["S"])), leg_Cp_median=float(np.median(leg)),
                        leg_Cp_q10_q90=[float(np.quantile(leg, q)) for q in (0.1, 0.9)],
                        module_Cp_median=[float(np.median(mods[:, j])) for j in range(len(groups))],
                        critical_share=[float(np.mean(mods.argmax(1) == j)) for j in range(len(groups))])
    if scheme == "two":
        cu = np.array([max(x["Cu"]) for x in ss])
        real[scheme].update(leg_Cu_finite=float(np.mean(np.isfinite(cu))),
                            leg_Cu_median=float(np.median(cu)) if np.isfinite(cu).any() else None)
ss = splits["half"]
order = np.argsort([max(x["Cp"]) for x in ss])
x = ss[order[N_SPLITS // 2]]
fi, si, fitd, S, V = x["fi"], x["si"], x["fitd"], x["S"], x["V"]
m = len(S)
K = math.ceil((1 - ALPHA) * (m + 1) - 1e-12)
jc = int(np.argmax(x["Cp"]))                                  # critical module of this split
g = groups[jc]
Vmax = V[:, g].max(1)
U = np.exp(-S) * Vmax
R_mod = [g5.reserve("paired", S, V[:, gg], np.zeros(len(gg)), ALPHA)[1] for gg in groups]
# build-up of the margined damage of the paired intervals (critical module, largest chip)
V_bare = np.array([g6.model_raw(pool, q, fitd, BARE, s=1.0)[g].max() for q in rec[:m]])
V_th = np.array([g6.model_raw(pool, q, fitd, BARE, s=g5.S_UP)[g].max() for q in rec[:m]])
V_st = np.array([g6.model_raw(pool, q, fitd, STATED, s=g5.S_UP)[g].max() for q in rec[:m]])
se = np.sqrt(np.diag(fitd["cov"]))
out = dict(
    alpha=ALPHA, n_recorded=N2, n_splits=N_SPLITS, t_plan=T_PLAN, seed=SEED, real_data=real,
    n_fit=int(len(fi)), n_scores=int(m), rank_K=int(K), module_names=list(g5.MODULE_NAMES),
    fit=dict(b=[float(v) for v in fitd["b"]], se=[float(v) for v in se], swing_exponent=float(-fitd["b"][1])),
    scores=dict(sorted=[float(v) for v in np.sort(S)], life_factor_min_max=[float(np.exp(S.min())),
                                                                           float(np.exp(S.max()))]),
    certificates_modules=[float(v) for v in x["Cp"]], critical_module=g5.MODULE_NAMES[jc],
    paired_interval_damage=dict(median=float(np.median(Vmax)), min=float(Vmax.min()), max=float(Vmax.max())),
    build_up_median=dict(bare=float(np.median(V_bare)), thermal_factor=float(np.median(V_th / V_bare)),
                         stated_margin_factor=float(np.median(V_st / V_th)),
                         statistical_factor=float(np.median(Vmax / V_st)),
                         total_factor=float(np.median(Vmax / V_bare))),
    paired_scores_sorted=[float(v) for v in np.sort(U)],
    certificate=float(max(x["Cp"])), pair_of_certificate=int(np.argmax(U)),
    S_of_that_pair=float(S[np.argmax(U)]), V_of_that_pair=float(Vmax[np.argmax(U)]),
    reserve_modules=[float(v) for v in R_mod], reserve_intervals=float(min(R_mod)),
    levels={})
late = json.loads((HERE / "giam_tai_kha_thi.json").read_text(encoding="utf-8"))
del pool, splits, ss
gc.collect()
for d in g5.LEVELS:
    pd = g5.DayPool(f"chu_ky_cruise_d{int(round(100 * d)):02d}_base.npz", BASE)
    Vd = np.array([pd.model(q, fitd, BASE) for q in rec[:m]])
    res = [g5.reserve("paired", S, Vd[:, gg], np.zeros(len(gg)), ALPHA) for gg in groups]
    out["levels"][f"{d:.1f}"] = dict(certificate=float(max(r[0] for r in res)),
                                     reserve_intervals=float(min(r[1] for r in res)),
                                     reserve_modules=[float(r[1]) for r in res],
                                     late_share=float(1 - late[f"{d:.1f}"]["on_time_share"]))
    v = out["levels"][f"{d:.1f}"]
    print(f"level {d:.1f}: C_P {v['certificate']:.4f}, R_c {v['reserve_intervals']:.1f} intervals "
          f"(modules {v['reserve_modules'][0]:.1f}, {v['reserve_modules'][1]:.1f}), late {v['late_share']:.3f}",
          flush=True)
    del pd
    gc.collect()
ok = [d for d, v in out["levels"].items() if v["reserve_intervals"] >= T_PLAN]
out["reserve_rule_choice"] = ok[0] if ok else max(out["levels"], key=lambda d: out["levels"][d]["reserve_intervals"])
out["reserve_rule_reaches_plan"] = bool(ok)
(HERE / "vi_du_so.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({k: v for k, v in out.items() if k not in ("scores", "paired_scores_sorted")}, indent=1))
