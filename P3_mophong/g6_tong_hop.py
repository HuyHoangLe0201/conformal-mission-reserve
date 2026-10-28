"""P3 case study (G6): tables and draft figures from the G6 runs (g6_*.jsonl, g6_D_det.json).

Per module (3.3 kV converter, 18/10/2026; the 1.7 kV summaries are kept in luu_1700V_20261018/): a static run holds
one record per module of the leg (outer, inner) under 'modules'. The tables pool the records of both modules, two per
repetition, which share the world and the bench of their repetition; Wilson intervals on pooled records are therefore
somewhat narrow, and the stress tests with many field modules per repetition (L, Y) give standard errors over
repetitions. Closed-loop runs (C) hold one record per leg life and run (policy|method).

Tables (markdown, printed and written to g6_tong_hop.md; numbers also in g6_tong_hop.json):
  A    coverage of the true damage and of the calibrated part, median ratio to the oracle and finite share of U1-U10 and
       of the anchored U6, per truth, bench size and level (pooled over the modules, with the standard error over
       repetitions, whose two module records share a world and a bench; per module in 'per_module');
  C    closed loop per truth and driving strategy, and the drift runs: served intervals, failures (Wilson 95 %), true
       damage at replacement, coverage per decision pooled over decisions and modules, by decision index and per life,
       derated and false-derating shares, mean level, late sections, collapse, static ages, and the cost per 100
       intervals of the renewal-reward argument (Barlow and Hunter 1960): a planned replacement costs 1, a failure in
       service r and an interval with every section late c_L, so the long-run cost per interval is
       sum(1 + (r - 1) failed + c_L late_sum) / sum(intervals) over the lives; differences of cost and service against
       the paired certificate under the reserve rule with paired bootstrap intervals over the lives;
  F4   coverage per life under increasing policy feedback (truth at the margins, both strategies);
  AGE  age replacement at fixed ages from the run-to-failure lives (true lives known, ex post);
  D, E, N, X, T, L, Y as before, and R: re-simulation necessity under a balancing pattern.
Figures: g6_F2_nhap (coverage and ratio against n1), g6_F4F5_nhap (coverage per life under feedback; served intervals
against damage at replacement), g6_vuot_bien_nhap (exceedance of the heating-time margin), as .pdf/.png.

Run: python g6_tong_hop.py
"""
import json
import math
import site
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

CERTS = ("U1 box", "U2 Monte Carlo", "U3 Bayes", "U5 union", "U6 paired", "U9 PAC")
TRUTHS = ("evidence", "boundary", "adverse_strict", "beyond")
C_TAGS = tuple(f"C_{sc}_{st}" for sc in TRUTHS for st in ("cruise", "coast")) + \
    ("C_drift", "C_drift_allow", "C_drift04", "C_drift04_allow")
COST_RATIOS = (10, 50)
LATE_COSTS = (0.0, 0.5, 2.0)
INDEX_BINS = ((0, 1), (1, 10), (10, 30), (30, 60), (60, 240))
REF_RUN = "reserve|paired"
F4_RUNS = ("pass|union", "pass|paired", "fixed20|union", "fixed20|paired", "reserve|union", "reserve|paired",
           "reserve|split0", "reserve|reuse", "reserve|aci")
K_MOD = 2
OUT, MD = {}, []


def load(tag):
    """Rows of g6_<tag>.jsonl, one per repetition (a repetition written twice by overlapping runs is kept once)."""
    f = HERE / f"g6_{tag}.jsonl"
    if not f.exists():
        return []
    rows = {}
    for x in f.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            rows.setdefault(r["rep"], r)
    return [rows[k] for k in sorted(rows)]


def mrows(tag):
    """Per-module records of a static run: the record of each module with its repetition and module index."""
    out = []
    for r in load(tag):
        for j, m in enumerate(r["modules"]):
            out.append(dict(m, rep=r["rep"], module=j))
    return out


def p(line=""):
    MD.append(line)
    print(line)


def binom_ci(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def rep_se(rows, value):
    """Mean of a per-record value and its standard error over repetitions (records of a repetition averaged first)."""
    by = {}
    for r in rows:
        by.setdefault(r["rep"], []).append(value(r))
    m = np.array([np.mean(v) for v in by.values()])
    return float(m.mean()), float(m.std(ddof=1) / math.sqrt(len(m))) if len(m) > 1 else math.nan


def f_(v, d=3):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"


def table_A(tag):
    rows = mrows(tag)
    if not rows:
        return None
    alphas = list(rows[0]["oracle"])
    n1s = list(rows[0]["n1"])
    res = {}
    nrep = len({r["rep"] for r in rows})
    p(f"\n### A, {tag}: {nrep} repetitions x {K_MOD} modules\n")
    p("| n1 | alpha | certificate | coverage (95 % CI) | outer / inner | calibrated part | ratio to oracle, median | "
      "finite |")
    p("|---|---|---|---|---|---|---|---|")
    for n1 in n1s:
        for a in alphas:
            D = np.array([r["D_star"] for r in rows])
            O = np.array([r["oracle"][a] for r in rows])
            U = np.array([r["n1"][n1]["U_new"] for r in rows])
            mod = np.array([r["module"] for r in rows])
            names = list(CERTS) + (["U6 anchored"] if "anchored" in rows[0]["n1"][n1] else []) + ["U10 oracle"]
            for c in names:
                if c == "U10 oracle":
                    C = O
                elif c == "U6 anchored":
                    C = np.array([r["n1"][n1]["anchored"][a] for r in rows])
                else:
                    C = np.array([r["n1"][n1]["certs"][a][c] for r in rows])
                fin = np.isfinite(C)
                k = int(np.sum(D <= C))
                lo, hi = binom_ci(k, len(D))
                ratio = float(np.median(C[fin] / O[fin])) if fin.any() else None
                cal = float(np.mean(U <= C)) if c not in ("U10 oracle", "U6 anchored") else None
                per = [float(np.mean(D[mod == j] <= C[mod == j])) for j in range(K_MOD)]
                hit = D <= C
                _, se = rep_se([dict(rep=r["rep"], h=float(x)) for r, x in zip(rows, hit)], lambda q: q["h"])
                res[f"{n1}|{a}|{c}"] = dict(coverage=k / len(D), ci=[lo, hi], se_rep=se, calibrated=cal, ratio=ratio,
                                            finite=float(fin.mean()), per_module=per, n=len(D))
                p(f"| {n1} | {a} | {c} | {k / len(D):.3f} ({lo:.3f}-{hi:.3f}) | {per[0]:.3f} / {per[1]:.3f} | "
                  f"{f_(cal)} | {'∞' if ratio is None else f'{ratio:.2f}'} | {fin.mean():.2f} |")
        st = np.mean([r["n1"][n1]["stat_ok"] for r in rows])
        res[f"{n1}|stat_ok"] = float(st)
        p(f"| {n1} | | statistical part alone | {st:.3f} | | | | |")
    OUT[tag] = res
    return res


def life_metrics(x):
    """Metrics of a list of life records of one run."""
    T = np.array([v["intervals"] for v in x], float)
    F = np.array([v["failed"] for v in x], bool)
    k = int(F.sum())
    lo, hi = binom_ci(k, len(x))
    late_sum = np.array([v["late_sum"] for v in x], float)
    out = dict(lives=len(x), intervals_mean=float(T.mean()), intervals_median=float(np.median(T)),
               failures=k, failed=k / len(x), failed_ci=[lo, hi], late_pooled=float(late_sum.sum() / max(T.sum(), 1)),
               collapse=float(np.mean([v["collapse"] for v in x])),
               harvest_ok=(float(np.mean([v["harvest"] for v in x if not v["failed"]]))
                           if any(not v["failed"] for v in x) else None),
               harvest_ok_median=(float(np.median([v["harvest"] for v in x if not v["failed"]]))
                                  if any(not v["failed"] for v in x) else None),
               unplanned_per_1000=float(1000 * k / max(T.sum(), 1)))
    g = lambda key: [v[key] for v in x if v.get(key) is not None]
    out["derated"] = float(np.mean(g("derated_share"))) if g("derated_share") else None
    out["false_of_derated"] = float(np.mean(g("false_share_of_derated"))) if g("false_share_of_derated") else None
    out["mean_level"] = float(np.mean(g("mean_level"))) if g("mean_level") else None
    out["aci"] = float(np.mean(g("aci_level"))) if g("aci_level") else None
    ages = g("age")
    out["age_mean"] = float(np.mean(ages)) if ages else None
    out["age_q10_q90"] = [float(np.quantile(ages, q)) for q in (0.1, 0.9)] if ages else None
    bits = [v["bits"] for v in x if v.get("bits")]
    if bits:
        dec = [np.array([[(b >> j) & 1 for j in range(K_MOD)] for b in bl], float) for bl in bits]
        allc = np.concatenate(dec)
        out["cov_pooled"] = float(allc.mean())
        out["cov_pooled_modules"] = [float(v) for v in allc.mean(0)]
        out["cov_all_pooled"] = float(allc.min(1).mean())
        life_ok = np.array([d.min() for d in dec])
        kk = int(life_ok.sum())
        out["cov_life"] = kk / len(dec)
        out["cov_life_ci"] = list(binom_ci(kk, len(dec)))
        # coverage by decision index (pooled over the lives that reach it and the modules)
        idx = {}
        for lo_, hi_ in INDEX_BINS:
            sel = [d[lo_:hi_] for d in dec if len(d) > lo_]
            if sel:
                s = np.concatenate(sel)
                idx[f"{lo_ + 1}-{hi_}"] = dict(coverage=float(s.mean()), decisions=int(s.shape[0]),
                                               lives=len(sel))
        out["cov_by_index"] = idx
        # coverage at each decision index, over the lives that reach it: the smallest value over the indices with at
        # least 100 lives
        per_t = []
        for t in range(max(len(d) for d in dec)):
            s = [d[t] for d in dec if len(d) > t]
            if len(s) >= 100:
                per_t.append(float(np.mean(s)))
        out["cov_index_min"] = min(per_t) if per_t else None
    else:
        out.update(cov_pooled=None, cov_pooled_modules=None, cov_all_pooled=None, cov_life=None, cov_life_ci=None,
                   cov_by_index=None, cov_index_min=None)
    out["cost_per_100"] = {f"{r}|{cl}": float(100 * np.sum(1 + (r - 1) * F + cl * late_sum) / max(T.sum(), 1))
                           for r in COST_RATIOS for cl in LATE_COSTS}
    return out


def cost_arrays(x):
    return (np.array([v["intervals"] for v in x], float), np.array([v["failed"] for v in x], float),
            np.array([v["late_sum"] for v in x], float))


def paired_diff(xa, xb, n_boot=2000, seed=1):
    """Differences b - a over common lives (paired): mean served intervals and cost per 100 intervals for every cost
    ratio and lateness cost, with bootstrap 95 % intervals over the lives."""
    Ta, Fa, La = cost_arrays(xa)
    Tb, Fb, Lb = cost_arrays(xb)
    n = len(Ta)
    rng = np.random.default_rng(seed)
    B = rng.integers(0, n, (n_boot, n))

    def cost(T, F, L, r, cl, idx):
        return 100 * (1 + (r - 1) * F[idx] + cl * L[idx]).sum(-1) / np.maximum(T[idx].sum(-1), 1)

    out = dict(intervals=float((Tb - Ta).mean()),
               intervals_ci=[float(np.quantile((Tb[B] - Ta[B]).mean(1), q)) for q in (0.025, 0.975)])
    full = np.arange(n)
    for r in COST_RATIOS:
        for cl in LATE_COSTS:
            d0 = float(cost(Tb, Fb, Lb, r, cl, full) - cost(Ta, Fa, La, r, cl, full))
            db = cost(Tb, Fb, Lb, r, cl, B) - cost(Ta, Fa, La, r, cl, B)
            out[f"cost|{r}|{cl}"] = dict(diff=d0, ci=[float(np.quantile(db, 0.025)), float(np.quantile(db, 0.975))])
    return out


def table_C():
    res, diffs, meta = {}, {}, {}
    p(f"\n### C, closed loop over leg lives (cost per 100 intervals; a planned replacement 1, a failure r, an interval "
      f"with all sections late c_L)\n")
    p("| group | run | lives | served, mean | failures (Wilson 95 %) | coverage per decision, pooled | outer / inner | "
      "lowest at an index | lives covered | derated | false (of derated) | mean level | late sections | collapse | "
      "damage at replacement | " + " | ".join(f"cost r={r} c_L={cl}" for r in COST_RATIOS for cl in LATE_COSTS) + " |")
    p("|---" * (15 + len(COST_RATIOS) * len(LATE_COSTS)) + "|")
    for tag in C_TAGS:
        rows = load(tag)
        if not rows:
            continue
        grp = tag[2:]
        runs = list(rows[0]["runs"])
        for run in runs:
            x = [r["runs"][run] for r in rows]
            m = life_metrics(x)
            res[f"{grp}|{run}"] = m
            cm = m["cov_pooled_modules"]
            p(f"| {grp} | {run} | {m['lives']} | {m['intervals_mean']:.1f} | {m['failures']} ({m['failed_ci'][0]:.3f}-"
              f"{m['failed_ci'][1]:.3f}) | {f_(m['cov_pooled'])} | "
              f"{'' if cm is None else f'{cm[0]:.3f} / {cm[1]:.3f}'} | {f_(m['cov_index_min'])} | {f_(m['cov_life'])} | "
              f"{f_(m['derated'], 2)} | {f_(m['false_of_derated'], 2)} | {f_(m['mean_level'], 2)} | "
              f"{100 * m['late_pooled']:.2f} % | {m['collapse']:.2f} | {f_(m['harvest_ok'])} | " +
              " | ".join(f"{m['cost_per_100'][f'{r}|{cl}']:.2f}" for r in COST_RATIOS for cl in LATE_COSTS) + " |")
        if REF_RUN in runs:
            xa = [r["runs"][REF_RUN] for r in rows]
            for run in runs:
                if run != REF_RUN:
                    diffs[f"{grp}|{run}"] = paired_diff(xa, [r["runs"][run] for r in rows])
        # static ages at commissioning, and the bench sizes behind the policies (kept apart from the runs of 'C')
        ages = {k: [r["ages"][k] for r in rows] for k in rows[0]["ages"]}
        meta[grp] = dict(ages={k: dict(mean=float(np.mean(v)), median=float(np.median(v)),
                                       q10_q90=[float(np.quantile(v, q)) for q in (0.1, 0.9)]) for k, v in ages.items()},
                         n_scores=rows[0]["n_scores"], lives=len(rows))
    OUT["C"] = res
    OUT["C_meta"] = meta
    OUT["C_diff"] = diffs
    p(f"\n### C, differences against {REF_RUN} on common lives (run minus {REF_RUN}; bootstrap 95 % over lives)\n")
    p("| group | run | served intervals | " + " | ".join(f"cost r={r} c_L={cl}" for r in COST_RATIOS for cl in LATE_COSTS)
      + " |")
    p("|---" * (3 + len(COST_RATIOS) * len(LATE_COSTS)) + "|")
    for k, v in diffs.items():
        grp, run = k.split("|", 1)
        p(f"| {grp} | {run} | {v['intervals']:+.1f} ({v['intervals_ci'][0]:+.1f}, {v['intervals_ci'][1]:+.1f}) | " +
          " | ".join(f"{v[f'cost|{r}|{cl}']['diff']:+.2f} ({v[f'cost|{r}|{cl}']['ci'][0]:+.2f}, "
                     f"{v[f'cost|{r}|{cl}']['ci'][1]:+.2f})" for r in COST_RATIOS for cl in LATE_COSTS) + " |")
    return res


def table_F4():
    """Coverage per leg life under increasing policy feedback (truth at the margins, 300 lives per run)."""
    res = {}
    for st in ("cruise", "coast"):
        rows = load(f"C_boundary_{st}")
        if not rows:
            continue
        for run in F4_RUNS:
            if run not in rows[0]["runs"]:
                continue
            m = OUT["C"][f"boundary_{st}|{run}"]
            res[f"{st}|{run}"] = dict(cov_life=m["cov_life"], ci=m["cov_life_ci"], lives=m["lives"],
                                      mean_level=m["mean_level"], intervals=m["intervals_mean"], failed=m["failed"],
                                      late=m["late_pooled"], cov_pooled=m["cov_pooled"],
                                      cov_index_min=m["cov_index_min"])
    if res:
        p("\n### F4, coverage per leg life under increasing policy feedback (truth at the margins)\n")
        p("| strategy | run | lives | lives covered (95 % CI) | pooled per decision | lowest at an index | failures | "
          "mean level | served | late |")
        p("|---|---|---|---|---|---|---|---|---|---|")
        for k, v in res.items():
            st, run = k.split("|", 1)
            p(f"| {st} | {run} | {v['lives']} | {v['cov_life']:.3f} ({v['ci'][0]:.3f}-{v['ci'][1]:.3f}) | "
              f"{v['cov_pooled']:.3f} | {f_(v['cov_index_min'])} | {v['failed']:.3f} | {f_(v['mean_level'], 2)} | "
              f"{v['intervals']:.1f} | {100 * v['late']:.1f} % |")
    OUT["F4"] = res


def table_AGE():
    """Age replacement at fixed ages, from the run-to-failure lives of each group (the same damage paths: no derating
    until the age or the failure); the cheapest age needs the true life distribution, which no operator has."""
    res = {}
    p("\n### AGE, age replacement from the run-to-failure lives (failure share, mean served intervals, cost per 100 "
      "intervals)\n")
    p("| group | age (intervals) | failures | served intervals | " + " | ".join(f"cost, r = {r}" for r in COST_RATIOS)
      + " |")
    p("|---" * (4 + len(COST_RATIOS)) + "|")
    for tag in C_TAGS:
        rows = load(tag)
        if not rows or "none|rtf" not in rows[0]["runs"]:
            continue
        grp = tag[2:]
        lives = [r["runs"]["none|rtf"] for r in rows]
        Tf = np.array([v["intervals"] if v["failed"] else np.inf for v in lives])
        best = {}
        for a in range(5, 241, 5):
            F = Tf <= a
            T = np.minimum(Tf, a)
            cost = {r: float(100 * np.sum(1 + (r - 1) * F) / T.sum()) for r in COST_RATIOS}
            res[f"{grp}|age{a}"] = dict(age=a, failed=float(F.mean()), intervals=float(T.mean()),
                                        cost_per_100={str(r): c for r, c in cost.items()})
            for r in COST_RATIOS:
                if r not in best or cost[r] < best[r][1]:
                    best[r] = (a, cost[r], float(F.mean()))
            if a in (30, 40, 60, 90, 120, 150, 240):
                p(f"| {grp} | {a} | {F.mean():.3f} | {T.mean():.1f} | " + " | ".join(f"{cost[r]:.2f}"
                                                                                    for r in COST_RATIOS) + " |")
        res[f"{grp}|best_age"] = {str(r): dict(age=b[0], cost=b[1], failed=b[2]) for r, b in best.items()}
        p(f"| {grp} | cheapest (true lives known) | " + "; ".join(
            f"r = {r}: age {b[0]}, cost {b[1]:.2f}, failures {b[2]:.3f}" for r, b in best.items()) + " | | " +
          " | " * (len(COST_RATIOS) - 1) + "|")
    OUT["AGE"] = res


def table_D():
    res = {}
    det = HERE / "g6_D_det.json"
    if det.exists():
        d = json.loads(det.read_text(encoding="utf-8"))
        p("\n### D, margin variants on the real Hernes data (half split, alpha 0.1, new leg)\n")
        p("| variant | margined damage of a 30-day interval, critical module, median (q90) | outer / inner | certified "
          "life of the leg, intervals, median (q10-q90) |")
        p("|---|---|---|---|")
        for k, v in d.items():
            q = v["certified_life_intervals_q10_q90"]
            mm = v["V_interval_median_modules"]
            p(f"| {k} | {v['V_interval_median']:.4f} ({v['V_interval_q90']:.4f}) | {mm[0]:.4f} / {mm[1]:.4f} | "
              f"{v['certified_life_intervals_median']:.1f} ({q[0]:.1f}-{q[1]:.1f}) |")
        res["det"] = d
    rows = mrows("D_cov")
    if rows:
        p(f"\n### D, coverage when the truth sits at the base margin and the certificate uses a narrower one "
          f"({len(rows)} module records, n1 40, alpha 0.1)\n")
        p("| certificate margin | coverage (95 % CI) | certificate / base certificate, median |")
        p("|---|---|---|")
        D = np.array([r["D_star"] for r in rows])
        B = np.array([r["variants"]["base"] for r in rows])
        for name in rows[0]["variants"]:
            C = np.array([r["variants"][name] for r in rows])
            k = int(np.sum(D <= C))
            lo, hi = binom_ci(k, len(D))
            res[name] = dict(coverage=k / len(D), ci=[lo, hi], ratio_to_base=float(np.median(C / B)))
            p(f"| {name} | {k / len(D):.3f} ({lo:.3f}-{hi:.3f}) | {np.median(C / B):.2f} |")
    OUT["D"] = res


def table_E():
    res = {}
    for tag in ("E_shift", "E_shift_evidence"):
        rows = mrows(tag)
        if not rows:
            continue
        p(f"\n### E, season shift, {tag} ({len(rows)} module records, n1 40, alpha 0.1)\n")
        p("| ambient shift (K) | true damage / unshifted, median | recorded damage, re-simulated / recorded | coverage, "
          "recorded intervals | coverage, re-simulated |")
        p("|---|---|---|---|---|")
        D0 = np.array([r["deltas"]["0.0"]["D_star"] for r in rows])
        for dl in rows[0]["deltas"]:
            D = np.array([r["deltas"][dl]["D_star"] for r in rows])
            Cr = np.array([r["deltas"][dl]["C_recorded"] for r in rows])
            Cs = np.array([r["deltas"][dl]["C_resim"] for r in rows])
            vr = float(np.median([r["deltas"][dl]["V_ratio"] for r in rows]))
            ok = np.isfinite(Cr) & np.isfinite(Cs)
            cr = float(np.median(Cs[ok] / Cr[ok])) if ok.any() else None
            res[f"{tag}|{dl}"] = dict(damage_ratio=float(np.median(D / D0)), V_ratio=vr, cert_ratio=cr,
                                      cov_recorded=float(np.mean(D <= Cr)), cov_resim=float(np.mean(D <= Cs)))
            p(f"| {dl} | {np.median(D / D0):.3f} | {vr:.3f} (certificate {f_(cr)}) | {np.mean(D <= Cr):.3f} | "
              f"{np.mean(D <= Cs):.3f} |")
    OUT["E"] = res


def table_R():
    """Re-simulation necessity (added 18/10/2026): coverage of the paired certificate on the recorded (base-pattern)
    intervals and re-simulated under the balancing pattern of the coming interval, per module."""
    res = {}
    for tag in ("R_boundary", "R_evidence"):
        rows = mrows(tag)
        if not rows:
            continue
        nrep = len({r["rep"] for r in rows})
        p(f"\n### R, re-simulation necessity, {tag} ({nrep} repetitions x {K_MOD} modules, n1 40, n2 100)\n")
        p("| pattern | module | alpha | true damage / base pattern, median | recorded damage, re-simulated / recorded | "
          "coverage, recorded (95 % CI) | coverage, re-simulated (95 % CI) |")
        p("|---|---|---|---|---|---|---|")
        Db = np.array([r["patterns"]["base"]["D_star"] for r in rows])
        mod = np.array([r["module"] for r in rows])
        for pat in rows[0]["patterns"]:
            D = np.array([r["patterns"][pat]["D_star"] for r in rows])
            for a in rows[0]["patterns"][pat]["recorded"]:
                Cr = np.array([r["patterns"][pat]["recorded"][a] for r in rows])
                Cs = np.array([r["patterns"][pat]["resim"][a] for r in rows])
                for j, name in ((None, "both"), (0, "outer"), (1, "inner")):
                    sel = np.ones(len(rows), bool) if j is None else mod == j
                    kr, ks = int(np.sum(D[sel] <= Cr[sel])), int(np.sum(D[sel] <= Cs[sel]))
                    n = int(sel.sum())
                    vr = float(np.median([r["patterns"][pat]["V_ratio"] for r, s in zip(rows, sel) if s]))
                    res[f"{tag}|{pat}|{name}|{a}"] = dict(
                        damage_ratio=float(np.median(D[sel] / Db[sel])), V_ratio=vr, cov_recorded=kr / n,
                        ci_recorded=list(binom_ci(kr, n)), cov_resim=ks / n, ci_resim=list(binom_ci(ks, n)), n=n)
                    v = res[f"{tag}|{pat}|{name}|{a}"]
                    p(f"| {pat} | {name} | {a} | {v['damage_ratio']:.3f} | {vr:.3f} | {kr / n:.3f} "
                      f"({v['ci_recorded'][0]:.3f}-{v['ci_recorded'][1]:.3f}) | {ks / n:.3f} "
                      f"({v['ci_resim'][0]:.3f}-{v['ci_resim'][1]:.3f}) |")
    OUT["R"] = res


def table_T():
    """Temporal dependence of the missions: coverage per interval design."""
    res = {}
    # 19/10/2026 (third review): the evidence truth pools T_evidence and its repetition T_evidence_b (another seed;
    # the first gave a calibrated part of 0.863, 2.2 standard errors below the exact level, the second 0.935), and the
    # intervals are mean +- 1.96 standard errors over repetitions, since the two modules of a repetition share its
    # world, bench and recorded intervals (Wilson intervals over module records treated them as independent)
    for tag, parts in (("T_boundary", ("T_boundary",)), ("T_evidence", ("T_evidence", "T_evidence_b"))):
        rows = [dict(x, rep=x["rep"] + 1000 * i) for i, t in enumerate(parts) for x in mrows(t)]
        if not rows:
            continue
        nrep = len({r["rep"] for r in rows})
        p(f"\n### T, temporal dependence, {tag} ({nrep} repetitions from {', '.join(parts)}, {len(rows)} module "
          f"records, n1 40, n2 100, alpha 0.1)\n")
        p("| design | coming / recorded model damage, median | paired: coverage (+-1.96 se) | paired: calibrated part "
          "(+-1.96 se) | union: coverage | union: calibrated part |")
        p("|---|---|---|---|---|---|")
        for dz in rows[0]["designs"]:
            v = [r["designs"][dz] for r in rows]
            ratio = np.array([x["V_new"] / x["V_rec_median"] for x in v])
            cov, se = rep_se(rows, lambda r: r["designs"][dz]["D_star"] <= r["designs"][dz]["paired"]["0.1"])
            cal, sec = rep_se(rows, lambda r: r["designs"][dz]["U_new"] <= r["designs"][dz]["paired"]["0.1"])
            cu, _ = rep_se(rows, lambda r: r["designs"][dz]["D_star"] <= r["designs"][dz]["union"]["0.1"])
            calu, _ = rep_se(rows, lambda r: r["designs"][dz]["U_new"] <= r["designs"][dz]["union"]["0.1"])
            ci = [max(0.0, cov - 1.96 * se), min(1.0, cov + 1.96 * se)]
            ci_c = [max(0.0, cal - 1.96 * sec), min(1.0, cal + 1.96 * sec)]
            res[f"{tag}|{dz}"] = dict(n=len(v), reps=nrep, damage_ratio=float(np.median(ratio)), cov_paired=cov,
                                      ci_paired=ci, cal_paired=cal, ci_cal=ci_c, cov_union=cu, cal_union=calu)
            p(f"| {dz} | {np.median(ratio):.3f} | {cov:.3f} ({ci[0]:.3f}-{ci[1]:.3f}) | {cal:.3f} "
              f"({ci_c[0]:.3f}-{ci_c[1]:.3f}) | {cu:.3f} | {calu:.3f} |")
    OUT["T"] = res


def table_N():
    """Anchor scenario: anchor sets '300' (tests at 300 s) and '30+300' (at 30 s and 300 s), n_a tests per time."""
    res = {}
    for tag in ("N_evidence", "N_boundary", "N_adverse_strict"):
        rows = mrows(tag)
        if not rows:
            continue
        p(f"\n### N, anchor tests at the field heating times, {tag} ({len(rows)} module records, n1 40)\n")
        p("| anchor set (s) | n_a per time | alpha | coverage (95 % CI) | certificate / stated, median | certificate / "
          "true damage, median | all bounds >= true effects |")
        p("|---|---|---|---|---|---|---|")
        D = np.array([r["D_star"] for r in rows])
        for a in rows[0]["stated"]:
            B = np.array([r["stated"][a] for r in rows])
            k = int(np.sum(D <= B))
            lo, hi = binom_ci(k, len(D))
            res[f"{tag}|stated|{a}"] = dict(coverage=k / len(D), ci=[lo, hi], ratio_to_stated=1.0,
                                           ratio_to_true=float(np.median(B / D)))
            p(f"| stated rule | | {a} | {k / len(D):.3f} ({lo:.3f}-{hi:.3f}) | 1.00 | {np.median(B / D):.2f} | |")
            for key in rows[0]["anchors"]:
                ts, na = key.split("|")
                C = np.array([r["anchors"][key]["certs"][a] for r in rows])
                ok = np.array([all(b >= r["true_effect"][t] for b, t in zip(r["anchors"][key]["bounds"], ts.split("+")))
                               for r in rows])
                k = int(np.sum(D <= C))
                lo, hi = binom_ci(k, len(D))
                res[f"{tag}|{key}|{a}"] = dict(coverage=k / len(D), ci=[lo, hi],
                                               ratio_to_stated=float(np.median(C / B)),
                                               ratio_to_true=float(np.median(C / D)), bounds_ok=float(ok.mean()))
                p(f"| {ts} | {na} | {a} | {k / len(D):.3f} ({lo:.3f}-{hi:.3f}) | {np.median(C / B):.2f} | "
                  f"{np.median(C / D):.2f} | {ok.mean():.3f} |")
        for t in rows[0]["true_effect"]:
            te = np.array([r["true_effect"][t] for r in rows if r["module"] == 0])
            stated = 0.5 * math.log(min(float(t), 60.0) / 4.5) + 0.5 * math.log(max(float(t) / 60.0, 1.0))
            res[f"{tag}|true_effect|{t}"] = dict(median=float(np.median(te)), q10=float(np.quantile(te, 0.1)),
                                                 q90=float(np.quantile(te, 0.9)), stated=stated)
            p(f"\ntrue heating-time effect at {t} s: median {np.median(te):.2f} (q10 {np.quantile(te, 0.1):.2f}, "
              f"q90 {np.quantile(te, 0.9):.2f}); stated rule {stated:.2f}")
    OUT["N"] = res


def table_X():
    rows = mrows("X_ton")
    if not rows:
        return
    p(f"\n### X, exceedance of the stated heating-time margin (truth at the margins, both slopes at g; {len(rows)} "
      f"module records, n1 40, alpha 0.1)\n")
    keys = list(rows[0]["g"]["0.5"]["anchors"])
    p("| true slope g | true damage / at g = 0.5, median | coverage, stated rule | " +
      " | ".join(f"coverage, anchors {k.replace('|', ' s, n_a ')}" for k in keys) + " |")
    p("|---" * (3 + len(keys)) + "|")
    res = {}
    D5 = np.array([r["g"]["0.5"]["D_star"] for r in rows])
    B = np.array([r["stated"]["0.1"] for r in rows])
    for g in rows[0]["g"]:
        D = np.array([r["g"][g]["D_star"] for r in rows])
        Ck = {k: np.array([r["g"][g]["anchors"][k]["certs"]["0.1"] for r in rows]) for k in keys}
        cov = {k: float(np.mean(D <= C)) for k, C in Ck.items()}
        res[g] = dict(damage_ratio=float(np.median(D / D5)), cov_stated=float(np.mean(D <= B)), cov_anchor=cov,
                      ratio_stated=float(np.median(B / D)),
                      ratio_anchor={k: float(np.median(C / D)) for k, C in Ck.items()})
        p(f"| {g} | {np.median(D / D5):.2f} | {np.mean(D <= B):.3f} | " +
          " | ".join(f"{cov[k]:.3f}" for k in keys) + " |")
    OUT["X"] = res


def table_L():
    """Lot shift: coverage of the true damage of the field modules (pooled over repetitions, modules and the 200
    field legs of each), its standard error over repetitions, finite share and median width relative to the paired
    certificate of the bench lot."""
    res = {}
    for tag in ("L_boundary", "L_evidence"):
        rows = mrows(tag)
        if not rows:
            continue
        nrep = len({r["rep"] for r in rows})
        p(f"\n### L, lot shift, {tag} ({nrep} repetitions x {K_MOD} modules x 200 field legs, n1 40, n2 100): coverage "
          f"of the true damage (standard error over repetitions) / of the calibrated part [median width against the "
          f"paired certificate of the bench lot]\n")
        alphas = list(rows[0]["certs"])
        names = list(rows[0]["certs"][alphas[0]]) + [f"recal {k.split('|')[0]}" for k in rows[0]["variants"]["same"]
                                                     ["recal"] if k.endswith(f"|{alphas[0]}")]
        p("| field lot | alpha | " + " | ".join(names) + " |")
        p("|---" * (2 + len(names)) + "|")
        for var in rows[0]["variants"]:
            for a in alphas:
                cells = []
                for nm in names:
                    cov, se = rep_se(rows, lambda r: r["variants"][var]["coverage"][a][nm])
                    cal, _ = rep_se(rows, lambda r: r["variants"][var]["calibrated"][a][nm])
                    if nm.startswith("recal"):
                        C = np.array([r["variants"][var]["recal"][f"{nm.split()[1]}|{a}"] for r in rows])
                    else:
                        C = np.array([r["certs"][a][nm] for r in rows])
                    B = np.array([r["certs"][a]["U6 paired"] for r in rows])
                    fin = np.isfinite(C) & np.isfinite(B)
                    width = float(np.median(C[fin] / B[fin])) if fin.any() else None
                    res[f"{tag}|{var}|{a}|{nm}"] = dict(coverage=cov, se=se, calibrated=cal,
                                                        finite=float(np.isfinite(C).mean()), width_vs_paired=width)
                    cells.append(f"{cov:.3f} ({se:.3f}) / {cal:.3f}"
                                 + ("" if np.isfinite(C).all() else f" (finite {np.isfinite(C).mean():.2f})")
                                 + ("" if width is None or nm == "U6 paired" else f" [x{width:.2f}]"))
                p(f"| {var} | {a} | " + " | ".join(cells) + " |")
    OUT["L"] = res


def table_Y():
    """Several stated components exceeded at once from the truth at the margins: coverage of the true damage pooled over
    repetitions, modules and field legs, with its standard error over repetitions, and the median true damage relative to
    no exceedance."""
    rows = mrows("Y_stack")
    if not rows:
        return
    nrep = len({r["rep"] for r in rows})
    p(f"\n### Y, stated components exceeded at once ({nrep} repetitions x {K_MOD} modules x 50 field legs, n1 40, "
      f"n2 100)\n")
    alphas = list(rows[0]["certs"])
    names = list(rows[0]["certs"][alphas[0]])
    p("| exceeded | true damage / none, median | alpha | " + " | ".join(names) + " |")
    p("|---" * (3 + len(names)) + "|")
    res = {}
    base = np.array([r["variants"]["none"]["D_median"] for r in rows])
    for var in rows[0]["variants"]:
        ratio = float(np.median(np.array([r["variants"][var]["D_median"] for r in rows]) / base))
        for a in alphas:
            cs = {nm: rep_se(rows, lambda r: r["variants"][var]["coverage"][a][nm]) for nm in names}
            res[f"{var}|{a}"] = dict(damage_ratio=ratio, coverage={nm: v[0] for nm, v in cs.items()},
                                     se={nm: v[1] for nm, v in cs.items()})
            p(f"| {var} | {ratio:.2f} | {a} | " + " | ".join(f"{cs[nm][0]:.3f} ({cs[nm][1]:.3f})" for nm in names) + " |")
    OUT["Y"] = res


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # F2: coverage and ratio against n1, paired / union / box / PAC, evidence and boundary, alpha 0.1
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for tag, ls in (("A_evidence_k0", "-"), ("A_boundary_k0", "--")):
        r = OUT.get(tag)
        if not r:
            continue
        n1s = sorted({int(k.split("|")[0]) for k in r if "|stat_ok" not in k})
        for c, col in (("U6 paired", "tab:blue"), ("U5 union", "tab:orange"), ("U1 box", "0.4"), ("U9 PAC", "tab:green")):
            cov = [r.get(f"{n}|0.1|{c}", {}).get("coverage", np.nan) for n in n1s]
            rat = [r.get(f"{n}|0.1|{c}", {}).get("ratio", None) for n in n1s]
            rat = [np.nan if v is None else v for v in rat]
            lab = f"{c.split()[1]} ({tag.split('_')[1]})"
            ax[0].plot(n1s, cov, ls, color=col, marker="o", ms=3, label=lab)
            ax[1].plot(n1s, rat, ls, color=col, marker="o", ms=3, label=lab)
    ax[0].axhline(0.9, color="tab:red", lw=0.6)
    ax[0].set_xlabel("bench scores $n_1$")
    ax[0].set_ylabel("coverage of the true damage")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("bench scores $n_1$")
    ax[1].set_ylabel("certificate / oracle quantile")
    ax[0].legend(*ax[1].get_legend_handles_labels(), fontsize=5.5, frameon=False, ncol=2, loc="lower center")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"g6_F2_nhap.{ext}", dpi=150)
    plt.close(fig)
    # F4/F5: coverage per life against the feedback strength of the policy; served intervals against the damage at
    # replacement (lives without failure) for every run, per truth
    f4, c = OUT.get("F4"), OUT.get("C")
    if f4 and c:
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
        cols = {"union": "tab:orange", "paired": "tab:blue", "split0": "tab:purple", "reuse": "tab:red",
                "aci": "tab:green", "mc": "tab:brown", "bayes": "tab:pink"}
        for st, mk in (("cruise", "o"), ("coast", "s")):
            for run, v in f4.items():
                if not run.startswith(st):
                    continue
                meth = run.split("|")[2]
                lev = c[f"boundary_{st}|{run.split('|', 1)[1].split('|')[0]}|paired"]["mean_level"] or 0.0
                ax[0].errorbar(lev, v["cov_life"], yerr=[[v["cov_life"] - v["ci"][0]], [v["ci"][1] - v["cov_life"]]],
                               fmt=mk, color=cols[meth], ms=3.5, capsize=2, mfc="none" if st == "coast" else cols[meth])
        ax[0].axhline(0.9, color="tab:red", lw=0.6)
        ax[0].set_xlabel("mean derating level of the policy (paired)")
        ax[0].set_ylabel("lives with every certificate covering")
        for a_, scen in ((ax[1], "evidence"), (ax[2], "boundary")):
            for k, v in c.items():
                parts = k.split("|")
                if len(parts) != 3 or not parts[0].startswith(scen) or v.get("harvest_ok") is None:
                    continue
                meth = parts[2]
                a_.scatter(v["intervals_mean"], v["harvest_ok"], s=14, color=cols.get(meth, "0.4"),
                           marker="s" if parts[0].endswith("coast") else "o")
            a_.set_xlabel("served intervals, mean")
            a_.set_ylabel("true damage at replacement")
            a_.set_title(f"{scen} truth", fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"g6_F4F5_nhap.{ext}", dpi=150)
        plt.close(fig)
    # exceedance of the stated heating-time margin, with and without anchor tests
    x = OUT.get("X")
    if x:
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
        gs = sorted(x, key=float)
        g = [float(v) for v in gs]
        ax[0].plot(g, [x[v]["cov_stated"] for v in gs], "-o", color="0.3", ms=4, label="stated rule")
        ax[1].plot(g, [x[v]["ratio_stated"] for v in gs], "-o", color="0.3", ms=4, label="stated rule")
        for k, col in zip(x[gs[0]]["cov_anchor"], ("tab:blue", "tab:green", "tab:orange", "tab:red")):
            lab = f"anchors at {k.split('|')[0]} s, {k.split('|')[1]} each"
            ax[0].plot(g, [x[v]["cov_anchor"][k] for v in gs], "-s", color=col, ms=4, label=lab)
            ax[1].plot(g, [x[v]["ratio_anchor"][k] for v in gs], "-s", color=col, ms=4, label=lab)
        ax[0].axhline(0.9, color="tab:red", lw=0.6)
        ax[0].set_xlabel("true heating-time slope $g$ (stated 0.5)")
        ax[0].set_ylabel("coverage of the true damage")
        ax[0].legend(fontsize=6, frameon=False)
        ax[1].axhline(1.0, color="tab:red", lw=0.6)
        ax[1].set_yscale("log")
        ax[1].set_xlabel("true heating-time slope $g$")
        ax[1].set_ylabel("certificate / true damage, median")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"g6_vuot_bien_nhap.{ext}", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    for tag in ("A_evidence_k0", "A_boundary_k0", "A_adverse_strict_k0", "A_beyond_k0", "A_evidence_k20",
                "A_evidence_n500"):
        table_A(tag)
    table_C()
    table_F4()
    table_AGE()
    table_D()
    table_E()
    table_R()
    table_T()
    table_N()
    table_X()
    table_L()
    table_Y()
    figures()
    (HERE / "g6_tong_hop.json").write_text(json.dumps(OUT, indent=1), encoding="utf-8", newline="\n")
    (HERE / "g6_tong_hop.md").write_text("\n".join(MD) + "\n", encoding="utf-8", newline="\n")
