"""P3, third review (19/10/2026): summary of the new runs of thu_nghiem_g6.py.

  U  paired certificate against union certificates with a small mission level (U_boundary, U_evidence)
  Q  truths outside the family of the stated margin (Q_boundary_int, Q_evidence_int, Q_boundary_lutz; static, as A)
  M  mission change within a life (M_<truth>_<first>_<second>)
  C  closed loop at alpha = 0.05 (C_boundary_cruise_a05, C_boundary_coast_a05)
  S  baseplate solder as a second mechanism of the truth (S_<truth>_<strategy>; static, as A)
  K  multi-split paired certificate (K_boundary, K_evidence; 15 splits of the same bench)

Coverage intervals are mean +- 1.96 standard errors over repetitions (the modules of a repetition share its world, bench
and recorded intervals); failure shares of lives carry Wilson 95 % intervals (one leg per life); cost per 100 intervals
in planned replacements: a life costs 1, plus c_F - 1 if it ends in failure, divided by the intervals served (lateness
not priced; the M and a05 runs do not derate the C runs' baselines). Paired differences over common lives with
bootstrap 95 % intervals.

Run: python tong_hop_pb3.py  -> tong_hop_pb3.json, tong_hop_pb3.md
"""
import json
import math
import site
import sys
from pathlib import Path

sys.path.insert(0, site.getusersitepackages())
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, MD = {}, []
C_F = (10, 50)
SOL_CU = 10.0          # S: copper baseplate, ten times fewer cycles (as thu_nghiem_g6.SOL_CU)


def p(line=""):
    MD.append(line)
    print(line)


def load(tag):
    f = HERE / f"g6_{tag}.jsonl"
    if not f.exists():
        return []
    rows = {}
    for x in f.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            rows.setdefault(r["rep"], r)
    return [rows[k] for k in sorted(rows)]


def rep_mean(rows, value):
    """Mean over repetitions of the per-repetition mean of value(module record), its standard error."""
    m = np.array([np.mean([value(x) for x in r["modules"]]) for r in rows], float)
    return float(m.mean()), float(m.std(ddof=1) / math.sqrt(len(m)))


def wilson(k, n, z=1.96):
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def cost100(T, F, cf):
    return float(100 * (1 + (cf - 1) * F).sum() / max(T.sum(), 1))


# ------------------------------------------------------------------------------------------------------------- U
def table_U():
    res = {}
    for sc in ("boundary", "evidence"):
        rows = load(f"U_{sc}")
        if not rows:
            continue
        p(f"\n### U_{sc}: paired against union with a small mission level ({len(rows)} repetitions, n2 100)\n")
        p("| n1 | alpha | certificate | finite share | median size / paired | coverage true (+-1.96 se) | "
          "coverage of U_new |")
        p("|---|---|---|---|---|---|---|")
        for n1 in rows[0]["modules"][0]["n1"]:
            for a in rows[0]["modules"][0]["n1"][n1]["certs"]:
                for name in rows[0]["modules"][0]["n1"][n1]["certs"][a]:
                    def c(x):
                        return x["n1"][n1]["certs"][a][name]
                    fin, _ = rep_mean(rows, lambda x: math.isfinite(c(x)))
                    pr = [c(x) / x["n1"][n1]["certs"][a]["paired"] for r in rows for x in r["modules"]
                          if math.isfinite(x["n1"][n1]["certs"][a]["paired"])]
                    med = float(np.median(pr)) if pr else math.nan
                    cov, se = rep_mean(rows, lambda x: x["D_star"] <= c(x))
                    cal, _ = rep_mean(rows, lambda x: x["n1"][n1]["U_new"] <= c(x))
                    res[f"{sc}|{n1}|{a}|{name}"] = dict(finite=fin, size_vs_paired=med, cov=cov, se=se, cov_Unew=cal)
                    p(f"| {n1} | {a} | {name} | {fin:.3f} | {med:.3f} | {cov:.3f} ({cov - 1.96 * se:.3f}-"
                      f"{min(1, cov + 1.96 * se):.3f}) | {cal:.3f} |")
    OUT["U"] = res


# ------------------------------------------------------------------------------------------------------------- Q
CERTS = ("U1 box", "U2 Monte Carlo", "U3 Bayes", "U5 union", "U6 paired", "U9 PAC")


def table_Q():
    res = {}
    for tag in ("Q_boundary_int", "Q_evidence_int", "Q_boundary_lutz", "Q_boundary_nc", "Q_evidence_nc",
                "Q_boundary_th", "Q_evidence_th", "Q_boundary_seq", "Q_evidence_seq"):
        rows = load(tag)
        if not rows:
            continue
        p(f"\n### {tag} ({len(rows)} repetitions, n2 100)\n")
        p("| n1 | alpha | certificate | coverage of the true damage (+-1.96 se) | median size / oracle |")
        p("|---|---|---|---|---|")
        for n1 in rows[0]["modules"][0]["n1"]:
            for a in rows[0]["modules"][0]["n1"][n1]["certs"]:
                for cert in CERTS:
                    def c(x):
                        return x["n1"][n1]["certs"][a][cert]
                    cov, se = rep_mean(rows, lambda x: x["D_star"] <= c(x))
                    rat = [c(x) / x["oracle"][a] for r in rows for x in r["modules"] if math.isfinite(c(x))]
                    med = float(np.median(rat)) if rat else math.nan
                    res[f"{tag}|{n1}|{a}|{cert}"] = dict(cov=cov, se=se, size_vs_oracle=med)
                    p(f"| {n1} | {a} | {cert} | {cov:.3f} ({max(0, cov - 1.96 * se):.3f}-{min(1, cov + 1.96 * se):.3f}) "
                      f"| {med:.2f} |")
                cal, sec = rep_mean(rows, lambda x: x["n1"][n1]["U_new"] <= x["n1"][n1]["certs"][a]["U6 paired"])
                res[f"{tag}|{n1}|{a}|calibrated"] = dict(cov=cal, se=sec)
                p(f"| {n1} | {a} | paired, calibrated part | {cal:.3f} ({cal - 1.96 * sec:.3f}-"
                  f"{min(1, cal + 1.96 * sec):.3f}) | |")
        oc = [x["D_star"] <= x["oracle"]["0.1"] for r in rows for x in r["modules"]]
        res[f"{tag}|oracle_cov_0.1"] = float(np.mean(oc))
    OUT["Q"] = res


# ------------------------------------------------------------------------------------------------------------- M
def lives_summary(rows, run):
    T = np.array([r["runs"][run]["intervals"] for r in rows], float)
    F = np.array([r["runs"][run]["failed"] for r in rows], float)
    out = dict(intervals=float(T.mean()), failed=float(F.mean()), failed_ci=wilson(int(F.sum()), len(F)),
               **{f"cost{cf}": cost100(T, F, cf) for cf in C_F})
    if "bits" in rows[0]["runs"][run]:
        bits = [b for r in rows for b in r["runs"][run]["bits"]]
        out["cov_decision"] = float(np.mean([((b & 1) + ((b >> 1) & 1)) / 2 for b in bits])) if bits else None
        after = [b for r in rows for t, b in enumerate(r["runs"][run]["bits"]) if t >= r["t_sw"]]
        out["cov_after_switch"] = float(np.mean([((b & 1) + ((b >> 1) & 1)) / 2 for b in after])) if after else None
        out["lives_all_covered"] = float(np.mean([all(b == 3 for b in r["runs"][run]["bits"]) for r in rows]))
    return out


def paired_cost_diff(rows, ra, rb, cf, n_boot=2000, seed=1):
    Ta = np.array([r["runs"][ra]["intervals"] for r in rows], float)
    Fa = np.array([r["runs"][ra]["failed"] for r in rows], float)
    Tb = np.array([r["runs"][rb]["intervals"] for r in rows], float)
    Fb = np.array([r["runs"][rb]["failed"] for r in rows], float)
    rng = np.random.default_rng(seed)
    B = rng.integers(0, len(Ta), (n_boot, len(Ta)))

    def c(T, F, idx):
        return 100 * (1 + (cf - 1) * F[idx]).sum(-1) / np.maximum(T[idx].sum(-1), 1)
    full = np.arange(len(Ta))
    d = float(c(Tb, Fb, full) - c(Ta, Fa, full))
    db = c(Tb, Fb, B) - c(Ta, Fa, B)
    return dict(diff=d, ci=[float(np.quantile(db, 0.025)), float(np.quantile(db, 0.975))])


def table_M():
    res = {}
    for sc in ("boundary", "evidence"):
        for a, b in (("cruise", "coast"), ("coast", "cruise")):
            tag = f"M_{sc}_{a}_{b}"
            rows = load(tag)
            if not rows:
                continue
            p(f"\n### {tag}: {a} then {b} from an interval drawn in 10-30 ({len(rows)} lives)\n")
            p("| run | served intervals | failed lives (Wilson 95 %) | cost c_F=10 | cost c_F=50 | coverage per decision "
              "| after the change |")
            p("|---|---|---|---|---|---|---|")
            r0 = {}
            for run in rows[0]["runs"]:
                s = lives_summary(rows, run)
                r0[run] = s
                p(f"| {run} | {s['intervals']:.1f} | {100 * s['failed']:.1f} ({100 * s['failed_ci'][0]:.1f}-"
                  f"{100 * s['failed_ci'][1]:.1f}) | {s['cost10']:.2f} | {s['cost50']:.2f} | "
                  f"{s.get('cov_decision') if s.get('cov_decision') is None else round(s['cov_decision'], 3)} | "
                  f"{s.get('cov_after_switch') if s.get('cov_after_switch') is None else round(s['cov_after_switch'], 3)} |")
            diffs = {}
            for other in rows[0]["runs"]:
                if other != "cert|paired":
                    diffs[other] = {str(cf): paired_cost_diff(rows, "cert|paired", other, cf) for cf in C_F}
            p("\nCost of each run minus that of cert|paired (bootstrap 95 %):\n")
            for other, d in diffs.items():
                p(f"- {other}: " + "; ".join(f"c_F={cf}: {v['diff']:+.2f} ({v['ci'][0]:+.2f} to {v['ci'][1]:+.2f})"
                                             for cf, v in d.items()))
            share_before = float(np.mean([min(r["ages"].values()) <= r["t_sw"] for r in rows]))
            res[tag] = dict(lives=len(rows), runs=r0, diff_vs_paired=diffs,
                            t_sw_mean=float(np.mean([r["t_sw"] for r in rows])),
                            share_some_static_age_before_change=share_before)
    OUT["M"] = res


# ------------------------------------------------------------------------------------------------------- C a05
def table_C05():
    res = {}
    for st in ("cruise", "coast"):
        tag = f"C_boundary_{st}_a05"
        rows = load(tag)
        if not rows:
            continue
        p(f"\n### {tag} ({len(rows)} lives, alpha 0.05, n1 40)\n")
        p("| run | served intervals | failed lives (Wilson 95 %) | coverage per decision | late sections | "
          "cost c_F=10 | cost c_F=50 |")
        p("|---|---|---|---|---|---|---|")
        r0 = {}
        for run in rows[0]["runs"]:
            x = [r["runs"][run] for r in rows]
            T = np.array([v["intervals"] for v in x], float)
            F = np.array([v["failed"] for v in x], float)
            cov = [v["coverage"] for v in x if v.get("coverage") is not None]
            late = [v["late_share"] for v in x if v.get("late_share") is not None]
            s = dict(intervals=float(T.mean()), failed=float(F.mean()), failed_ci=wilson(int(F.sum()), len(F)),
                     coverage=float(np.mean(cov)) if cov else None, late=float(np.mean(late)) if late else None,
                     **{f"cost{cf}": cost100(T, F, cf) for cf in C_F})
            r0[run] = s
            p(f"| {run} | {s['intervals']:.1f} | {100 * s['failed']:.1f} ({100 * s['failed_ci'][0]:.1f}-"
              f"{100 * s['failed_ci'][1]:.1f}) | {'' if s['coverage'] is None else round(s['coverage'], 3)} | "
              f"{'' if s['late'] is None else round(100 * s['late'], 1)} | {s['cost10']:.2f} | {s['cost50']:.2f} |")
        res[tag] = r0
    OUT["C05"] = res


# ------------------------------------------------------------------------------------------------------------- F
def table_F():
    """Fleets of n_legs legs sharing one bench (rep_F): failures per fleet across benches."""
    res = {}
    for tag in ("F_boundary_cruise", "F_boundary_coast", "F_beyond_cruise"):
        rows = load(tag)
        if not rows:
            continue
        nl = rows[0]["n_legs"]
        p(f"\n### {tag} ({len(rows)} fleets of {nl} legs, one bench each)\n")
        p("| run | failed legs, mean (%) | fleets: median, 90 %, max (%) | fleets with >= 2 failures (%) | "
          "cost c_F=10 | cost c_F=50 | coverage per decision |")
        p("|---|---|---|---|---|---|---|")
        r0 = {}
        for run in rows[0]["runs"]:
            f = np.array([r["runs"][run]["failed"] / nl for r in rows])
            T = np.array([r["runs"][run]["intervals"] for r in rows], float)
            Fn = np.array([r["runs"][run]["failed"] for r in rows], float)
            dec = sum(r["runs"][run]["decisions"] for r in rows)
            cov = sum(r["runs"][run]["covered"] for r in rows) / dec if dec else None
            s = dict(failed=float(f.mean()), median=float(np.median(f)), q90=float(np.quantile(f, 0.9)),
                     max=float(f.max()), fleets_ge2=float(np.mean(Fn >= 2)),
                     cost10=float(100 * (nl * len(rows) + 9 * Fn.sum()) / T.sum()),
                     cost50=float(100 * (nl * len(rows) + 49 * Fn.sum()) / T.sum()), coverage=cov,
                     intervals=float(T.sum() / (nl * len(rows))))
            r0[run] = s
            p(f"| {run} | {100 * s['failed']:.1f} | {100 * s['median']:.0f}, {100 * s['q90']:.0f}, {100 * s['max']:.0f} | "
              f"{100 * s['fleets_ge2']:.0f} | {s['cost10']:.2f} | {s['cost50']:.2f} | "
              f"{'' if cov is None else round(cov, 3)} |")
        res[tag] = dict(fleets=len(rows), legs=nl, pac_rank=sorted({r["pac_rank"] for r in rows}), runs=r0)
    OUT["F"] = res


def table_C40():
    res = {}
    for st in ("cruise", "coast"):
        tag = f"C_boundary_{st}_n40"
        rows = load(tag)
        if not rows:
            continue
        p(f"\n### {tag} ({len(rows)} lives, alpha 0.1, n1 40, PAC eta 0.1)\n")
        p("| run | served intervals | failed lives (Wilson 95 %) | coverage per decision | late sections | "
          "cost c_F=10 | cost c_F=50 |")
        p("|---|---|---|---|---|---|---|")
        r0 = {}
        for run in rows[0]["runs"]:
            x = [r["runs"][run] for r in rows]
            T = np.array([v["intervals"] for v in x], float)
            F = np.array([v["failed"] for v in x], float)
            cov = [v["coverage"] for v in x if v.get("coverage") is not None]
            late = [v["late_share"] for v in x if v.get("late_share") is not None]
            s = dict(intervals=float(T.mean()), failed=float(F.mean()), failed_ci=wilson(int(F.sum()), len(F)),
                     coverage=float(np.mean(cov)) if cov else None, late=float(np.mean(late)) if late else None,
                     **{f"cost{cf}": cost100(T, F, cf) for cf in C_F})
            r0[run] = s
            p(f"| {run} | {s['intervals']:.1f} | {100 * s['failed']:.1f} ({100 * s['failed_ci'][0]:.1f}-"
              f"{100 * s['failed_ci'][1]:.1f}) | {'' if s['coverage'] is None else round(s['coverage'], 3)} | "
              f"{'' if s['late'] is None else round(100 * s['late'], 1)} | {s['cost10']:.2f} | {s['cost50']:.2f} |")
        res[tag] = r0
    OUT["C40"] = res


# ------------------------------------------------------------------------------------------------------------- S
def table_S():
    """Baseplate solder as a second mechanism of the truth (S_*, the A_*_k0 runs with a solder channel): coverage of
    the true damage of the module, the larger of the junction and solder damages, with an AlSiC or a copper baseplate."""
    res = {}
    for tag in ("S_boundary_cruise", "S_evidence_cruise", "S_boundary_coast", "S_evidence_coast"):
        rows = load(tag)
        if not rows:
            continue
        changes = []
        x0 = [x for r in rows for x in r["modules"]]
        rat = np.array([x["solder"]["D_sol"] / x["D_star"] for x in x0])
        gov = {b: float(np.mean([x["solder"]["D_star"][b] > x["D_star"] for x in x0])) for b in ("AlSiC", "Cu")}
        res[f"{tag}|ratio"] = dict(median=float(np.median(rat)), q90=float(np.quantile(rat, 0.9)),
                                   q99=float(np.quantile(rat, 0.99)), max=float(rat.max()), governs=gov,
                                   c_sol_median=float(np.median([r["c_sol"] for r in rows])))
        p(f"\n### {tag} ({len(rows)} repetitions): solder / junction damage median {np.median(rat):.3f}, "
          f"90 % {np.quantile(rat, 0.9):.3f}, 99 % {np.quantile(rat, 0.99):.3f}, max {rat.max():.3f}; solder governs "
          f"{100 * gov['AlSiC']:.1f} % of modules (AlSiC), {100 * gov['Cu']:.1f} % (Cu)\n")
        p("| n1 | alpha | certificate | junction only | AlSiC | Cu |")
        p("|---|---|---|---|---|---|")
        for n1 in rows[0]["modules"][0]["n1"]:
            for a in rows[0]["modules"][0]["n1"][n1]["certs"]:
                for cert in ("U5 union", "U6 paired"):
                    def c(x):
                        return x["n1"][n1]["certs"][a][cert]
                    cj = rep_mean(rows, lambda x: x["D_star"] <= c(x))
                    cs = {b: rep_mean(rows, lambda x: x["solder"]["D_star"][b] <= c(x)) for b in ("AlSiC", "Cu")}
                    res[f"{tag}|{n1}|{a}|{cert}"] = dict(junction=cj, **cs)
                    p(f"| {n1} | {a} | {cert} | {cj[0]:.3f} | {cs['AlSiC'][0]:.3f} | {cs['Cu'][0]:.3f} |")
                    changes += [dict(n1=n1, alpha=a, cert=cert, base=b, junction=cj[0], solder=cs[b][0])
                                for b in ("AlSiC", "Cu") if abs(cs[b][0] - cj[0]) > 1e-12]
        res[f"{tag}|changes"] = changes
        p(f"\nCells changed by the solder channel: {changes if changes else 'none'}")
        if "U_sol" not in x0[0]["solder"]:
            continue
        # two certificates, one per mechanism, each at alpha/2 = 0.05 (union bound for the module at 0.10): covered when
        # both hold; with copper the true solder damage and its certificate both scale by ten, so the coverage is the
        # same, but the size is not. Size of the pair for a new module: the larger of the two certificates, against the
        # junction-only certificate at 0.10 (the reserve of a leg follows the larger certificate)
        def us(x, a):
            return x["solder"]["U_sol"][a]
        sa = {a: rep_mean(rows, lambda x: x["solder"]["D_sol"] <= us(x, a)) for a in ("0.1", "0.05")}
        res[f"{tag}|solder_alone"] = sa
        p(f"\nSolder certificate alone ({len(rows)} reps, 40 tests): coverage of the solder damage "
          f"{sa['0.1'][0]:.3f} at 0.10, {sa['0.05'][0]:.3f} at 0.05\n")
        p("| n1 | certificate | two certificates, coverage | size / junction-only (AlSiC, median) | (Cu) | "
          "solder binds (AlSiC) | (Cu) |")
        p("|---|---|---|---|---|---|---|")
        for n1 in rows[0]["modules"][0]["n1"]:
            for cert in ("U5 union", "U6 paired"):
                def cj(x, a):
                    return x["n1"][n1]["certs"][a][cert]
                cov = rep_mean(rows, lambda x: x["D_star"] <= cj(x, "0.05") and x["solder"]["D_sol"] <= us(x, "0.05"))
                size, bind = {}, {}
                for b, f in (("AlSiC", 1.0), ("Cu", SOL_CU)):
                    rt = [max(cj(x, "0.05"), f * us(x, "0.05")) / cj(x, "0.1") for x in x0
                          if math.isfinite(cj(x, "0.1")) and cj(x, "0.1") > 0]
                    size[b] = float(np.median(rt)) if rt else math.inf
                    bind[b] = float(np.mean([f * us(x, "0.05") > cj(x, "0.05") for x in x0]))
                res[f"{tag}|two|{n1}|{cert}"] = dict(cov=cov, size=size, binds=bind)
                p(f"| {n1} | {cert} | {cov[0]:.3f} | {size['AlSiC']:.2f} | {size['Cu']:.2f} | "
                  f"{100 * bind['AlSiC']:.1f} % | {100 * bind['Cu']:.1f} % |")
    OUT["S"] = res


# ------------------------------------------------------------------------------------------------------------- K
def table_K():
    """Multi-split paired certificate (K_*, the A_*_k0 runs with 15 splits of the same bench): coverage and size of
    one split, of the median over the splits at alpha, and of the median at alpha/2 (guaranteed 1 - alpha by Markov's
    inequality), and the spread of the certificate between splits."""
    res = {}
    for tag in ("K_boundary", "K_evidence"):
        rows = load(tag)
        if not rows:
            continue
        x0 = [x for r in rows for x in r["modules"]]
        p(f"\n### {tag} ({len(rows)} repetitions, 15 splits per bench, alpha 0.10)\n")
        p("| n1 | certificate | coverage (+-1.96 se) | median size / oracle | spread q90/q10 between splits |")
        p("|---|---|---|---|---|")
        for n1 in ("20", "40"):
            def mu(x, a):
                return np.asarray(x["n1"][n1]["multi"][a], float)
            kinds = (("one split", lambda x: mu(x, "0.1")[0]),
                     ("median of 15 at alpha", lambda x: float(np.median(mu(x, "0.1")))),
                     ("median of 15 at alpha/2", lambda x: float(np.median(mu(x, "0.05")))))
            for name, cert in kinds:
                cov = rep_mean(rows, lambda x: x["D_star"] <= cert(x))
                size = float(np.median([cert(x) / x["oracle"]["0.1"] for x in x0]))
                res[f"{tag}|{n1}|{name}"] = dict(cov=cov, size_vs_oracle=size)
                p(f"| {n1} | {name} | {cov[0]:.3f} ({cov[0] - 1.96 * cov[1]:.3f}-{min(1, cov[0] + 1.96 * cov[1]):.3f}) "
                  f"| {size:.2f} | |")
            spread = [float(np.quantile(mu(x, "0.1"), 0.9) / np.quantile(mu(x, "0.1"), 0.1)) for x in x0]
            each = rep_mean(rows, lambda x: float(np.mean(x["D_star"] <= mu(x, "0.1"))))
            res[f"{tag}|{n1}|spread"] = dict(median=float(np.median(spread)), q90=float(np.quantile(spread, 0.9)),
                                             mean_cov_each_split=each)
            p(f"| {n1} | spread of one split (median, 90 % over modules) | mean coverage of a split {each[0]:.3f} | | "
              f"{np.median(spread):.2f}, {np.quantile(spread, 0.9):.2f} |")
    OUT["K"] = res


# ------------------------------------------------------------------------------------------------------------- W
def table_W():
    """Coverage of the statistical part of the margin alone (W_boundary): first-order, Monte Carlo and Scheffe
    ellipsoid bounds; per repetition the mean over its intervals and modules; size relative to the first-order bound."""
    rows = load("W_boundary")
    if not rows:
        return
    res = {}
    p(f"\n### W_boundary ({len(rows)} repetitions, {len(rows[0]['n1']['20']) // 2} intervals x 2 modules each)\n")
    p("| n1 | bound | coverage (+-1.96 se) | median size / first-order |")
    p("|---|---|---|---|")
    for n1 in rows[0]["n1"]:
        for st in ("delta", "mc", "ellipsoid"):
            per = np.array([np.mean([rec[st][0] for rec in r["n1"][n1]]) for r in rows])
            cov, se = float(per.mean()), float(per.std(ddof=1) / math.sqrt(len(per)))
            size = float(np.median([rec[st][1] / rec["delta"][1] for r in rows for rec in r["n1"][n1]]))
            res[f"{n1}|{st}"] = dict(cov=cov, se=se, size_vs_delta=size)
            p(f"| {n1} | {st} | {cov:.3f} ({cov - 1.96 * se:.3f}-{min(1, cov + 1.96 * se):.3f}) | {size:.2f} |")
    OUT["W"] = res


if __name__ == "__main__":
    table_U()
    table_Q()
    table_M()
    table_C05()
    table_F()
    table_C40()
    table_S()
    table_K()
    table_W()
    (HERE / "tong_hop_pb3.json").write_text(json.dumps(OUT, indent=1), encoding="utf-8", newline="\n")
    (HERE / "tong_hop_pb3.md").write_text("\n".join(MD) + "\n", encoding="utf-8", newline="\n")
