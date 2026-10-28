"""P3 case study (G7): tables and draft figures of the information loop (thu_nghiem_g7.py).

Tables (markdown, printed and written to g7_tong_hop.md; numbers in g7_tong_hop.json):
  module lives per truth and configuration: served intervals, failures, collapses (bound wider than S_CAP), true damage
  at replacement, coverage per decision and per life, coverage of the thermal bound per module, bound width (mean, end,
  largest) per module, share of the balancing patterns, test pulses per interval, share of the pulses in the margined
  and in the true damage; paired differences between configurations.
Figures: g7_F7_nhap (phase map Phi(F) - F of Proposition P3.26 for the inner module, start and mid-life, with and without
dwell-window calibration and with operation as the source of information), g7_quy_dao_nhap (bound width of the outer and
inner modules over the life), as .pdf/.png.

Run: python -I g7_tong_hop.py
"""
import json
import math
import site
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

OUT, MD = {}, []
CONFS = ("stated", "stated_bal", "op", "op_bal", "pulse_gated", "pulse_gated_min", "pulse_gated_plan",
         "pulse_gated_plan_min", "pulse_always")
LABEL = {"stated": "stated bound 1.10", "stated_bal": "stated bound 1.10, balancing",
         "op": "operation only", "op_bal": "operation only, balancing",
         "pulse_gated": "pulses gated by reserve (10-30)", "pulse_gated_min": "gated (10-30) + dwell minimum",
         "pulse_gated_plan": "pulses gated by plan (30-60)", "pulse_gated_plan_min": "gated (30-60) + dwell minimum",
         "pulse_always": "pulses at full rate"}


def load(tag):
    f = HERE / f"g7_{tag}.jsonl"
    if not f.exists():
        return []
    rows = {}
    for x in f.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            rows.setdefault(r["rep"], r)
    return [rows[k] for k in sorted(rows)]


def p(line=""):
    MD.append(line)
    print(line)


def table_lives(tag):
    rows = load(tag)
    if not rows:
        return
    p(f"\n### {tag}: module lives ({len(rows)} repetitions, n1 20, n2 50, alpha 0.1)\n")
    p("| configuration | intervals, mean (median) | failures | collapses | true damage at replacement | coverage per "
      "decision | coverage per life | thermal bound covers outer / inner | bound width outer / inner, mean | largest | 'in' / "
      "'out' share | pulses per interval | pulse share of margined / true damage |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    res = {}
    for c in CONFS:
        x = [r["lives"][c] for r in rows if c in r["lives"]]
        if not x:
            continue
        I = np.array([v["intervals"] for v in x])
        g = lambda k: [v[k] for v in x if v[k] is not None]
        th = np.array(g("th_cov"))
        om = np.array(g("omega_mean"))
        omx = np.array(g("omega_max"))
        row = dict(intervals_mean=float(I.mean()), intervals_median=float(np.median(I)),
                   failed=float(np.mean(g("failed"))), collapse=float(np.mean(g("collapse"))),
                   harvest=float(np.mean(g("harvest"))), coverage=float(np.mean(g("coverage"))),
                   cov_life=float(np.mean(g("cov_life"))), th_cov=th.mean(0).tolist(), omega=om.mean(0).tolist(),
                   omega_max=np.median(omx, 0).tolist(),
                   share_in=float(np.mean([v["share"]["in"] for v in x])),
                   share_out=float(np.mean([v["share"]["out"] for v in x])),
                   pulses=float(np.mean([v["pulses"] / max(v["intervals"], 1) for v in x])),
                   pulse_model=float(np.mean(g("pulse_model_share"))), pulse_true=float(np.mean(g("pulse_true_share"))))
        res[c] = row
        p(f"| {LABEL[c]} | {row['intervals_mean']:.1f} ({row['intervals_median']:.0f}) | {row['failed']:.2f} | "
          f"{row['collapse']:.2f} | {row['harvest']:.3f} | {row['coverage']:.3f} | {row['cov_life']:.3f} | "
          f"{row['th_cov'][0]:.3f} / {row['th_cov'][1]:.3f} | {row['omega'][0]:.3f} / {row['omega'][1]:.3f} | "
          f"{row['omega_max'][0]:.3f} / {row['omega_max'][1]:.3f} | {row['share_in']:.2f} / {row['share_out']:.2f} | "
          f"{row['pulses']:.0f} | {100 * row['pulse_model']:.1f} % / {100 * row['pulse_true']:.1f} % |")
    p("\nPaired differences in served intervals (mean; lives longer / shorter):\n")
    for a, b in (("op_bal", "op"), ("stated_bal", "stated"), ("op", "stated"), ("pulse_gated_min", "pulse_gated"),
                 ("pulse_gated_plan_min", "pulse_gated_plan"), ("pulse_always", "pulse_gated"),
                 ("pulse_always", "pulse_gated_plan"), ("pulse_always", "pulse_gated_min"),
                 ("pulse_always", "stated_bal")):
        d = np.array([r["lives"][a]["intervals"] - r["lives"][b]["intervals"] for r in rows
                      if a in r["lives"] and b in r["lives"]])
        if d.size:
            res[f"{a}-{b}"] = dict(mean=float(d.mean()), longer=int((d > 0).sum()), shorter=int((d < 0).sum()))
            p(f"- {LABEL[a]} minus {LABEL[b]}: {d.mean():+.1f} ({int((d > 0).sum())} / {int((d < 0).sum())})")
    OUT[tag] = res


def crossings(F, d):
    """Fixed points of a phase map: zeros of d = Phi(F) - F by linear interpolation between grid points (exact for a
    linear map). Until 17/10/2026 the geometric midpoint of the bracketing cell was taken, which is off by up to one
    grid step (a factor 1.26 in F, 12 % in bound width)."""
    return [float(F[i] - d[i] * (F[i + 1] - F[i]) / (d[i + 1] - d[i])) for i in range(len(F) - 1) if d[i] * d[i + 1] < 0]


def phase_summary(tag):
    rows = load(tag)
    if not rows:
        return None
    F = np.geomspace(5.0, 5000.0, len(rows[0]["phase"]["start"]["op"]))
    res = {}
    for age in ("start", "mid"):
        for c in rows[0]["phase"][age]:
            A = np.array([r["phase"][age][c] for r in rows])          # [reps, F, (phi1, phi2, n_p, R)]
            med = np.median(A, 0)
            res[f"{age}|{c}"] = dict(F=F.tolist(), phi=[med[:, 0].tolist(), med[:, 1].tolist()],
                                     n_p=med[:, 2].tolist(), R=med[:, 3].tolist(),
                                     q10=[np.quantile(A[:, :, 0], 0.1, 0).tolist(),
                                          np.quantile(A[:, :, 1], 0.1, 0).tolist()],
                                     q90=[np.quantile(A[:, :, 0], 0.9, 0).tolist(),
                                          np.quantile(A[:, :, 1], 0.9, 0).tolist()])
            # fixed points of the median map for the inner module (key fixed_T2): sign changes of Phi(F) - F
            d = med[:, 1] - F
            fp = crossings(F, d)
            res[f"{age}|{c}"]["fixed_T2"] = fp
            res[f"{age}|{c}"]["below_diag_at_min_F_T2"] = bool(d[0] < 0)
    OUT[f"{tag}|phase"] = res
    p(f"\n### {tag}: phase map of the information of the inner module (median over {len(rows)} worlds)\n")
    p("| age | configuration | Phi(F) - F at F = 5 | fixed points of the median map (F) | bound width at the largest "
      "fixed point |")
    p("|---|---|---|---|---|")
    for k, v in res.items():
        age, c = k.split("|")
        fp = v["fixed_T2"]
        w = f"{1.645 / math.sqrt(max(fp)):.3f}" if fp else "--"
        p(f"| {age} | {LABEL[c]} | {v['phi'][1][0] - v['F'][0]:+.1f} | "
          f"{', '.join(f'{x:.0f}' for x in fp) if fp else 'none'} | {w} |")
    return res


def table_lam():
    """Forgetting factor 0.6 / 0.8 / 0.9 on the same 30 worlds (evidence truth)."""
    base = {r["rep"]: r for r in load("L_evidence")}
    sets = {"0.6": load("S_lam06"), "0.8": [base[k] for k in sorted(base) if k < 30], "0.9": load("S_lam09")}
    if not sets["0.6"] or not sets["0.9"]:
        return
    p("\n### Forgetting factor (evidence truth, the same 30 worlds; rates recalibrated to the same bound widths)\n")
    p("| lambda | configuration | intervals, mean | largest bound width, median | phase map at mid-life: fixed points "
      "of the inner module |")
    p("|---|---|---|---|---|")
    res = {}
    for lam, rows in sets.items():
        F = np.geomspace(5.0, 5000.0, len(rows[0]["phase"]["mid"]["op"]))
        for c in ("pulse_gated", "pulse_gated_plan", "pulse_gated_plan_min", "pulse_always"):
            I = np.mean([r["lives"][c]["intervals"] for r in rows])
            om = np.median([r["lives"][c]["omega_max"][1] for r in rows if r["lives"][c]["omega_max"] is not None])
            fp = "--"
            if c in rows[0]["phase"]["mid"]:
                med = np.median(np.array([r["phase"]["mid"][c] for r in rows]), 0)
                d = med[:, 1] - F
                pts = crossings(F, d)
                fp = ", ".join(f"{x:.0f}" for x in pts) if pts else "none (collapse)"
            res[f"{lam}|{c}"] = dict(intervals=float(I), omega_max=float(om), fixed_mid=fp)
            p(f"| {lam} | {LABEL[c]} | {I:.1f} | {om:.3f} | {fp} |")
    OUT["lambda"] = res


def table_sens():
    """Confidence z of the thermal bound, drift of the thermal path and minimum dwell pulses, each on the same 30 worlds
    as the base runs (z 1.645, drift 0.2, 60 pulses)."""
    ev = {r["rep"]: r for r in load("L_evidence")}
    bd = {r["rep"]: r for r in load("L_boundary")}
    first = lambda d: [d[k] for k in sorted(d) if k < 30]
    groups = (("z of the thermal bound", (("1.28", load("S_z128")), ("1.645", first(ev)), ("2.33", load("S_z233"))),
               ("op", "pulse_gated_plan", "pulse_gated_plan_min", "pulse_always")),
              ("drift of the thermal path per unit damage (boundary)",
               (("0", load("S_rho0")), ("0.2", first(bd)), ("0.4", load("S_rho04"))),
               ("stated", "op", "pulse_gated_plan_min", "pulse_always")),
              ("minimum dwell pulses per interval", (("20", load("S_nmin20")), ("60", first(ev)),
                                                     ("120", load("S_nmin120"))),
               ("pulse_gated_min", "pulse_gated_plan", "pulse_gated_plan_min")))
    res = {}
    for name, sets, confs in groups:
        if any(not rows for _, rows in sets):
            continue
        p(f"\n### Sensitivity: {name} (the same 30 worlds)\n")
        p("| value | configuration | intervals, mean | failures | coverage per life | thermal bound covers outer / inner | "
          "bound width inner, mean / largest (median) | pulse share of margined damage |")
        p("|---|---|---|---|---|---|---|---|")
        for val, rows in sets:
            for c in confs:
                x = [r["lives"][c] for r in rows]
                g = lambda k: [v[k] for v in x if v[k] is not None]
                th = np.array(g("th_cov"))
                om = np.array(g("omega_mean"))
                omx = np.array(g("omega_max"))
                row = dict(intervals=float(np.mean([v["intervals"] for v in x])), failed=float(np.mean(g("failed"))),
                           cov_life=float(np.mean(g("cov_life"))), th_cov=th.mean(0).tolist(),
                           omega=float(om[:, 1].mean()), omega_max=float(np.median(omx[:, 1])),
                           pulse=float(np.mean(g("pulse_model_share"))))
                res[f"{name}|{val}|{c}"] = row
                p(f"| {val} | {LABEL[c]} | {row['intervals']:.1f} | {row['failed']:.2f} | {row['cov_life']:.3f} | "
                  f"{row['th_cov'][0]:.3f} / {row['th_cov'][1]:.3f} | {row['omega']:.3f} / {row['omega_max']:.3f} | "
                  f"{100 * row['pulse']:.1f} % |")
    OUT["sens"] = res


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ph = OUT.get("L_evidence|phase") or OUT.get("L_boundary|phase")
    if ph:
        # phase portrait: Phi(F) against F on log axes; fixed points are the crossings of the diagonal
        fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
        cols = {"pulse_gated_plan": ("tab:orange", "-"), "pulse_gated_plan_min": ("tab:green", "-"),
                "pulse_gated": ("tab:red", ":"), "op": ("tab:blue", "-"), "op_bal": ("tab:purple", "--")}
        for a_, age in zip(ax, ("start", "mid")):
            F = None
            for c, (col, ls) in cols.items():
                v = ph.get(f"{age}|{c}")
                if not v:
                    continue
                F = np.array(v["F"])
                a_.plot(F, np.array(v["phi"][1]), ls, color=col, lw=1.3, label=LABEL[c])
                for fp in v["fixed_T2"]:
                    a_.plot(fp, fp, "o", color=col, ms=4, mfc="none" if c == "pulse_gated_plan" and
                            fp == min(v["fixed_T2"]) and len(v["fixed_T2"]) > 1 else col)
            a_.plot(F, F, color="0.3", lw=0.7, label="$\\Phi(F)=F$")
            a_.set_xscale("log")
            a_.set_yscale("log")
            a_.set_xlabel("information $F$ about the thermal scale of the inner module")
            a_.set_title("(a) new module" if age == "start" else "(b) mid-life", fontsize=8)
            a2 = a_.secondary_xaxis("top", functions=(lambda f: 1.645 / np.sqrt(np.maximum(f, 1e-9)),
                                                       lambda w: (1.645 / np.maximum(w, 1e-9)) ** 2))
            a2.set_xlabel("bound width $z/\\sqrt{F}$", fontsize=7)
            a2.tick_params(labelsize=6)
        ax[0].set_ylabel("information next interval, $\\Phi(F)$")
        ax[0].legend(fontsize=5.5, frameon=False, loc="upper left")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"g7_F7_nhap.{ext}", dpi=150)
    for tag in ("L_evidence", "L_boundary"):
        rows = load(tag)
        if not rows:
            continue
        # bound width against the fraction of each life (no survivor bias), median and 10-90 % band over lives
        fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
        edges = np.linspace(0, 1, 21)
        mid = (edges[:-1] + edges[1:]) / 2
        for c, col in (("op", "tab:blue"), ("op_bal", "tab:purple"), ("pulse_gated", "tab:red"),
                       ("pulse_gated_plan", "tab:orange"), ("pulse_gated_plan_min", "tab:green"),
                       ("pulse_always", "0.4")):
            for j, a_ in enumerate(ax):
                B = []
                for r in rows:
                    om = np.array(r["lives"][c]["trace"]["omega"])
                    if len(om) < 2:
                        continue
                    u = np.arange(len(om)) / (len(om) - 1)
                    k = np.minimum(np.digitize(u, edges) - 1, len(mid) - 1)
                    B.append([om[k == i, j].mean() if np.any(k == i) else np.nan for i in range(len(mid))])
                B = np.array(B)
                a_.plot(mid, np.nanmedian(B, 0), color=col, lw=1.2, label=LABEL[c] if j == 0 else None)
                if c in ("pulse_gated_plan", "pulse_gated_plan_min"):
                    a_.fill_between(mid, np.nanquantile(B, 0.1, 0), np.nanquantile(B, 0.9, 0), color=col,
                                    alpha=0.12, lw=0)
        for j, a_ in enumerate(ax):
            a_.set_yscale("log")
            a_.set_xlabel("fraction of the module life")
            a_.set_title(f"({'ab'[j]}) {'outer module' if j == 0 else 'inner module'}, {tag.split('_')[1]} truth", fontsize=8)
        ax[0].set_ylabel("bound width $z/\\sqrt{F}$")
        ax[0].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"g7_quy_dao_{tag.split('_')[1]}_nhap.{ext}", dpi=150)


if __name__ == "__main__":
    for tag in ("L_evidence", "L_boundary"):
        table_lives(tag)
        phase_summary(tag)
    table_lam()
    table_sens()
    figures()
    (HERE / "g7_tong_hop.json").write_text(json.dumps(OUT, indent=1), encoding="utf-8", newline="\n")
    (HERE / "g7_tong_hop.md").write_text("\n".join(MD) + "\n", encoding="utf-8", newline="\n")
