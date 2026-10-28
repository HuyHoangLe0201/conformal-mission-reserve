"""P3: final figures F2, F4, F5, F6, F7, F8 and F9 (closed loop against the maintenance baselines, added 17/10/2026)
from the saved results (no simulation).

Sources: g6_tong_hop.json (G6 and the open items G11), e3_cruise_base_delta.json and e3_coast_base_delta.json (G5, E3,
both driving strategies since 18/10/2026), g7_tong_hop.json and
g7_L_evidence.jsonl (G7). Style: kieu_hinh.py. Outputs: ../P3_hinh_bang/F*.pdf and .png.
F1 (diagram) is P3_hinh_bang/F1_cau_truc.tex; F3 (data) is ../P3_kiemtoan/hinh_cuoi_f3.py.

Run: python -I hinh_cuoi.py
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import kieu_hinh as kh  # noqa: E402
import numpy as np  # noqa: E402

G6 = json.loads((HERE / "g6_tong_hop.json").read_text(encoding="utf-8"))
G7 = json.loads((HERE / "g7_tong_hop.json").read_text(encoding="utf-8"))
E3 = json.loads((HERE / "e3_cruise_base_delta.json").read_text(encoding="utf-8"))
E3C = json.loads((HERE / "e3_coast_base_delta.json").read_text(encoding="utf-8"))


def f2():
    """Paired and union certificates against the bench sample size: coverage and tightness (alpha = 0.1)."""
    fig, ax = kh.figure(kh.W15, 58, 1, 2)
    certs = ("U6 paired", "U5 union", "U1 box", "U9 PAC")
    mark = {"U6 paired": "o", "U5 union": "s", "U1 box": "D", "U9 PAC": "v"}   # not by colour alone (19/10/2026)
    for tag, ls, truth in (("A_evidence_k0", "-", "evidence"), ("A_boundary_k0", "--", "at the margins")):
        r = G6[tag]
        n1s = sorted({int(k.split("|")[0]) for k in r if "stat_ok" not in k})
        for c in certs:
            col, name = kh.CERT[c]
            pts = [(n, r[f"{n}|0.1|{c}"]) for n in n1s if r[f"{n}|0.1|{c}"]["finite"] > 0.5]
            if not pts:
                continue
            x = [n for n, _ in pts]
            ax[0].plot(x, [v["coverage"] for _, v in pts], ls, color=col, marker=mark[c], ms=2.8)
            ax[1].plot(x, [v["ratio"] for _, v in pts], ls, color=col, marker=mark[c], ms=2.8)
        if tag == "A_evidence_k0":
            pts = [(n, r[f"{n}|0.1|U6 anchored"]) for n in n1s if f"{n}|0.1|U6 anchored" in r]
            ax[1].plot([n for n, _ in pts], [v["ratio"] for _, v in pts], ":", color=kh.BLUE, marker="^", ms=2.8)
    kh.refline(ax[0], 0.9, "nominal coverage $1-\\alpha$")
    ax[0].set_ylim(0.885, 1.005)
    ax[0].set_ylabel("coverage of the true damage")
    ax[1].set_yscale("log")
    ax[1].set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax[1].set_ylabel("certificate / oracle quantile")
    for a in ax:
        a.set_xscale("log")
        a.set_xticks([10, 20, 40, 80], ["10", "20", "40", "80"])
        a.minorticks_off()
        a.set_xlabel("bench scores $n_1$")
    h = [kh.plt.Line2D([], [], color=kh.CERT[c][0], marker=mark[c], ms=2.8, label=kh.CERT[c][1]) for c in certs]
    h += [kh.plt.Line2D([], [], color=kh.BLUE, ls=":", marker="^", ms=2.8, label=kh.CERT["U6 anchored"][1])]
    h += [kh.plt.Line2D([], [], color="k", ls="-", label="truth: evidence"),
          kh.plt.Line2D([], [], color="k", ls="--", label="truth: at the margins")]
    ax[0].legend(handles=h, loc="lower left", ncol=2, fontsize=6, bbox_to_anchor=(0.0, 0.17))
    kh.label(ax[0], "a")
    kh.label(ax[1], "b")
    fig.tight_layout(w_pad=1.5)
    kh.save(fig, "F2")


def f4():
    """Coverage per leg life against the feedback strength of the policy (300 lives per run, truth at the margins): each
    certificate at the mean derating level it reaches itself; filled: cruising, open: coasting."""
    f4 = G6["F4"]
    fig, ax = kh.figure(kh.W1, 62)
    meths = ("union", "paired", "split0", "reuse", "aci")
    mark = dict(zip(meths, ("s", "o", "D", "^", "v")))     # not by colour alone (19/10/2026)
    for st, fill in (("cruise", "full"), ("coast", "none")):
        for i, m in enumerate(meths):
            col, name = kh.METHOD[m]
            pts = sorted(((v["mean_level"] or 0.0, v) for k, v in f4.items()
                          if k.startswith(st + "|") and k.endswith(f"|{m}")), key=lambda q: q[0])
            if not pts:
                continue
            x = np.array([q for q, _ in pts]) + 0.005 * (i - 2)
            y = np.array([v["cov_life"] for _, v in pts])
            lo = np.array([v["ci"][0] for _, v in pts])
            hi = np.array([v["ci"][1] for _, v in pts])
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=mark[m], color=col, ms=2.8, lw=0.8, capsize=1.5,
                        elinewidth=0.6, fillstyle=fill, mew=0.7, label=name if st == "cruise" else None)
    kh.refline(ax, 0.9, "nominal coverage $1-\\alpha$", ha="left")
    ax.set_xlabel("mean derating level")
    ax.set_ylabel("leg lives covered at every decision")
    h, l = ax.get_legend_handles_labels()
    h += [kh.plt.Line2D([], [], color="0.3", marker="o", ls="", ms=2.8, label="cruising"),
          kh.plt.Line2D([], [], color="0.3", marker="o", ls="", ms=2.8, fillstyle="none", label="coasting")]
    ax.legend(handles=h, loc="lower right", fontsize=6, ncol=2)
    fig.tight_layout()
    kh.save(fig, "F4")


def f5():
    """Served intervals against the true damage at replacement (lives without failure, 300 lives per run): certificate
    policies and the baselines without derating, per driving strategy, for two truths."""
    c = G6["C"]
    fig, ax = kh.figure(kh.W15, 62, 1, 2)
    pmk = {"pass": "v", "fixed20": "s", "reserve": "o", "reserve_cap": "D", "reserve_budget": "P", "replace": "^"}
    pname = {"pass": "pass/fail", "fixed20": "fixed 20 %", "reserve": "reserve", "reserve_cap": "reserve, cap",
             "reserve_budget": "reserve, late budget", "replace": "replacement only"}
    base = {"none|tbr60": ("planned 60 intervals", "X"), "none|cert_age": ("certificate age at commissioning", "*"),
            "none|mc_age": ("Monte Carlo B10 age", "h")}
    for a, scen in zip(ax, ("evidence", "boundary")):
        for k, v in c.items():
            parts = k.split("|")
            if len(parts) != 3 or not parts[0].startswith(scen + "_") or v.get("harvest_ok") is None:
                continue
            fill = "none" if parts[0].endswith("coast") else "full"
            run = f"{parts[1]}|{parts[2]}"
            if parts[1] in pmk:
                col = kh.METHOD[parts[2]][0]
                a.plot(v["intervals_mean"], v["harvest_ok"], pmk[parts[1]], color=col, ms=3.2, mew=0.7,
                       fillstyle=fill, ls="none")
            elif run in base:
                a.plot(v["intervals_mean"], v["harvest_ok"], base[run][1], color=kh.BLACK, ms=3.6, mew=0.7,
                       fillstyle=fill, ls="none")
        a.set_xscale("log")
        lo, hi = a.get_xlim()
        tk = [t for t in (10, 15, 20, 30, 40, 60, 80) if lo <= t <= hi]
        a.set_xticks(tk, [str(t) for t in tk])
        a.minorticks_off()
        a.set_xlabel("served intervals per leg, mean")
        a.set_ylabel("true damage at replacement")
        a.set_title("truth: evidence" if scen == "evidence" else "truth: at the margins", fontsize=7.5)
    hm = [kh.plt.Line2D([], [], color=kh.METHOD[m][0], marker="o", ls="", ms=3, label=kh.METHOD[m][1])
          for m in ("union", "paired", "split0", "reuse", "aci", "mc", "bayes")]
    hp = [kh.plt.Line2D([], [], color="0.25", marker=mk, ls="", ms=3, label=pname[q]) for q, mk in pmk.items()]
    hb = [kh.plt.Line2D([], [], color=kh.BLACK, marker=mk, ls="", ms=3.4, label=nm) for nm, mk in base.values()]
    hs = [kh.plt.Line2D([], [], color="0.25", marker="o", ls="", ms=3, label="cruising (filled)"),
          kh.plt.Line2D([], [], color="0.25", marker="o", fillstyle="none", ls="", ms=3, label="coasting (open)")]
    fig.legend(handles=hm + hp + hb + hs, loc="lower center", ncol=6, fontsize=6, bbox_to_anchor=(0.5, -0.02))
    kh.label(ax[0], "a")
    kh.label(ax[1], "b")
    fig.tight_layout(rect=(0, 0.2, 1, 1), w_pad=1.5)
    kh.save(fig, "F5")


def f6():
    """Collapse by caution: (a) test weight against derating, cruising (solid) and coasting (dashed), without and with
    anchor tests; (b, c) certified damage of the leg (critical chip, median interval) against derating for margin slopes
    below, at and above the lifetime exponent, when coasting and when cruising."""
    fig, ax = kh.figure(kh.W2, 58, 1, 3)
    cols = {0: kh.VERMILION, 5: kh.ORANGE, 20: kh.BLUE}
    for e, ls, st in ((E3, "-", "cruising"), (E3C, "--", "coasting")):
        lv = [float(x) for x in e["levels"]]
        for n, col in cols.items():
            ax[0].plot([100 * d for d in lv], [e["a"][str(d)][f"p_test_anchors_{n}"] for d in lv], ls, marker="o",
                       color=col, ms=2.6, label=f"{n} anchor tests" if st == "cruising" else None)
    ax[0].axhline(0.05, color="k", ls=":", lw=0.6, label="threshold 0.05")
    ax[0].plot([], [], "-", color=kh.GREY, label="cruising")
    ax[0].plot([], [], "--", color=kh.GREY, label="coasting")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_xlabel("derating depth (%)")
    ax[0].set_ylabel("test weight")
    ax[0].legend(loc="center right", fontsize=6.5, bbox_to_anchor=(1.0, 0.6))
    gam = {"0.0": (kh.BLUE, ":"), "0.6": (kh.SKY, "-"), "1.2": (kh.GREEN, "-"), "2.08": (kh.ORANGE, "--"),
           "2.5": (kh.VERMILION, "-.")}
    lab = {"0.0": r"$\gamma$ = 0", "0.6": r"$\gamma$ = 0.6", "1.2": r"$\gamma$ = 1.2 (stated)",
           "2.08": r"$\gamma=\beta$ = 2.08", "2.5": r"$\gamma$ = 2.5"}
    for a_, e, title in ((ax[1], E3C, "coasting"), (ax[2], E3, "cruising")):
        lv = [float(x) for x in e["levels"]]
        for g, (col, ls) in gam.items():
            base = e["b"]["0.0"][g]
            a_.plot([100 * d for d in lv], [e["b"][str(d)][g] / base for d in lv], ls, color=col, marker="o", ms=2.6,
                    label=lab[g])
        a_.axhline(1, color=kh.GREY, lw=0.5)
        a_.set_yscale("log")
        a_.set_yticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
        a_.minorticks_off()
        a_.set_xlabel("derating depth (%)")
        a_.set_title(title, fontsize=7)
    ax[1].set_ylabel("certified damage / non-derated")
    ax[1].legend(loc="upper left", fontsize=6.5)
    lo = min(min(e["b"][str(float(d))][g] / e["b"]["0.0"][g] for d in e["levels"] for g in gam) for e in (E3, E3C))
    hi = max(max(e["b"][str(float(d))][g] / e["b"]["0.0"][g] for d in e["levels"] for g in gam) for e in (E3, E3C))
    for a_ in ax[1:]:
        a_.set_ylim(0.9 * lo, 1.1 * hi)
    for a_, letter in zip(ax, "abc"):
        kh.label(a_, letter)
    fig.tight_layout(w_pad=1.2)
    kh.save(fig, "F6")


def f7():
    """Information loop: phase portrait for a new module and at mid-life; bound width over the module life."""
    ph = G7["L_evidence|phase"]
    fig, ax = kh.figure(kh.W2, 62, 1, 3)
    conf = {"pulse_gated_plan": (kh.VERMILION, "-", "pulses gated by the reserve"),
            "pulse_gated_plan_min": (kh.GREEN, "-", "same, with dwell-window minimum"),
            "op": (kh.BLUE, "-", "operation only"), "op_bal": (kh.BLUE, "--", "operation only, balancing")}
    for a, age in zip(ax[:2], ("start", "mid")):
        F = None
        for c, (col, ls, name) in conf.items():
            v = ph[f"{age}|{c}"]
            F = np.array(v["F"])
            a.plot(F, np.array(v["phi"][1]), ls, color=col, lw=1.0, label=name)
            fps = v["fixed_T2"]
            for fp in fps:
                unstable = c == "pulse_gated_plan" and len(fps) > 1 and fp == min(fps)
                a.plot(fp, fp, "o", ms=3.5, color=col, mfc="white" if unstable else col, mew=0.8)
        a.plot(F, F, color=kh.GREY, lw=0.6)
        a.set_xscale("log")
        a.set_xticks([10, 100, 1000], ["10", "100", "1000"])
        a.minorticks_off()
        a.set_yscale("log")
        a.set_yticks([10, 100, 1000], ["10", "100", "1000"])
        a.set_xlabel(r"information $F$ about the thermal scale, inner module")
        a.text(0.97, 0.05, "new leg" if age == "start" else "mid-life", transform=a.transAxes, ha="right",
               va="bottom", fontsize=7.5)
        top = a.secondary_xaxis("top", functions=(lambda f: 1.645 / np.sqrt(np.maximum(f, 1e-9)),
                                                   lambda w: (1.645 / np.maximum(w, 1e-9)) ** 2))
        top.set_xticks([0.5, 0.2, 0.1, 0.05], ["0.5", "0.2", "0.1", "0.05"])
        top.minorticks_off()
        top.set_xlabel(r"bound width $z/\sqrt{F}$", fontsize=6.5)
        top.tick_params(labelsize=6)
    ax[0].set_ylabel(r"information next interval, $\Phi(F)$")
    ax[0].legend(loc="upper left", fontsize=5.8)
    # bound width over the module life (evidence truth), median and 10-90 % band
    rows = [json.loads(x) for x in (HERE / "g7_L_evidence.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    edges = np.linspace(0, 1, 21)
    mid = (edges[:-1] + edges[1:]) / 2
    for c, col, ls, name in (("pulse_gated_plan", kh.VERMILION, "-", "pulses gated by the reserve"),
                             ("pulse_gated_plan_min", kh.GREEN, "-", "same, with dwell-window minimum"),
                             ("op", kh.BLUE, "-", "operation only"), ("pulse_always", kh.GREY, "--", "pulses at full rate")):
        B = []
        for r in rows:
            om = np.array(r["lives"][c]["trace"]["omega"])
            if len(om) < 2:
                continue
            u = np.arange(len(om)) / (len(om) - 1)
            k = np.minimum(np.digitize(u, edges) - 1, len(mid) - 1)
            B.append([om[k == i, 1].mean() if np.any(k == i) else np.nan for i in range(len(mid))])
        B = np.array(B)
        ax[2].plot(mid, np.nanmedian(B, 0), ls, color=col, lw=1.0, label=name)
        if c in ("pulse_gated_plan", "pulse_gated_plan_min"):
            ax[2].fill_between(mid, np.nanquantile(B, 0.1, 0), np.nanquantile(B, 0.9, 0), color=col, alpha=0.15, lw=0)
    ax[2].set_yscale("log")
    ax[2].set_yticks([0.05, 0.1, 0.2, 0.5, 1.0], ["0.05", "0.1", "0.2", "0.5", "1"])
    ax[2].minorticks_off()
    ax[2].set_xlabel("fraction of the leg life")
    ax[2].set_ylabel("bound width, inner module")
    ax[2].legend(loc="upper left", fontsize=5.8)
    for a, l in zip(ax, "abc"):
        kh.label(a, l, x=-0.2, y=1.0 if l in "ab" else 1.02)
    fig.tight_layout(w_pad=1.2)
    kh.save(fig, "F7")


def f8():
    """Exceedance of the stated heating-time margin: coverage and tightness, with and without anchor tests."""
    x = G6["X"]
    gs = sorted(x, key=float)
    g = [float(v) for v in gs]
    fig, ax = kh.figure(kh.W15, 56, 1, 2)
    ax[0].plot(g, [x[v]["cov_stated"] for v in gs], "-o", color="k", ms=2.8, label="stated heating-time rule")
    ax[1].plot(g, [x[v]["ratio_stated"] for v in gs], "-o", color="k", ms=2.8, label="stated heating-time rule")
    for k, col, fmt in zip(x[gs[0]]["cov_anchor"], (kh.BLUE, kh.GREEN), ("--s", "-^")):
        ts = k.split("|")[0].replace("+", " and ")
        lab = f"anchor tests at {ts} s"
        ax[0].plot(g, [x[v]["cov_anchor"][k] for v in gs], fmt, color=col, ms=2.8, label=lab, mfc="none")
        ax[1].plot(g, [x[v]["ratio_anchor"][k] for v in gs], fmt, color=col, ms=2.8, label=lab, mfc="none")
    kh.refline(ax[0], 0.9, "nominal coverage $1-\\alpha$", va="top")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("coverage of the true damage")
    kh.refline(ax[1], 1.0, "certificate = true damage")
    ax[1].set_yscale("log")
    ax[1].set_yticks([0.3, 0.5, 1, 2, 4], ["0.3", "0.5", "1", "2", "4"])
    ax[1].minorticks_off()
    ax[1].set_ylabel("certificate / true damage, median")
    for a in ax:
        a.set_xlabel(r"true heating-time slope $g$ (stated 0.5)")
    ax[0].legend(loc="lower left", fontsize=6)
    kh.label(ax[0], "a")
    kh.label(ax[1], "b")
    fig.tight_layout(w_pad=1.5)
    kh.save(fig, "F8")


F9_GROUPS = [("boundary_cruise", "cruise"), ("boundary_coast", "coast"), ("evidence_cruise", "cruise"),
             ("evidence_coast", "coast"), ("adverse_strict_cruise", "cruise"), ("adverse_strict_coast", "coast"),
             ("beyond_cruise", "cruise"), ("beyond_coast", "coast")]
F9_TRUTH = {"boundary": "margins", "evidence": "evidence", "adverse_strict": "adverse", "beyond": "beyond"}


def f9():
    """Closed loop against the maintenance baselines (300 common lives per truth and driving strategy): cost per 100
    intervals for a failure costing r = 10 and r = 50 planned replacements (lateness not priced), and failed lives."""
    c = G6.get("C", {})
    groups = [(g, lab) for g, lab in F9_GROUPS if any(k.startswith(g + "|") for k in c)]
    if not groups:
        print("F9: no closed-loop results yet")
        return
    fig, ax = kh.figure(kh.W2, 78, 1, 3)
    runs = list(kh.RUN)
    x0 = np.arange(len(groups))
    for i, run in enumerate(runs):
        col, name, mk, _ = kh.RUN[run]
        dx = 0.62 * (i / max(len(runs) - 1, 1) - 0.5)
        for a_, what in zip(ax, ("10|0.0", "50|0.0", "failed")):
            ys = []
            for g, _ in groups:
                v = c.get(f"{g}|{run}")
                ys.append(np.nan if v is None else (100 * v["failed"] if what == "failed" else v["cost_per_100"][what]))
            a_.plot(x0 + dx, ys, mk, color=col, ms=3.2, mew=0.7, ls="none", label=name if what == "10|0.0" else None,
                    fillstyle="none" if run.startswith("none|") else "full")
    for a_, title in zip(ax, ("cost per 100 intervals\n(failure cost 10)", "cost per 100 intervals\n(failure cost 50)",
                              "failed lives (%)")):
        a_.set_xticks(x0, [lab for _, lab in groups], fontsize=5.6)
        a_.set_xlim(-0.6, len(groups) - 0.4)
        a_.set_title(title, fontsize=7)
        truths = [g.rsplit("_", 1)[0] for g, _ in groups]
        for k in range(1, len(groups)):
            a_.axvline(k - 0.5, color="0.7" if truths[k] != truths[k - 1] else "0.9", lw=0.5, zorder=0)
        for t in dict.fromkeys(truths):
            xs = [k for k, u in enumerate(truths) if u == t]
            a_.text(sum(xs) / len(xs), -0.13, F9_TRUTH.get(t, t), transform=a_.get_xaxis_transform(), ha="center",
                    va="top", fontsize=6)
    for a_ in ax[:2]:
        a_.set_yscale("log")
        lo, hi = a_.get_ylim()
        tk = [v for v in (0.5, 1, 2, 5, 10, 20, 50, 100, 200) if lo <= v <= hi]
        a_.set_yticks(tk, [f"{v:g}" for v in tk])
        a_.minorticks_off()
    ax[2].set_yscale("symlog", linthresh=1.0)
    ax[2].set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    ax[2].set_ylim(-0.3, 150)
    fig.legend(loc="lower center", ncol=5, fontsize=6, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0,
               handletextpad=0.3)
    for a_, letter in zip(ax, "abc"):
        kh.label(a_, letter)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.15, 1, 1))
    kh.save(fig, "F9")


if __name__ == "__main__":
    for f in (f2, f4, f5, f6, f7, f8, f9):
        f()
