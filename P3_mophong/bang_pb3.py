"""P3, third review (19/10/2026): Tables S12-S15 of the Supplementary Material from tong_hop_pb3.json.

  S12  paired certificate against union certificates with a small mission level (U)
  S13  truths outside the family of the stated margin (Q)
  S14  mission change within a life (M)
  S15  closed loop at alpha = 0.05 (C_boundary_*_a05)
  S19  baseplate-solder fatigue (han_de.json and the S runs)
  S20  multi-split paired certificate (the K runs and chia_nhieu.json)
  S21  coverage of the statistical part of the margin (the W run)
  S22  out-of-sample check of the heating-time rule (../P3_kiemtoan/g27_ton_ngoai_mau.json)

Run after tong_hop_pb3.py: python bang_pb3.py -> ../P3_RESS/TS12.tex ... TS15.tex
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "P3_RESS"
J = json.loads((HERE / "tong_hop_pb3.json").read_text(encoding="utf-8"))
TRUTH = {"boundary": "at the margins", "evidence": "evidence"}


def write(name, text):
    (OUT_DIR / name).write_text(text, encoding="utf-8", newline="\n")
    print("written", OUT_DIR / name)


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


# ------------------------------------------------------------------------------------------------------------ S12
def size_cell(v):
    if v is None:
        return "--"
    if v["finite"] < 0.5 or not math.isfinite(v["size_vs_paired"]):
        return r"$\infty$"
    return f"{v['size_vs_paired']:.2f}"


def ts12():
    U = J["U"]
    rows = []
    for sc in ("boundary", "evidence"):
        for n1 in ("10", "20", "40", "80"):
            cells = []
            for a in ("0.1", "0.05"):
                pv = U.get(f"{sc}|{n1}|{a}|paired")
                cells.append(r"$\infty$" if pv is None or pv["finite"] < 0.5 else "1")
                for name in ("union a/2", "union aV=0.02", "union aV=0.01", "union aV=0.005"):
                    cells.append(size_cell(U.get(f"{sc}|{n1}|{a}|{name}")))
            rows.append(f"{TRUTH[sc] if n1 == '10' else ''} & {n1} & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    body = "\n".join(rows[:-1])
    cov = [v["cov_Unew"] for k, v in U.items() if k.endswith("|paired") and v["finite"] >= 0.5 and k.split("|")[2] == "0.1"]
    return r"""\begin{table}[htbp]
\centering
\caption{Paired certificate against union certificates whose levels are allocated in advance, $\alpha_S+\alpha_V=\alpha$
(200 repetitions per truth, $n_2=100$, both modules): median size of each union certificate relative to the paired
one; $\infty$: infinite in most repetitions. The equal split is $\alpha_V=\alpha/2$. With $n_2=100$, $\alpha_V=0.005$
is never finite, and $\alpha_V=0.01$ takes the largest recorded interval. $n_1$: number of bench scores, from a bench
of about $2n_1$ modules split in half. Coverage of the paired score $U_{\mathrm{new}}$ by the paired certificate at $\alpha=0.10$: """ + \
        f"{min(cov):.3f}--{max(cov):.3f}" + r""".}
\label{tab:S12}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccccccc@{}}
\toprule
& & \multicolumn{5}{c}{$\alpha=0.10$} & \multicolumn{5}{c}{$\alpha=0.05$} \\
\cmidrule(lr){3-7}\cmidrule(l){8-12}
Truth & $n_1$ & paired & $\alpha/2$ & $0.02$ & $0.01$ & $0.005$ & paired & $\alpha/2$ & $0.02$ & $0.01$ & $0.005$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


# ------------------------------------------------------------------------------------------------------------ S13
QCERT = (("U1 box", "box"), ("U2 Monte Carlo", "Monte Carlo"), ("U3 Bayes", "Bayes"), ("U5 union", "union"),
         ("U6 paired", "paired"), ("U9 PAC", "PAC"))
QTAG = (("Q_boundary_int", "margins, interaction"), ("Q_evidence_int", "evidence, interaction"),
        ("Q_boundary_lutz", "margins, steep low-swing lives"), ("Q_boundary_nc", "margins, strength not common"),
        ("Q_boundary_th", "margins, thermal error growing"), ("Q_boundary_seq", "margins, Miner factor of the mix"))


def ts13():
    Q = J["Q"]
    rows = []
    for tag, lab in QTAG:
        for n1 in ("20", "40"):
            cells = []
            for key, _ in QCERT:
                v = Q.get(f"{tag}|{n1}|0.1|{key}")
                cells.append("--" if v is None else f"{v['cov']:.3f}")
            cal = Q.get(f"{tag}|{n1}|0.1|calibrated")
            sz = Q.get(f"{tag}|{n1}|0.1|U6 paired")
            cells.append("--" if cal is None else f"{cal['cov']:.3f}")
            cells.append("--" if sz is None else f"{sz['size_vs_oracle']:.2f}")
            rows.append(f"{lab if n1 == '20' else ''} & {n1} & " + " & ".join(cells) + r" \\")
    return r"""\begin{table}[htbp]
\centering
\caption{Truths outside the family of the stated margin ($n_2=100$, $\alpha=0.10$, 200 repetitions, both modules):
coverage of the true damage by each certificate (an infinite certificate covers), coverage of the calibrated part of
the paired certificate, and median size of the paired certificate relative to the oracle quantile. Interaction: below
26~K the heating-time effect above 4.5~s steepens with the swing, the life falling by $c\,\log(26/\Delta T)\,\log(t_{\mathrm{on}}/4.5)$
in log units, $c$ uniform on 0.10--0.30, which no slope of the stated margin can follow. Steep low-swing lives: below 26~K
lives rise faster than a power law, by $c\,\log^2(26/\Delta T)$ in log units, $c$ uniform on 1--3. Strength not common:
the log-strength of the field module at field conditions differs from its reference strength by a normal deviation
of standard deviation 0.05--0.15. Thermal error growing: the true swing of a cycle is the scaled one times
$1+c\,\log(t_{\mathrm{on}}/4.5)$ above 4.5~s, $c$ uniform on 0.02--0.05, as when the slow branches of the thermal path
degrade. Miner factor of the mix: the Miner factor falls to $\kappa(1-c f)$, $f$ the share of the true damage carried by
cycles below 26~K, $c$ uniform on 0.1--0.3. Under the evidence truth the last three cover 1.000 at both bench sizes,
with calibrated parts of 0.92--0.95. Monte Carlo standard error of a coverage near 0.90: about 0.02.}
\label{tab:S13}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccccc@{}}
\toprule
Truth & $n_1$ & box & Monte Carlo & Bayes & union & paired & PAC & \makecell{paired,\\calibrated} &
\makecell{paired /\\oracle} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


# ------------------------------------------------------------------------------------------------------------ S14
MRUNS = (("cert|paired", "paired certificate"), ("cert|paired stale", "paired, not re-simulated"),
         ("cert|union", "union certificate"), ("cert|mc", "Monte Carlo certificate"),
         ("none|tbr60", "planned 60 intervals"), ("none|rtf", "run to failure"),
         ("none|cert_age", "certificate age"), ("replan|cert_age", "\\quad replanned at the change"),
         ("none|mc_age", "Monte Carlo B10 age"), ("replan|mc_age", "\\quad replanned at the change"),
         ("none|mc_opt10", "Monte Carlo optimal age"), ("replan|mc_opt10", "\\quad replanned at the change"))
MTAG = ("M_boundary_cruise_coast", "M_boundary_coast_cruise", "M_evidence_cruise_coast", "M_evidence_coast_cruise")


def ts14():
    M = J["M"]
    rows = []
    for run, lab in MRUNS:
        cells = []
        for tag in MTAG:
            v = M.get(tag, {}).get("runs", {}).get(run)
            cells += ["--"] * 3 if v is None else [f"{v['intervals']:.1f}", pct(v["failed"]), f"{v['cost10']:.2f}"]
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if run == "cert|mc":
            rows.append(r"\addlinespace")
    return r"""\begin{table}[htbp]
\centering
\caption{Mission change within a leg life (300 lives per configuration, bench of 60 modules, $n_2=50$, $\alpha=0.10$, no
derating): the driving strategy changes at an interval drawn uniformly in 10--30, on the same exogenous days. Served
intervals, failed lives (\%) and cost per 100 intervals for a failure costing $c_F=10$ planned replacements. The
certificates replace the leg when either module fails on the policy data and re-simulate the recorded days under the
strategy in force (not re-simulated: under the first strategy); static ages are set at commissioning for the first
strategy; replanned ages are recomputed at the change, for the second strategy, from the margined damage accrued so far
(paired reserve on all bench scores, or the Monte Carlo law conditioned on survival), and act only on legs still in
service. Wilson 95\,\% intervals of a failure share of 1\,\%: 0.3--2.9\,\%.}
\label{tab:S14}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccccc@{}}
\toprule
& \multicolumn{6}{c}{At the margins} & \multicolumn{6}{c}{Evidence} \\
\cmidrule(lr){2-7}\cmidrule(l){8-13}
& \multicolumn{3}{c}{cruising, then coasting} & \multicolumn{3}{c}{coasting, then cruising}
& \multicolumn{3}{c}{cruising, then coasting} & \multicolumn{3}{c}{coasting, then cruising} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(l){11-13}
Run & served & failed & cost & served & failed & cost & served & failed & cost & served & failed & cost \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


# ------------------------------------------------------------------------------------------------------------ S15
CRUNS = (("reserve|paired", "paired, reserve rule"), ("reserve|union", "union, reserve rule"),
         ("replace|paired", "paired, replacement only"), ("replace|union", "union, replacement only"),
         ("none|tbr60", "planned 60 intervals"), ("none|rtf", "run to failure"),
         ("none|cert_age", "certificate age"), ("none|mc_age", "Monte Carlo B5 age"),
         ("none|mc_opt10", "Monte Carlo optimal age, $c_F=10$"))


def ts15():
    C = J["C05"]
    rows = []
    for run, lab in CRUNS:
        cells = []
        for st in ("cruise", "coast"):
            v = C.get(f"C_boundary_{st}_a05", {}).get(run)
            if v is None:
                cells += ["--"] * 5
                continue
            cells += [f"{v['intervals']:.1f}", pct(v["failed"]),
                      "--" if v["coverage"] is None else f"{v['coverage']:.3f}",
                      "--" if v["late"] is None else pct(v["late"]), f"{v['cost10']:.2f}"]
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if run == "replace|union":
            rows.append(r"\addlinespace")
    return r"""\begin{table}[htbp]
\centering
\caption{Closed loop at $\alpha=0.05$ (truth at the margins, 300 lives per driving strategy, bench of 120 modules,
half of them fitting the model, 30 scores for the policy and 30 for the paired certificate, $n_2=50$): served intervals, failed lives (\%),
coverage per decision, late sections (\%) and cost per 100 intervals for $c_F=10$, late sections not priced. The
static ages use the same level: the Monte Carlo age is the B5 life.}
\label{tab:S15}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccc@{}}
\toprule
& \multicolumn{5}{c}{Cruising} & \multicolumn{5}{c}{Coasting} \\
\cmidrule(lr){2-6}\cmidrule(l){7-11}
Run & served & failed & coverage & late & cost & served & failed & coverage & late & cost \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


# ------------------------------------------------------------------------------------------------------------ S16
DTAG = (("A_boundary_k0", "at the margins"), ("A_adverse_strict_k0", "adverse"), ("A_beyond_k0", "beyond"),
        ("Q_boundary_int", "margins, interaction"), ("Q_boundary_lutz", "margins, steep low-swing lives"),
        ("Q_boundary_nc", "margins, strength not common"), ("Q_boundary_th", "margins, thermal error growing"),
        ("Q_boundary_seq", "margins, Miner factor of the mix"))


def ts16():
    D = json.loads((HERE / "chan_doan_am.json").read_text(encoding="utf-8"))
    rows = []
    for tag, lab in DTAG:
        for n1 in ("20", "40", "80"):
            v = D["misses"].get(f"{tag}|{n1}|0.1")
            if v is None:
                continue
            rows.append(f"{lab if n1 == '20' else ''} & {n1} & {pct(v['miss'])} & {pct(v['violated'])} & "
                        f"{pct(v['held_miss'])} \\\\")
    c = D["calibrated"]
    return r"""\begin{table}[htbp]
\centering
\caption{Origin of the misses of the paired certificate ($n_2=100$, $\alpha=0.10$, 200 repetitions, both modules).
The paired score $U_{\mathrm{new}}$ of the field module at its true strength bounds the true damage exactly when the
stated components hold for that module, so a miss ($D^\star>C_P$) either violates a stated component ($D^\star>
U_{\mathrm{new}}$) or is a calibrated miss with the stated components holding; Theorem~\ref{thm:p3-pair} bounds only the latter, by
$\alpha$. Under the evidence truth, and its variant with the interaction, no miss and no violation occur. Across """ + \
        f"{c['cells']}" + r""" cells of the static experiments (Tables~\ref{tab:p3-T3}, \ref{tab:S3}, \ref{tab:S12} and~\ref{tab:S13}, both levels), the
coverage of the calibrated part deviates from the exact level $\lceil(1-\alpha)(m+1)\rceil/(m+1)$ by """ + \
        f"{c['mean_z']:+.2f}" + r""" standard errors on average (standard deviation """ + f"{c['sd_z']:.2f}" + r"""), with """ + \
        f"{c['below']}" + r""" cells below $-2$ and """ + f"{c['above']}" + r""" above $+2$, against about """ + \
        f"{0.023 * c['cells']:.0f}" + r""" on each side by chance: the calibrated part is exact or slightly conservative.}
\label{tab:S16}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccc@{}}
\toprule
Truth & $n_1$ & misses (\%) & \makecell{stated component\\violated (\%)} & \makecell{misses with the stated\\components holding (\%)} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


# ------------------------------------------------------------------------------------------------------------ S17
def ts17():
    P = json.loads((HERE / "phan_ra_hu_hai.json").read_text(encoding="utf-8"))
    cr, co = P["cruise"], P["coast"]
    rc, ro = cr["chips"][cr["critical_chip"]], co["chips"][co["critical_chip"]]

    def lab(s):
        return s.replace(">= ", r"$\geq$").replace("-", "--").replace(" K", "~K").replace(" s", "~s")

    rows = [r"\multicolumn{5}{@{}l}{\textit{swing}} \\"]
    for k in rc["swing"]:
        rows.append(f"{lab(k)} & {pct(rc['swing'][k][0])} & {pct(rc['swing'][k][1])} & {pct(ro['swing'][k][0])} & "
                    f"{pct(ro['swing'][k][1])} \\\\")
    rows.append(r"\addlinespace")
    rows.append(r"\multicolumn{5}{@{}l}{\textit{heating time}} \\")
    for k in rc["heating"]:
        rows.append(f"{lab(k)} & {pct(rc['heating'][k][0])} & {pct(rc['heating'][k][1])} & {pct(ro['heating'][k][0])} & "
                    f"{pct(ro['heating'][k][1])} \\\\")
    rows.append(r"\addlinespace")
    rows.append(f"damage per day ($10^{{-5}}$) & {1e5 * rc['bare_per_day']:.2f} & {1e5 * rc['margined_per_day']:.1f} & "
                f"{1e5 * ro['bare_per_day']:.2f} & {1e5 * ro['margined_per_day']:.1f} \\\\")
    rows.append(f"largest cycle of each day (\\%) & & {pct(rc['share_largest_cycle_of_day'])} & & "
                f"{pct(ro['share_largest_cycle_of_day'])} \\\\")
    return r"""\begin{table}[htbp]
\centering
\caption{Where the damage comes from: shares (\%) of the damage of the critical chip by swing and by heating time of
its cycles, over all 4215 vehicle days without derating (diode D1 of the outer module when cruising, about """ + \
        f"{rc['cycles_per_day']:.0f}" + r""" cycles a day; IGBT T1 when coasting, about """ + f"{ro['cycles_per_day']:.0f}" + \
        r""" a day). Bare: lifetime model fitted on all 34 Hernes modules, nominal thermal model, no stated margin.
Margined: upper thermal scale 1.10 and the stated margin without its statistical part, which scales whole intervals.
Pulses shorter than 0.15~s are out of scope. Bins use the nominal swing and heating time. Last row: share of the
margined damage carried by the most damaging cycle of each day.}
\label{tab:S17}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
& \multicolumn{2}{c}{Cruising (D1)} & \multicolumn{2}{c}{Coasting (T1)} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
& bare & margined & bare & margined \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


# ------------------------------------------------------------------------------------------- Table 5 (manuscript)
T6_ROWS = (("replace|paired", "cert|paired", "paired certificate, replacement"),
           ("replace|union", "cert|union", "union certificate, replacement"),
           ("reserve|paired", None, "paired certificate, reserve rule$^{*}$"),
           ("none|tbr60", "none|tbr60", "planned 60 intervals"),
           ("none|cert_age", "none|cert_age", "certificate age at commissioning"),
           (None, "replan|cert_age", r"\quad replanned at the change"),
           ("none|mc_age", "none|mc_age", "Monte Carlo B10 age"),
           ("none|mc_opt10", "none|mc_opt10", "Monte Carlo optimal age"),
           ("none|rtf", "none|rtf", "run to failure"))
T6_GROUPS = (("evidence_cruise", "evidence, cruising"), ("boundary_cruise", "margins, cruising"),
             ("boundary_coast", "margins, coasting"), ("beyond_cruise", "beyond, cruising"))


def t6():
    G = json.loads((HERE / "g6_tong_hop.json").read_text(encoding="utf-8"))["C"]
    M = J["M"]["M_boundary_cruise_coast"]["runs"]
    rows = []
    for c_key, m_key, lab in T6_ROWS:
        cells = []
        for g, _ in T6_GROUPS:
            v = G.get(f"{g}|{c_key}")
            cells += ["--", "--"] if v is None else [pct(v["failed"]), f"{v['cost_per_100']['10|0.0']:.2f}"]
        v = M.get(m_key) if m_key else None
        cells += ["--", "--"] if v is None else [pct(v["failed"]), f"{v['cost10']:.2f}"]
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if c_key == "reserve|paired":
            rows.append(r"\addlinespace")
    late = G["boundary_coast|reserve|paired"]["late_pooled"]
    return r"""\begin{table}[pos=!tbp]
\centering
\caption{Maintenance decisions compared on common lives (evidence truth, truth at the margins and beyond, 300 lives
each, $\alpha=0.10$):
failed lives (\%; one life is 0.33\,\%) and cost per 100 intervals (8.2 years) in planned replacements for a failure
costing $c_F=10$, late sections not priced; run to failure is censored at 240 intervals (20 years). Mission change:
cruising, then coasting from an interval drawn in 10--30, without derating, with a bench of 60 modules and 50
recorded intervals (Table~S14).}
\label{tab:p3-T6}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccc@{}}
\toprule
& \multicolumn{2}{c}{\makecell{Evidence,\\cruising}} & \multicolumn{2}{c}{\makecell{Margins,\\cruising}} &
\multicolumn{2}{c}{\makecell{Margins,\\coasting}} & \multicolumn{2}{c}{\makecell{Beyond,\\cruising}}
& \multicolumn{2}{c}{\makecell{Mission change,\\margins}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}
Decision rule & failed & cost & failed & cost & failed & cost & failed & cost & failed & cost \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}

\smallskip
\begin{minipage}{\textwidth}\scriptsize
$^{*}$With derating; when coasting it makes """ + f"{100 * late:.0f}" + r"""\,\% of the sections late. Static ages are set at commissioning
for the first mission; the replanned age is recomputed at the change, assumed detected, from the margined damage
accrued so far. Wilson 95\,\% intervals: 0.7\,\% (2 lives) 0.2--2.4\,\%, 1.0\,\% 0.3--2.9\,\%, so they do not
differ. Other truths, policies and costs: Tables~S7 and~S14.
\end{minipage}
\end{table}
"""


# ------------------------------------------------------------------------------------------------------------ S18
FRUNS = (("paired", "paired certificate"), ("union", "union certificate"), ("pac", "PAC certificate"),
         ("tbr60", "planned 60 intervals"), ("cert_age", "certificate age at commissioning"),
         ("mc_age", "Monte Carlo B10 age"), ("mc_opt10", "Monte Carlo optimal age"))
FTAG = (("F_boundary_cruise", "margins, cruising"), ("F_boundary_coast", "margins, coasting"),
        ("F_beyond_cruise", "beyond, cruising"))
CRUNS40 = (("reserve|paired", "paired, reserve rule"), ("reserve|union", "union, reserve rule"),
           ("reserve|pac", "PAC, reserve rule"), ("replace|paired", "paired, replacement only"),
           ("replace|union", "union, replacement only"), ("replace|pac", "PAC, replacement only"),
           ("none|tbr60", "planned 60 intervals"), ("none|mc_opt10", "Monte Carlo optimal age"))


def ts18():
    F, C = J.get("F", {}), J.get("C40", {})
    nl = next(iter(F.values()))["legs"] if F else 20
    nf = next(iter(F.values()))["fleets"] if F else 100
    rows_a = []
    for run, lab in FRUNS:
        cells = []
        for tag, _ in FTAG:
            v = F.get(tag, {}).get("runs", {}).get(run)
            cells += ["--"] * 4 if v is None else [pct(v["failed"]), f"{100 * v['q90']:.0f}", f"{100 * v['max']:.0f}",
                                                    f"{v['cost10']:.2f}"]
        rows_a.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if run == "pac":
            rows_a.append(r"\addlinespace")
    rows_b = []
    for run, lab in CRUNS40:
        cells = []
        for st in ("cruise", "coast"):
            v = C.get(f"C_boundary_{st}_n40", {}).get(run)
            cells += ["--"] * 4 if v is None else [f"{v['intervals']:.1f}", pct(v["failed"]),
                                                    "--" if v["coverage"] is None else f"{v['coverage']:.3f}",
                                                    f"{v['cost10']:.2f}"]
        rows_b.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if run == "replace|pac":
            rows_b.append(r"\addlinespace")
    return r"""\begin{table}[htbp]
\centering
\caption{Fleets and the PAC certificate ($\alpha=0.10$, bench of 120 modules, half of them fitting the model, 30 or 31 scores per side,
$n_2=50$). (a) """ + f"{nf}" + r""" fleets of """ + f"{nl}" + r""" legs, each fleet sharing one bench and one set of
recorded intervals, without derating: failed legs (\%) averaged over the fleets, at the 90\,\% quantile over the
fleets (interpolated) and in the worst fleet, and cost per 100 intervals for $c_F=10$. (b) Closed loop at the margins, 300 lives per driving
strategy, each with its own bench: served intervals, failed lives (\%), coverage per decision and cost for $c_F=10$,
late sections not priced. The PAC certificate holds its level for the bench at hand with probability $1-\eta=0.90$
(Section~\ref{sec:samplesize}); with 30--31 scores it takes the largest paired score.}
\label{tab:S18}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\textit{(a) Fleets sharing one bench}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccccc@{}}
\toprule
& \multicolumn{4}{c}{Margins, cruising} & \multicolumn{4}{c}{Margins, coasting} & \multicolumn{4}{c}{Beyond, cruising} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-13}
Run & mean & 90\,\% & worst & cost & mean & 90\,\% & worst & cost & mean & 90\,\% & worst & cost \\
\midrule
""" + "\n".join(rows_a) + r"""
\bottomrule
\end{tabular*}

\medskip
\textit{(b) Closed loop with the PAC certificate}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}
\toprule
& \multicolumn{4}{c}{Cruising} & \multicolumn{4}{c}{Coasting} \\
\cmidrule(lr){2-5}\cmidrule(l){6-9}
Run & served & failed & coverage & cost & served & failed & coverage & cost \\
\midrule
""" + "\n".join(rows_b) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


# ------------------------------------------------------------------------------------------------------------ S19
SVAR = ("nominal, AlSiC", "with dwell, AlSiC", "stacked with dwell, AlSiC", "stacked with dwell, Cu")
STAGS = (("S_boundary_cruise", "at the margins", "cruising"), ("S_boundary_coast", "", "coasting"),
         ("S_evidence_cruise", "evidence", "cruising"), ("S_evidence_coast", "", "coasting"))


def sig(x):
    return f"{x:.3f}" if x < 0.1 else f"{x:.2f}"


def ts19_hidden(S):
    """Junction-only coverage cells not shown in Table S19(b) (union certificate; paired at n1 = 10 and at alpha =
    0.05 and 0.01) that the solder channel changes, and the copper cells below the nominal level."""
    n_cells = 0
    hit = {b: [] for b in ("AlSiC", "Cu")}
    for tag, _, _ in STAGS:
        for key, v in S.items():
            parts = key.split("|")
            if parts[0] != tag or len(parts) != 4 or parts[1] == "two":
                continue
            _, n1, a, cert = parts
            if cert == "U6 paired" and a == "0.1" and n1 != "10":
                continue
            n_cells += 1
            for b in hit:
                if abs(v[b][0] - v["junction"][0]) > 1e-12:
                    hit[b].append(v["junction"][0] - v[b][0])
    parts = []
    for b, name in (("AlSiC", "AlSiC"), ("Cu", "copper")):
        if hit[b]:
            parts.append(f"{len(hit[b])} with {name}, lowering them by at most {max(hit[b]):.3f}")
        else:
            parts.append(f"none with {name}")
    txt = (f"Of the other {n_cells} junction-only coverages (union certificate; paired certificate at " +
           r"$n_1=10$ and at $\alpha=0.05$ and 0.01), the solder channel changes " + " and ".join(parts) + ".")
    # copper cells below the nominal level 1 - alpha
    low = []
    for tag, _, st in STAGS:
        for key, v in S.items():
            parts = key.split("|")
            if parts[0] != tag or len(parts) != 4 or parts[1] == "two":
                continue
            _, n1, a, cert = parts
            if v["Cu"][0] < 1 - float(a) - 1e-9:
                low.append((tag, st, n1, a, cert, v["Cu"][0]))
    if low:
        tags = sorted({x[0] for x in low})
        where = " and ".join(("at the margins " if t.startswith("S_boundary") else "under the evidence truth ")
                             + ("when coasting" if t.endswith("coast") else "when cruising") for t in tags)
        a05 = [x[5] for x in low if x[3] == "0.05"]
        a10 = [x[5] for x in low if x[3] == "0.1"]
        bits = []
        if a10:
            bits.append(r"$\alpha=0.10$: " + rng_txt(a10))
        if a05:
            bits.append(r"$\alpha=0.05$: " + rng_txt(a05))
        txt += (f" With copper, {len(low)} junction-only coverages, all {where}, fall below the nominal level ("
                + "; ".join(bits) + ").")
    return txt


def rng_txt(vals, d=3):
    lo, hi = min(vals), max(vals)
    return f"{lo:.{d}f}" if f"{lo:.{d}f}" == f"{hi:.{d}f}" else f"{lo:.{d}f}--{hi:.{d}f}"


def ts19():
    H = json.loads((HERE / "han_de.json").read_text(encoding="utf-8"))
    S = J["S"]
    rows_a = []
    for st, lab in (("cruise", "Cruising"), ("coast", "Coasting")):
        for mname in ("outer", "inner"):
            h = H[st][mname]
            v = h["variants"]
            cells = [f"{h['stats']['swing_daily_max_median']:.1f}", f"{h['stats']['ton_median_of_largest']:,.0f}",
                     f"{v['nominal, AlSiC']['life_years']:,.0f}"]
            cells = [c.replace(",", r"\,") for c in cells]
            cells += [sig(v[k]["vs_margined"]) for k in SVAR]
            cells.append(sig(v["stacked with dwell, AlSiC"]["vs_bare"]))
            rows_a.append(f"{lab if mname == 'outer' else ''} & {mname} & " + " & ".join(cells) + r" \\")
    rows_b, mx, gov_al, gov_cu = [], [], [], []
    for tag, lab, st in STAGS:
        for n1 in ("20", "40", "80"):
            v = S[f"{tag}|{n1}|0.1|U6 paired"]
            w = S[f"{tag}|two|{n1}|U6 paired"]
            cells = [f"{v['junction'][0]:.3f}", f"{v['AlSiC'][0]:.3f}", f"{v['Cu'][0]:.3f}", f"{w['cov'][0]:.3f}",
                     f"{w['size']['AlSiC']:.2f}", f"{w['size']['Cu']:.2f}"]
            rows_b.append(f"{lab if n1 == '20' else ''} & {st if n1 == '20' else ''} & {n1} & " + " & ".join(cells)
                          + r" \\")
        if tag == "S_boundary_coast":
            rows_b.append(r"\midrule")
        r = S[f"{tag}|ratio"]
        mx.append(pct(r["max"], 0))
        gov_al.append(pct(r["governs"]["AlSiC"]))
        gov_cu.append(pct(r["governs"]["Cu"]))
    # union certificate, coasting at the margins with copper: junction-only and two certificates
    un1 = [S[f"S_boundary_coast|{n1}|0.1|U5 union"]["Cu"][0] for n1 in ("20", "40", "80")]
    # the pair of union certificates only where the union certificate at alpha/2 is finite (n1 = 20 gives too few
    # scores at 0.025 per side: infinite, so the pair would cover as the solder certificate alone)
    un2_n1 = [n1 for n1 in ("20", "40", "80")
              if math.isfinite(S[f"S_boundary_coast|two|{n1}|U5 union"]["size"]["Cu"])]
    un2 = [S[f"S_boundary_coast|two|{n1}|U5 union"]["cov"][0] for n1 in un2_n1]
    sa = [S[f"{t}|solder_alone"]["0.1"][0] for t, _, _ in STAGS]
    return r"""\begin{table}[htbp]
\centering
\caption{Baseplate-solder fatigue, a mechanism outside the certificate. (a) Case-temperature cycles of the certified
modules over the 4215 vehicle days: median of the largest case swing of a day, median heating time of the largest
1\,\% of the case swings, solder life under the nominal law, and solder damage relative to the junction damage of the
module, that of its most damaged chip, margined or bare (Table~\ref{tab:S17}). Solder law for traction modules with an
AlSiC baseplate, as in the case-study module~\cite{mitsubishi2012cm1000hc}: $N_f=3\cdot10^4\,(80~\mathrm{K}/\Delta
T_C)^{4.4}$, from the manufacturer's values at 80, 65 and 40~K, which give exponents of
4.41--4.44~\cite{infineon2021an}, extrapolated to the smaller swings found here; a copper baseplate has ten times fewer
cycles. Hold time: since the curve holds for cycles of several minutes, the life of a cycle heated longer than 120~s is
divided by $(t_{\mathrm{on}}/120~\mathrm{s})^{0.5}$, with the stated heating-time slope of the margin. Stacked: hold-time rule,
thermal scale 1.10, case swing under the hottest chip 1.3 times that of the lumped case node, and lives halved.
(b) Coverage at $\alpha=0.10$ by paired certificates when the true damage of a module is the larger of its junction and
solder damages ($n_2=100$, 200 repetitions, both modules; when cruising, the junction-only columns repeat the runs of
Table~\ref{tab:p3-T3}). Truth: the solder damage follows the nominal law with the hold-time rule, times per world a swing
factor under the hottest chip uniform on 1--1.3 and lives shorter than the curve by a factor uniform on 1--2, and per
module the thermal scale (both swing factors to the power 4.4) and a lognormal solder strength of log standard deviation
0.5. Two certificates: the junction certificate and a certificate of the solder, each at $\alpha/2=0.05$. The solder
certificate pairs 40 thermal-cycling tests of the field lot at the reference swing of the curve with the solder damage
of the recorded intervals under the same law and hold-time rule, margined by the swing factor 1.3 and the thermal bound
1.10, here applied to the case swing. Truth and certificate share the law and the hold-time rule, so the solder
certificate is valid by construction; a real one would also need a stated margin for the extrapolation to small swings
and long heating times. A copper bench measures lives ten times shorter,
so the copper certificate is ten times the AlSiC one, and a module counts as covered when both certificates hold, the
same event with either baseplate. Size: median over modules of the larger of the two certificates relative to the
junction-only certificate. The four runs are, in order, at the margins when cruising and when coasting, then under the
evidence truth when cruising and when coasting. The solder certificate alone covers """ + \
        ", ".join(f"{x:.3f}" for x in sa[:3]) + " and " + f"{sa[3]:.3f}" + r""" at $\alpha=0.10$. Solder damage reaches
at most """ + ", ".join(mx[:3]) + " and " + mx[3] + r"""\,\% of the junction damage with AlSiC and exceeds it in """ + \
        ", ".join(gov_al[:3]) + " and " + gov_al[3] + r"""\,\% of the modules; with copper it exceeds it in """ + \
        ", ".join(gov_cu[:3]) + " and " + gov_cu[3] + r"""\,\%. The union certificates behave alike: at the margins
when coasting with copper, the junction-only union certificate covers """ + rng_txt(un1) + \
        " and the pair of union certificates " + rng_txt(un2) + \
        f" ($n_1={', '.join(un2_n1)}$; at $n_1=20$ the union certificate at $\\alpha/2$ is infinite). " + \
        ts19_hidden(S) + r"""}
\label{tab:S19}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccccc@{}}
\multicolumn{10}{@{}l}{(a) Solder damage of the case-temperature cycles}\\
\toprule
& & \makecell{largest daily\\case swing (K)} & \makecell{heating\\time (s)} & \makecell{solder life\\(years)} &
\multicolumn{4}{c}{solder / margined junction damage} & \makecell{stacked /\\bare junction} \\
\cmidrule(lr){6-9}
Mission & Module & & & & nominal & hold time & stacked & stacked, Cu & \\
\midrule
""" + "\n".join(rows_a) + r"""
\bottomrule
\end{tabular*}

\smallskip
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllcccccc@{}}
\multicolumn{9}{@{}l}{(b) Coverage of the larger of the junction and solder damages, paired certificates, $\alpha=0.10$}\\
\toprule
& & & \multicolumn{3}{c}{junction certificate only} & \multicolumn{3}{c}{two certificates} \\
\cmidrule(lr){4-6}\cmidrule(l){7-9}
Truth & Mission & $n_1$ & \makecell{junction\\damage} & AlSiC & Cu & coverage & \makecell{size,\\AlSiC} &
\makecell{size,\\Cu} \\
\midrule
""" + "\n".join(rows_b) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


# ------------------------------------------------------------------------------------------------------------ S20
def ts20():
    K = J["K"]
    R = json.loads((HERE / "chia_nhieu.json").read_text(encoding="utf-8"))
    kinds = ("one split", "median of 15 at alpha", "median of 15 at alpha/2")
    rows = []
    for tag, lab in (("K_boundary", "at the margins"), ("K_evidence", "evidence")):
        for n1 in ("20", "40"):
            cells = []
            for k in kinds:
                v = K[f"{tag}|{n1}|{k}"]
                cells += [f"{v['cov'][0]:.3f}", f"{v['size_vs_oracle']:.2f}"]
            sp = K[f"{tag}|{n1}|spread"]
            cells.append(f"{sp['median']:.2f}")
            rows.append(f"{lab if n1 == '20' else ''} & {n1} & " + " & ".join(cells) + r" \\")
    h, t = R["schemes"]["half"], R["schemes"]["two"]
    s1, s2 = (str(s) for s in R["seeds"])
    half_single = [h[s]["0.1"]["single_leg_median"] for s in (s1, s2)]
    half_q = [h[s]["0.1"]["single_leg_q10_q90"] for s in (s1, s2)]
    two_q05 = [t[s]["0.05"]["single_leg_q10_q90"] for s in (s1, s2)]
    multi_a = [t[s]["0.05"]["multi_leg"] for s in (s1, s2)]
    multi_h = [h[s]["0.1"]["multi_leg"] for s in (s1, s2)]
    return r"""\begin{table}[htbp]
\centering
\caption{Multi-split paired certificate (Section~\ref{sec:realcert} and Section~\ref{sec:S14}; $n_2=100$, $\alpha=0.10$, 200 repetitions, both
modules, cruising): coverage of the true damage and median size relative to the oracle quantile for the certificate of
one random split, the median of the certificates of 15 splits of the same bench at $\alpha$, and their median at
$\alpha/2$, whose calibrated part misses with probability at most $\alpha$ by Markov's
inequality~\cite{solari2022}; spread: median over repetitions and modules of the ratio of the 90\,\% to the 10\,\%
quantile of the 15 certificates at $\alpha$. The one-split columns repeat Table~\ref{tab:p3-T3}. On the real bench data
(Section~\ref{sec:bench}) and the same """ + f"{R['n_recorded']}" + r""" recorded intervals, two independent sets of """ + \
        f"{R['B']}" + r""" splits give for the leg a median single-split certificate of """ + \
        " and ".join(f"{x:.3f}" for x in half_single) + r""" (10--90\,\% ranges """ + \
        " and ".join(f"{a:.3f}--{b:.3f}" for a, b in half_q) + r""") with 18 scores; another draw of 200 splits gives the
0.030 of Section~\ref{sec:realcert}, so the median of the single-split certificates itself varies with the draw. The
median certificates at $\alpha$ are """ + " and ".join(f"{x:.3f}" for x in multi_h) + r""". At $\alpha/2$, which needs
19 scores, for instance with two fitting modules per run (24 scores), single splits have 10--90\,\% ranges of """ + \
        " and ".join(f"{a:.3f}--{b:.3f}" for a, b in two_q05) + r""", and the multi-split certificate is """ + \
        " and ".join(f"{x:.3f}" for x in multi_a) + r""".}
\label{tab:S20}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccccccc@{}}
\toprule
& & \multicolumn{2}{c}{one split} & \multicolumn{2}{c}{median of 15, $\alpha$} & \multicolumn{2}{c}{median of 15,
$\alpha/2$} & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Truth & $n_1$ & coverage & size & coverage & size & coverage & size & spread \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


# ------------------------------------------------------------------------------------------------------- S21, S22
def ts22():
    G = json.loads((HERE.parent / "P3_kiemtoan" / "g27_ton_ngoai_mau.json").read_text(encoding="utf-8"))
    h, c, jg = G["hernes"], G["choi2017"], G["junghaenel2017"]
    rows = []
    for lab, d, fit_on, pred in (
            (r"Hernes et al.~\cite{hernes2021}", h,
             f"runs 1--3, 5 ($t_{{\\mathrm{{on}}}}\\le{h['t_max_train']:g}$~s)", f"run 4 ({h['t_test']:g}~s)"),
            (r"Choi et al.~\cite{choi2017}", c, f"$t_{{\\mathrm{{on}}}}\\le{c['t_max_train']:g}$~s",
             " and ".join(f"{t:g}" for t in c["t_test"]) + "~s")):
        for key, rule in (("fitted_extrapolation", f"fitted exponent ${d['fitted_ton_exponent']:.2f}$"),
                          ("stated_rule", r"stated slope $\gamma_t=0.5$")):
            v = d[key]
            rows.append(f"{lab if key == 'fitted_extrapolation' else ''} & "
                        f"{fit_on if key == 'fitted_extrapolation' else ''} & "
                        f"{pred if key == 'fitted_extrapolation' else ''} & {rule} & "
                        f"{round(v['covered'] * v['n'])}/{v['n']} & {v['median_ratio_to_bound']:.2f} & "
                        f"{v['min_ratio_to_bound']:.2f} \\\\")
        rows.append(r"\addlinespace")
    rj = [x for m in jg.values() for x in m["ratio_observed_to_stated_rule"].values()]
    return r"""\begin{table}[htbp]
\centering
\caption{Out-of-sample check of the stated heating-time rule on real lifetime data (Section~\ref{sec:margin}). Each
campaign is fitted on its shorter heating times and its longest ones are predicted by extrapolating the fitted
heating-time exponent or by the stated slope $\gamma_t=0.5$ from the longest fitted heating time; the conformal lower
10\,\% bound adds the $\lfloor0.1(n+1)\rfloor$-th smallest leave-one-out residual of the $n$ fitted modules.
Covered: predicted modules above the bound; ratio: their life relative to the bound (median, smallest). In
Junghaenel and Scheuermann~\cite{junghaenel2017mr} (110~K, no scatter reported), the lives at 7--60~s are """ + \
        f"{min(rj):.1f}--{max(rj):.1f}" + r""" times those the stated rule predicts from 2~s. The Hernes
campaign is the bench of the case study; in it heating time, swing and current were set together (0.15--0.75~s at
1250~A, 1.1 and 4.5~s at 890~A), and the fit without the 4.5~s run has four coefficients for four conditions, so its
exponents cannot be separated. The other two campaigns are of other module types.}
\label{tab:S22}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llllccc@{}}
\toprule
Campaign & Fitted on & Predicted & Rule & covered & \makecell{ratio,\\median} & \makecell{ratio,\\smallest} \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


def ts21():
    W = J["W"]
    rows = []
    names = (("delta", "first order (case study)"), ("mc", "sampled"), ("ellipsoid", r"Scheff\'e ellipsoid"))
    for st, lab in names:
        cells = []
        for n1 in ("20", "40"):
            v = W[f"{n1}|{st}"]
            cells += [f"{v['cov']:.3f} $\\pm$ {1.96 * v['se']:.3f}", f"{v['size_vs_delta']:.2f}"]
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    return r"""\begin{table}[htbp]
\centering
\caption{Coverage of the statistical part of the margin alone at $\alpha_{stat}=0.05$ (truth at the margins, 200
repetitions of 10 coming cruising intervals and both modules; $\pm1.96$ standard errors over repetitions): the true
damage relative to a tested condition against the margined damage with the first-order bound along the gradient of
$\log V$ (the case study), the $1-\alpha_{stat}$ quantile of $V$ over 2000 draws of the coefficient error from its
multivariate Student $t$ distribution (sampled bound), which keeps the curvature of $\log V$, and the largest $V$ over the Scheff\'e confidence
ellipsoid; each bound is maximized over the tested conditions. Size: median ratio to the first-order bound.}
\label{tab:S21}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
& \multicolumn{2}{c}{$n_1=20$} & \multicolumn{2}{c}{$n_1=40$} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Statistical part & coverage & size & coverage & size \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


if __name__ == "__main__":
    write("T6.tex", t6())
    if J.get("F") or J.get("C40"):
        write("TS18.tex", ts18())
    write("TS12.tex", ts12())
    write("TS13.tex", ts13())
    write("TS14.tex", ts14())
    if J.get("C05"):
        write("TS15.tex", ts15())
    write("TS16.tex", ts16())
    write("TS17.tex", ts17())
    if J.get("S"):
        write("TS19.tex", ts19())
    if J.get("K"):
        write("TS20.tex", ts20())
    write("TS22.tex", ts22())
    if J.get("W"):
        write("TS21.tex", ts21())
