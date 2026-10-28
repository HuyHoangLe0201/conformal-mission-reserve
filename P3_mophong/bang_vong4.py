"""P3 fourth round (20/10/2026): Supplementary Tables S23-S27 (Section S15) from vong4_tong_hop.json, e3_neo_co_dinh.json,
so_sanh_et.json and et_chung_chi.json, written to ../P3_RESS/TS23.tex ... TS27.tex.

  S23  conformal against parametric certificates when the strength law has a weak subpopulation, and at alpha = 0.01
  S24  closed loop (replacement only) and mission change under the weak subpopulation; ages replanned after a delay
  S25  value of anchor tests in the closed loop
  S26  electro-thermal truth
  S27  weighted certificate with anchor tests at the full-load loading

Run: python -I bang_vong4.py
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "P3_RESS"
J = json.loads((HERE / "vong4_tong_hop.json").read_text(encoding="utf-8"))


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8", newline="\n")
    print("written", OUT / name)


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def pct(x):
    return "--" if x is None else f"{100 * x:.1f}"


# ---------------------------------------------------------------------------------------------------------- S23
CERT23 = (("U2 Monte Carlo", "Monte Carlo"), ("U3 Bayes", "Bayes"), ("U5 union", "union"), ("U6 paired", "paired"))
COLS23 = (("A_boundary_k0", "40", "0.1", "default, 0.10, 40"), ("P_boundary_weak", "40", "0.1", "weak, 0.10, 40"),
          ("P_boundary_weak", "40", "0.05", "weak, 0.05, 40"), ("P_boundary_weak", "80", "0.05", "weak, 0.05, 80"),
          ("P_boundary_a001", "200", "0.01", "default, 0.01, 200"),
          ("P_boundary_weak_a001", "100", "0.01", "weak, 0.01, 100"),
          ("P_boundary_weak_a001", "200", "0.01", "weak, 0.01, 200"))


def ts23():
    S = J["static"]
    rows = []
    for key, lab in CERT23:
        cells = []
        for tag, n1, a, _ in COLS23:
            x = S[tag][n1][a][key]
            cells.append(f"{f3(x['calibrated'])} & {f3(x['coverage'])}" if x["finite"] > 0.5 else "-- & --")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    head = " & ".join(r"\multicolumn{2}{c}{\makecell{" + c[3].replace(", ", r",\\", 1) + "}}" for c in COLS23)
    cmid = "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(COLS23)))
    return r"""\begin{table}[htbp]
\centering
\caption{Conformal against parametric certificates when the strength law is not of the Weibull (Gumbel) family that the
Monte Carlo and Bayes certificates assume: truth at the margins, with a weak subpopulation (2--10\,\% of the devices of
the lot, bench and field alike, with lives 1.6--2.7 times shorter) or without it (default), and at $\alpha=0.01$ with
benches of 200 and 400 modules ($n_2=200$; elsewhere $n_2=100$). Per column: truth, $\alpha$, $n_1$; cal.: coverage of the calibrated part
(the field module's score at its true strength), true: coverage of the true damage; 200 repetitions, both modules;
--: certificate infinite. Monte Carlo standard error of a coverage of 0.95 about 0.011, of 0.99 about 0.005.}
\label{tab:S23}
\scriptsize
\setlength{\tabcolsep}{1.6pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l""" + "cc" * len(COLS23) + r"""@{}}
\toprule
& """ + head + r""" \\
""" + cmid + "\nCertificate & " + " & ".join(["cal. & true"] * len(COLS23)) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


# ---------------------------------------------------------------------------------------------------------- S24
RUNS24 = (("replace|paired", "cert|paired", "paired certificate"), ("replace|union", "cert|union", "union certificate"),
          ("replace|mc", "cert|mc", "Monte Carlo certificate"), ("replace|bayes", None, "Bayes certificate"),
          ("none|cert_age", "none|cert_age", "certificate age at commissioning"),
          ("none|mc_age", "none|mc_age", "Monte Carlo B10 age"),
          ("none|tbr60", "none|tbr60", "planned 60 intervals"),
          (None, "replan|cert_age", "certificate age replanned at the change"),
          (None, "replan2|cert_age", r"\quad detected 2 intervals late"),
          (None, "replan6|cert_age", r"\quad detected 6 intervals late"),
          (None, "replan12|cert_age", r"\quad detected 12 intervals late"),
          (None, "replan|mc_age", "Monte Carlo B10 age replanned at the change"),
          (None, "replan12|mc_age", r"\quad detected 12 intervals late"))
COLS24 = (("closed", "C_boundary_cruise_weak", "weak, cruising"), ("closed", "C_boundary_coast_weak", "weak, coasting"),
          ("mission", "M_boundary_cruise_coast_dl", "change, default"),
          ("mission", "M_boundary_cruise_coast_weak", "change, weak"))


def ts24():
    rows = []
    for kc, km, lab in RUNS24:
        cells = []
        for grp, tag, _ in COLS24:
            k = kc if grp == "closed" else km
            x = J[grp].get(tag, {}).get(k) if k else None
            cells += ["--", "--"] if x is None else [pct(x["failed"]), f"{x['cost10']:.2f}"]
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if kc == "none|tbr60":
            rows.append(r"\addlinespace")
    head = " & ".join(r"\multicolumn{2}{c}{" + c[2] + "}" for c in COLS24)
    cmid = "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(COLS24)))
    return r"""\begin{table}[htbp]
\centering
\caption{Replacement decisions under the weak subpopulation of Table~\ref{tab:S23} (truth at the margins, 300 lives per
column, $\alpha=0.10$, bench of 60 modules, $n_2=50$, no derating): failed lives (\%) and cost per 100 intervals for
$c_F=10$ (for $c_F=50$ see the text).
Weak, cruising and coasting: closed loop on one mission; change: cruising, then coasting from an interval drawn in
10--30 (Table~S14), with the default strength law and with the weak subpopulation. Replanned ages are recomputed from
the margined damage accrued up to the detection of the change, at the change or 2, 6 or 12 intervals after it; a leg
replaced by its static age before detection is not replanned.}
\label{tab:S24}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}
\toprule
& """ + head + r""" \\
""" + cmid + "\nDecision rule & " + " & ".join(["failed & cost"] * len(COLS24)) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


# ---------------------------------------------------------------------------------------------------------- S25
RUNS25 = (("replace|paired", "paired certificate"), ("replace|union", "union certificate"),
          ("none|cert_age", "certificate age at commissioning"), ("none|mc_age", "Monte Carlo B10 age"),
          ("none|mc_opt10", "Monte Carlo optimal age"), ("none|tbr60", "planned 60 intervals"),
          ("none|rtf", "run to failure"))


def ts25():
    C = J["closed"]
    blocks = []
    for sc, lab_sc in (("evidence", "Evidence truth"), ("boundary", "Truth at the margins")):
        blocks.append(r"\addlinespace" + "\n" + r"\multicolumn{13}{@{}l}{\textit{" + lab_sc + r"}} \\")
        for k, lab in RUNS25:
            cells = []
            for st in ("cruise", "coast"):
                for src in (f"C_{sc}_{st}", f"V_{sc}_{st}_real", f"V_{sc}_{st}_ideal"):
                    x = C.get(src, {}).get(k)
                    cells += ["--", "--"] if x is None else [f"{x['served']:.0f}", pct(x["failed"])]
            blocks.append(f"{lab} & " + " & ".join(cells) + r" \\")
    return r"""\begin{table}[htbp]
\centering
\caption{Value of anchor tests in the closed loop (replacement only, no derating, 300 common lives per column,
$\alpha=0.10$, bench of 60 modules, $n_2=50$): mean served intervals and failed lives (\%). Stated: the stated margin of the case
study (Table~\ref{tab:p3-T5}); 5 anchors: the heating-time rule replaced by five anchor tests at each of 30 and 300~s (80~K, joint
upper bound at 0.95, Section~S5); ideal: every stated lifetime component (swing slope, heating-time slopes, activation
energy) replaced by its true value, as ideal anchor tests would measure it, the Miner factor, thermal bound and
statistical part being kept. Static ages are computed with the same margin. Costs per 100 intervals are in
Section~\ref{sec:S15}.}
\label{tab:S25}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccccc@{}}
\toprule
& \multicolumn{6}{c}{Cruising} & \multicolumn{6}{c}{Coasting} \\
\cmidrule(lr){2-7}\cmidrule(l){8-13}
& \multicolumn{2}{c}{stated} & \multicolumn{2}{c}{5 anchors} & \multicolumn{2}{c}{ideal}
& \multicolumn{2}{c}{stated} & \multicolumn{2}{c}{5 anchors} & \multicolumn{2}{c}{ideal} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(l){12-13}
Decision rule & served & failed & served & failed & served & failed & served & failed & served & failed & served & failed \\
\midrule
""" + "\n".join(blocks) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


# ---------------------------------------------------------------------------------------------------------- S26
def ts26():
    so = json.loads((HERE / "so_sanh_et.json").read_text(encoding="utf-8"))
    ec = json.loads((HERE / "et_chung_chi.json").read_text(encoding="utf-8"))
    S = J["static"]
    rows_a = []
    for st, lab in (("cruise", "cruising"), ("coast", "coasting")):
        x = so[st]
        q = x["ratio_et_q"]
        hq = x["headroom_q"]
        r = ec[st]
        rows_a.append(f"{lab} & {q[1]:.2f} & {q[2]:.2f} & {q[3]:.2f} & {q[4]:.2f} & {hq[2]:.2f} & "
                      f"{100 * x['share_et_above_headroom']:.1f} & {r['linear 150 C']['reserve_median']:.1f} & "
                      f"{r['electro-thermal']['reserve_median']:.1f}" + r" \\")
    rows_b = []
    for sc, lab in (("boundary", "at the margins"), ("evidence", "evidence")):
        cells = []
        for st in ("cruise", "coast"):
            d = S.get(f"E_{sc}_et_{st}")
            for n1 in ("20", "40"):
                x = d[n1]["0.1"]["U6 paired"] if d else None
                cells.append("--" if x is None else f"{x['coverage']:.3f}")
        rows_b.append(f"{lab} & " + " & ".join(cells) + r" \\")
    return r"""\begin{table}[htbp]
\centering
\caption{Electro-thermal truth (repository script P3\_mophong/dien\_nhiet.py): chip losses at the instantaneous junction temperature (on-state curves
at 25, 125 and 150~$^\circ$C, switching energies extrapolated linearly below 125~$^\circ$C) and a junction-to-case
impedance growing with the junction temperature as $((T_j+273.15)/298.15)^{1.3}$, on the same vehicle days. (a) Ratio
of the true damage per module on the electro-thermal response to that on the case-study response (losses at
150~$^\circ$C, linear network), both at thermal scale 1.0, over 500 intervals (10, 50, 90\,\% quantiles and maximum);
headroom: median ratio of the case-study damage at the stated scale 1.10 to that at 1.0; share of intervals whose
electro-thermal damage exceeds the headroom; median certified reserve of a new leg on the real bench data (40 half
splits, $\alpha=0.10$) with each response as the certificate's thermal model. (b) Coverage of the true damage by the
paired certificate (case-study model) when the truth runs on the electro-thermal response ($\alpha=0.10$, 200
repetitions).}
\label{tab:S26}
\footnotesize
\textit{(a) Damage ratios and certified reserve}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}
\toprule
& \multicolumn{4}{c}{electro-thermal / case study} & & & \multicolumn{2}{c}{certified reserve} \\
\cmidrule(lr){2-5}\cmidrule(l){8-9}
Mission & 10\,\% & 50\,\% & 90\,\% & max & headroom & above (\%) & case study & electro-thermal \\
\midrule
""" + "\n".join(rows_a) + r"""
\bottomrule
\end{tabular*}

\medskip
\textit{(b) Coverage with the electro-thermal truth}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
& \multicolumn{2}{c}{cruising} & \multicolumn{2}{c}{coasting} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Truth & $n_1=20$ & $n_1=40$ & $n_1=20$ & $n_1=40$ \\
\midrule
""" + "\n".join(rows_b) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


# ---------------------------------------------------------------------------------------------------------- S27
def ts27():
    e = json.loads((HERE / "e3_neo_co_dinh.json").read_text(encoding="utf-8"))
    rows = []
    for st, lab in (("cruise", "cruising"), ("coast", "coasting")):
        for fac in ("1.0", "1.5"):
            x = e[f"{st}_x{fac}"]
            cells = []
            for d in ("0.0", "0.1", "0.2", "0.3", "0.5"):
                y = x["levels"][d]
                cells.append(f"{y['X_median_K']:.1f} & {y['p_test_20']:.3f}")
            rows.append(f"{lab}, $\\times{fac}$ & " + " & ".join(cells) + r" \\")
    # (b) sensitivity to the kernel bandwidths and to the placement of 20 anchor tests (e3_neo_bang_thong.py)
    b = json.loads((HERE / "e3_neo_bang_thong.json").read_text(encoding="utf-8"))
    rows_b = []
    for st, lab in (("cruise", "cruising"), ("coast", "coasting")):
        for name, m, txt in (("full", "0.5", "at full load, kernels $\\times0.5$"), ("full", "2.0", "at full load, $\\times2$"),
                             ("full", "4.0", "at full load, $\\times4$"), ("spread", "1.0", "spread over the levels")):
            p = b[st]["p_test"][f"{name}|{m}"]
            rows_b.append(f"{lab}, {txt} & " + " & ".join(f"{p[d]:.3f}" for d in ("0.0", "0.1", "0.2", "0.3", "0.5"))
                          + r" \\")
    return r"""\begin{table}[htbp]
\centering
\caption{Weighted certificate with anchor tests at the loading at full load, where a bench planned before the controller
acts would place them ($\alpha_S=0.05$; the certificate is infinite when the test weight $p_{n_1+1}$ exceeds 0.05;
200 intervals per mission). (a) 20 anchor tests at full load: median damage-weighted swing $X$ (K) of the critical chip
and test weight at each derating level, for the case-study converter ($\times1.0$) and one with 1.5 times its junction
swings ($\times1.5$); with 40 or 80 anchor tests the weights at derated levels change by at most 0.03. (b) Case-study
converter: test weight with both kernel bandwidths (Silverman's rule) multiplied by 0.5, 2 or 4, and with the 20
anchor tests spread over the loadings of the six derating levels (three or four at each).}
\label{tab:S27}
\footnotesize
\setlength{\tabcolsep}{3pt}
\textit{(a) Anchor tests at full load}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccc@{}}
\toprule
& \multicolumn{2}{c}{no derating} & \multicolumn{2}{c}{10\,\%} & \multicolumn{2}{c}{20\,\%} & \multicolumn{2}{c}{30\,\%}
& \multicolumn{2}{c}{50\,\%} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}
Mission, design & $X$ & weight & $X$ & weight & $X$ & weight & $X$ & weight & $X$ & weight \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}

\medskip
\textit{(b) Kernel bandwidths and placement of the anchor tests (test weight)}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
Mission, anchor tests & no derating & 10\,\% & 20\,\% & 30\,\% & 50\,\% \\
\midrule
""" + "\n".join(rows_b) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


if __name__ == "__main__":
    write("TS23.tex", ts23())
    if "C_boundary_coast_weak" in J["closed"]:
        write("TS24.tex", ts24())
    if any(k.startswith("V_") for k in J["closed"]):
        write("TS25.tex", ts25())
    if (HERE / "so_sanh_et.json").exists() and (HERE / "et_chung_chi.json").exists():
        write("TS26.tex", ts26())
    write("TS27.tex", ts27())
