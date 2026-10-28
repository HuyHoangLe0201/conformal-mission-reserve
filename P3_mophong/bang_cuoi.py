"""P3: final tables T1, T3 and T5 of the manuscript, Tables S2-S10 of the Supplementary Material, and the review
document of all final figures and tables.

3.3 kV converter and per-module certificates (18/10/2026; the 1.7 kV version of this script is kept in
luu_1700V_20261018/ma): every closed-loop table now reads the merged runs of g6_tong_hop.json ('C', 300 lives per
truth and driving strategy, every policy and baseline on common lives), the static tables pool the two modules of a
leg, and two tables are new: S9, the information loop of Section 6.8, and S10, the test of re-simulation.

T1 (uncertainty sources) and Table S8 (data sets) are fixed text from the outline. T3 and Tables S2-S7, S9 and S10 are
read from g6_tong_hop.json, g7_tong_hop.json, g6_vuot_nhieu.json and e3_bang_thong.json; T5 is the stated margin of
G3/G5 with its budget from g5_ngan_sach_chienluoc_cruise.json and g6_D_det.json. A result that is missing prints as
'--', so the script runs while experiments are still open.
Outputs in ../P3_hinh_bang/:
  T1.tex, T3.tex, T5.tex     one table environment each, ready for \\input in the manuscript (booktabs, tabularx);
  TS2.tex ... TS10.tex       tables of the Supplementary Material (article class);
  P3_hinh_bang_cuoi.tex      figures F1-F9 with their captions and tables T1, T3, T5 (xelatex, biber, xelatex x2).
In the project, the tables are also copied to ../P3_RESS, and the figure captions of the overview are checked against
those of the manuscript (../P3_RESS/sec*.tex): a caption changed in the manuscript must be copied here.

Run: python -I bang_cuoi.py
"""
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "P3_hinh_bang"
RESS = HERE.parent / "P3_RESS"


def jload(name):
    f = HERE / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


G6 = jload("g6_tong_hop.json")
G7 = jload("g7_tong_hop.json")
VN = jload("g6_vuot_nhieu.json")
INF = r"$\infty$"


def f(v, d=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if isinstance(v, float) and math.isinf(v):
        return INF
    return f"{v:.{d}f}"


def pct(v, d=0):
    return "--" if v is None else f"{100 * v:.{d}f}"


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8", newline="\n")
    print("written", OUT / name)


# ---------------------------------------------------------------------------------------------------------------------
# T1. Uncertainty sources
T1 = r"""\begin{table}[tbp]
\centering
\caption{Sources of uncertainty in the end-of-interval damage, the data that inform them, and how each enters the
certificate. Labels are observable when the quantity itself can be measured on the population or the mission before the
decision.}
\label{tab:p3-T1}
\footnotesize
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{2.3cm}>{\raggedright\arraybackslash}X
>{\raggedright\arraybackslash}p{3.3cm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{3.4cm}@{}}
\toprule
Source & Enters through & Labels observable & Data in the case study & Treatment \\
\midrule
Strength scatter of the device population & log-strength $S$ of a device on the reference conditions
$\mathcal T_{\mathrm{ref}}$ & yes: cycles to failure of bench devices at tested conditions & 34 modules in five runs
\cite{hernes2021} & calibrated: ranks of the scores, level $\alpha$ (paired)
or $\alpha_S$ (union) \\
\addlinespace
Mission variability & inputs $Z$ of a maintenance interval: timetable, dwell times, ambient & yes: recorded intervals,
re-simulated under the deployed policy & 112 days of a high-speed corridor \cite{zhang2022hsr} & calibrated: paired with the scores (level $\alpha$) or own quantile (level $\alpha_V$) \\
\addlinespace
Lifetime extrapolation and damage accumulation & margin $\varepsilon_{ex}(x)$ on cycle lives off
$\mathcal T_{\mathrm{ref}}$ (swing, heating time, current, minimum temperature, statistical part) and Miner factor
$\kappa_M$ & no: no life is observed at field conditions (low swing, long heating time) before the decisions & published
single-factor and mixed-loading tests \cite{junghaenel2017mr,choi2017,choi2018mr,hernes2021pcim,choi2017apec}; solder
fatigue under long dwell \cite{engelmaier1983} & stated with confidence $1-\alpha_{ex}$ (Table~\ref{tab:p3-T5});
anchor tests at field heating times replace the largest component \\
\addlinespace
Thermal response & upper thermal model used in the margined damage $V$ & partly: junction rise measured with test
pulses in dwell windows and in operation & dwell windows of the timetable; thermal path degraded by 20--40\,\% at end
of life (assumed) & stated with confidence $1-\alpha_{th}$; recalibrated each interval (Section~\ref{sec:infoloop}) \\
\bottomrule
\end{tabularx}
\end{table}
"""

# ---------------------------------------------------------------------------------------------------------------------
# Table S8. Data sets
TS8 = r"""\begin{table}[htbp]
\centering
\caption{Data sets, their use and their limitations.}
\label{tab:S8}
\footnotesize
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{2.6cm}>{\raggedright\arraybackslash}X
>{\raggedright\arraybackslash}p{4.1cm}>{\raggedright\arraybackslash}p{4.1cm}@{}}
\toprule
Data set & Content & Use & Limitations \\
\midrule
Hernes et al.\ \cite{hernes2021} & 3.3~kV/1~kA modules in five runs: $\Delta T$ 30--70~K (lowest module 26~K),
$T_{j,\min}=60$~$^\circ$C, $t_{\mathrm{on}}$ 0.15--4.5~s, two currents; 34 of 36 failed modules digitized, within
1\,\% of the published table & $\mathcal T_{\mathrm{ref}}$; fit of $N^{\mathrm{mod}}$ and scores; $\beta=2.08$ & four
modules withdrawn for another program, without cycle counts; no test below 26~K or above 4.5~s \\
\addlinespace
Hernes et al.\ \cite{hernes2021pcim} & same module type: Miner sums at failure of 48 of 49 modules; five
constant-amplitude runs, 40 then 70~K (7 modules), 70 then 40~K (6) & $\kappa_M=0.85$; conformal check
(Fig.~\ref{fig:p3-F3}d) & 13 mixed-loading modules; check not independent \\
\addlinespace
Choi et al.\ \cite{choi2017} & 600~V/30~A transfer-molded intelligent power module (IPM): 39 modules, six swing
durations of 0.59--10~s at $\Delta T_j\approx81$~K; from Weibull plots & slope $-0.14$ in the swing duration; equal
scatter across conditions & other technology; durations far below the field heating times \\
\addlinespace
Choi et al.\ \cite{choi2018mr} & same IPM: 18 modules at $\Delta T_j$ 40, 60, 70~K, 1~s; table per module & scatter
differs between campaigns, so $\mathcal T_{\mathrm{ref}}$ and population lie within one campaign & two modules
suspended by an external fuse \\
\addlinespace
Junghaenel and Scheuermann \cite{junghaenel2017mr}, Table~1 & heating time 0.07--2~s at 70~K and 2--60~s at 110~K;
wire-bond and solder failures & slopes $-0.19$ (solder), $-0.27$ (wire bond) at 2--60~s; stated $\gamma_t=0.5$ &
no baseplate; no single-factor test above 60~s \\
\addlinespace
Choi et al.\ \cite{choi2017apec} & four modules, two profiles of mixed swing durations & Miner factors 0.84 and 0.93 &
four modules \\
\addlinespace
Zeng et al.\ \cite{zeng2019} & 1200~V/300~A baseplate modules, wire-bond failures; 18 modules at constant amplitude and
12 under block and alternating sequences & Miner sums 1.08 (blocks, after swing correction) and 1.01 (alternating) &
other module type; wire bonds only \\
\addlinespace
Hoffmann et al.\ \cite{hoffmann2020,hoffmann2023} & SiC modules; Weibull parameters per run, scatter changing with
$\Delta T$ & scatter and extrapolation error depend on the loading; activation energy of SiC (Table~\ref{tab:p3-T5}) &
other technology; not used for absolute lives \\
\addlinespace
Zhang et al.\ \cite{zhang2022hsr} & Beijing South--Shanghai Hongqiao (1318~km, 350~km/h): running, dwell and distance
records of 37 train numbers per direction, 7554 train-days, 61\,834 sections, 8~Oct 2019 to 27~Jan 2020, 1-min
resolution; daily temperature of the station area & inputs $Z$, dwell windows, 30-day intervals by resampling days &
no summer (warmer season: Table~\ref{tab:S4}b); 5.0\,\% of sections given planned times; speed and current from a
train model \\
\addlinespace
Excluded & Zhang et al.\ \cite{zhang2021data}: a single device; Choi et al.\ \cite{choi2016mr}: interim report of
\cite{choi2017}, 10\,\% lives only; fixed CIPS08 exponents \cite{bayerer2008} (Section~\ref{sec:bench}) & -- & -- \\
\bottomrule
\end{tabularx}
\end{table}
"""

# ---------------------------------------------------------------------------------------------------------------------
# T3. Coverage and tightness of a single certificate (static)
CERTS = [("U1 box", "box"), ("U2 Monte Carlo", "Monte Carlo"), ("U3 Bayes", "Bayes"),
         ("U5 union", "union"), ("U6 paired", "paired"), ("U9 PAC", "PAC"),
         ("U6 anchored", r"paired, ideal anchor tests$^{\dagger}$"), ("U10 oracle", r"oracle$^{\dagger}$")]
MARGIN_COLS = [("20", "0.1"), ("40", "0.1"), ("40", "0.05"), ("40", "0.01")]
EVID_COLS = [("20", "0.1"), ("40", "0.1")]


def a_val(tag, n1, a, c):
    return G6.get(tag, {}).get(f"{n1}|{a}|{c}")


def a_cell(tag, n1, a, c, what):
    v = a_val(tag, n1, a, c)
    if v is None:
        return "--"
    if what == "cov":
        return f(v["coverage"], 3)
    return INF if v["finite"] < 0.5 else f(v["ratio"])


def t3_static():
    rows = []
    for key, lab in CERTS:
        cells = []
        for n1, a in MARGIN_COLS:
            cells += [a_cell("A_boundary_k0", n1, a, key, "cov"), a_cell("A_boundary_k0", n1, a, key, "ratio")]
        cells += [a_cell("A_evidence_k0", n1, a, key, "ratio") for n1, a in EVID_COLS]
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def cal(tag, n1, a, c="U6 paired"):
    v = a_val(tag, n1, a, c)
    return None if v is None else v["calibrated"]


EV = G6.get("A_evidence_k0", {})
EV_COV = [v["coverage"] for k, v in EV.items() if k.count("|") == 2 and "U10" not in k and "anchored" not in k] or [None]
EV_ANCH = [v["coverage"] for k, v in EV.items() if k.endswith("|U6 anchored") and v["finite"] > 0.5] or [None]
N_REC = a_val("A_boundary_k0", "40", "0.1", "U6 paired")
SE_REP = N_REC.get("se_rep") if N_REC else None
SE_OR = a_val("A_boundary_k0", "40", "0.1", "U10 oracle")
SE_OR = SE_OR.get("se_rep") if SE_OR else None

T3 = r"""\begin{table}[tbp]
\centering
\caption{Coverage and tightness of a single certificate for a new leg ($n_2=100$ recorded intervals, 200
repetitions per cell, each with the outer and the inner module): coverage of the true end-of-interval damage and median
ratio of the certificate to the oracle quantile of the true damage, over both modules.}
\label{tab:p3-T3}
\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccc@{}}
\toprule
& \multicolumn{8}{c}{Truth at the margins} & \multicolumn{2}{c}{\makecell{Evidence truth,\\ratio, $\alpha=0.10$}} \\
\cmidrule(lr){2-9}\cmidrule(l){10-11}
& \multicolumn{2}{c}{$\alpha=0.10$, $n_1=20$} & \multicolumn{2}{c}{$\alpha=0.10$, $n_1=40$}
& \multicolumn{2}{c}{$\alpha=0.05$, $n_1=40$} & \multicolumn{2}{c}{$\alpha=0.01$, $n_1=40$} & $n_1=20$ & $n_1=40$ \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Certificate & cov. & ratio & cov. & ratio & cov. & ratio & cov. & ratio & & \\
\midrule
""" + t3_static() + r"""
\bottomrule
\end{tabular*}

\smallskip
\begin{minipage}{\textwidth}\scriptsize
$\infty$: infinite in every repetition (coverage then trivial). Under the evidence truth every certificate except
the oracle covers """ + f(min(x for x in EV_COV if x is not None) if EV_COV[0] is not None else None, 3) + \
    r""" (ideal anchor tests: """ + ((f"{min(EV_ANCH):.3f}" if f"{min(EV_ANCH):.3f}" == f"{max(EV_ANCH):.3f}"
                                    else f"{min(EV_ANCH):.3f}--{max(EV_ANCH):.3f}") if EV_ANCH[0] is not None else "--") + \
    r"""). Calibrated part of the paired certificate at the margins: """ + \
    f"{f(cal('A_boundary_k0', '20', '0.1'), 3)} and {f(cal('A_boundary_k0', '40', '0.1'), 3)} at $\\alpha=0.10$, " \
    f"{f(cal('A_boundary_k0', '40', '0.05'), 3)} at $\\alpha=0.05$" + r""". Monte Carlo standard error of the paired
coverage at the margins ($n_1=40$, $\alpha=0.10$; both modules of a repetition share a bench): """ + f"{f(SE_REP, 3)}" + \
    r""".
$^{\dagger}$Uses the truth; for analysis only.
\end{minipage}
\end{table}
"""

# ---------------------------------------------------------------------------------------------------------------------
# T5. Stated extrapolation margin and its budget
DET = G6.get("D", {}).get("det", {})
BUD = jload("g5_ngan_sach_chienluoc_cruise.json").get("base", {})


def journey_row():
    """Budget of the damage-weighted cruising journey for the critical chip of the leg: stated components from
    g5_ngan_sach (fit on all 34 modules), statistical part from the half splits of experiment D, the basis of the two
    single cycles."""
    lb = BUD.get("log_budget_critical")
    if not lb or "base" not in DET or "V_interval_median_half" not in DET["base"]:
        return None, None
    stat = math.log(DET["base"]["V_interval_median_half"] / DET["stat none"]["V_interval_median_half"])
    c = dict(cal=0.17, stat=stat, dT=lb["swing gamma ln(dT_T/dT)"],
             ton=lb["heating 4.5-60 s"] + lb["heating above 60 s"], tmin=lb["T_min Arrhenius"], miner=lb["Miner"])
    c["total"] = c["stat"] + c["dT"] + c["ton"] + c["tmin"] + c["miner"]
    return c, BUD.get("critical_chip")


JOURNEY, CRIT = journey_row()


def budget_row(lab, c):
    if c is None:
        return f"{lab} & " + " & ".join(["--"] * 8) + r" \\"
    return (f"{lab} & {c['cal']:.2f} & {c['stat']:.2f} & {c['dT']:.2f} & {c['ton']:.2f} & {c['tmin']:.2f} & "
            f"{c['miner']:.2f} & {c['total']:.2f} & {math.exp(c['total']):.0f} \\\\")


BUDGET = "\n".join([
    budget_row(r"40~K, 90~s, $T_{\min}$ 60~$^\circ$C",
               dict(cal=0.17, stat=0.53, dT=0.0, ton=1.50, tmin=0.0, miner=0.16, total=2.19)),
    budget_row(r"10~K, 90~s, $T_{\min}$ 70~$^\circ$C",
               dict(cal=0.17, stat=1.67, dT=1.15, ton=1.50, tmin=0.16, miner=0.16, total=4.64)),
    budget_row(r"journey, cruising, chip " + (CRIT or "--") + r"$^{*}$", JOURNEY)])
LB = BUD.get("log_budget_critical", {})

T5 = r"""\begin{table}[tbp]
\centering
\caption{Stated extrapolation margin of the case study. (a) Components, values, evidence and sensitivity ranges; the
thermal bound, held at its own level $\alpha_{th}$, is listed for completeness. (b) Margin budget in log units (log of
the factor on damage) for two traction cycles and for the damage-weighted journey, with the calibrated part for
comparison.}
\label{tab:p3-T5}
\footnotesize
\textit{(a) Components}\\[2pt]
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{2.6cm}>{\raggedright\arraybackslash}p{3.5cm}
>{\raggedright\arraybackslash}p{2.0cm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{2.9cm}@{}}
\toprule
Component & Rule & Stated value & Evidence & Sensitivity (Section~\ref{sec:sens}) \\
\midrule
Statistical error of $N^{\mathrm{mod}}$ off the tested conditions & one-sided bound on $\log V$ of the interval along
its gradient, relative to the tested conditions & Student $t_{0.95}$; $\alpha_{stat}=0.05$ within $\alpha_{ex}$, approximate &
fit covariance; simulated coverage """ + (
    f"{min(G6.get(t, {}).get(f'{n}|stat_ok', 1) for t in ('A_boundary_k0', 'A_evidence_k0') for n in ('10', '20', '40', '80')):.3f}--"
    f"{max(G6.get(t, {}).get(f'{n}|stat_ok', 0) for t in ('A_boundary_k0', 'A_evidence_k0') for n in ('10', '20', '40', '80')):.3f}"
    if G6.get("A_boundary_k0") else "--") + r""" & Scheff\'e ellipsoid; none \\
\addlinespace
Swing below $\Delta T_{\mathcal T}=26$~K & $\gamma\log(\Delta T_{\mathcal T}/\Delta T)$ & $\gamma=1.2$ & swing exponent
changing from 4.42 to 3.21 at a mechanism switch \cite{scheuermann2018} & $\gamma\in\{0,0.6,2.08\}$;
$\Delta T_{\mathcal T}=68$~K \\
\addlinespace
Heating time 4.5--60~s & $N^{\mathrm{mod}}$ at 4.5~s plus $\gamma_t\log(t_{\mathrm{on}}/4.5\,\mathrm{s})$ &
$\gamma_t=0.5$ & slopes $-0.19$ (solder) and $-0.27$ (wire bond) at 2--60~s \cite{junghaenel2017mr}; $-0.14$
\cite{choi2017}; $-0.463$ (CIPS08 \cite{bayerer2008}); the long-heating-time tests lack a baseplate &
$\gamma_t\in\{0.29,1.0\}$ \\
\addlinespace
Heating time above 60~s & same slope, not capped & $\gamma_t=0.5$ & solder fatigue grows with the dwell up to hours,
12--20 times from 1~min to 10~h \cite{engelmaier1983}, against 24.5 for $\gamma_t=0.5$; no single-factor test above
60~s & cap at 60~s; slope 0.27 above 60~s \\
\addlinespace
Current & clipped to 890--1250~A; no credit below 890~A & -- & current exponent not identified apart from the
heating-time exponent (correlation 0.99); field current at most 554~A & -- \\
\addlinespace
Minimum temperature above 60~$^\circ$C & Arrhenius term, activation energy $E_a$; no credit below 60~$^\circ$C & $E_a=0.159$~eV & 0.069--0.159~eV
for wire bond and solder \cite{scheuermann2018}; 0.111~eV (CIPS08); 0.132~eV for silicon carbide \cite{hoffmann2023} &
$E_a=0.111$~eV \\
\addlinespace
Miner's rule & $-\log\kappa_M$ & $\kappa_M=0.85$ & 0.85 and 0.97 for the two load orders \cite{hernes2021pcim};
0.84 and 0.93 under mixed swing durations \cite{choi2017apec}; 1.08 and 1.01 for block and alternating sequences
\cite{zeng2019} & $\kappa_M=1$ \\
\midrule
Thermal bound (level $\alpha_{th}$) & upper scale of the modeled junction rise, stated or recalibrated each interval
(Section~\ref{sec:infoloop}) & stated 1.10 & field degradation of the thermal path, 20--40\,\% at end of life &
confidence, forgetting, pulses and drift (Table~S9) \\
\bottomrule
\end{tabularx}

\medskip
\textit{(b) Margin budget (log units)}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}
\toprule
& Calibrated & \multicolumn{5}{c}{Stated} & & \\
\cmidrule(lr){2-2}\cmidrule(lr){3-7}
Cycle or mission & $-q_S$ & statistical & swing & heating time & $T_{\min}$ & Miner & stated total & factor \\
\midrule
""" + BUDGET + r"""
\bottomrule
\end{tabular*}

\smallskip
\begin{minipage}{\textwidth}\scriptsize
(b) Calibrated part: lower 10\,\% quantile of the 34 leave-one-out log-strengths; statistical part with the half
split of the bench data, for the cycles with the normal quantile 1.645 in place of $t_{0.95}$. $^{*}$Damage-weighted over
the cruising journey without derating, the critical chip of the leg; heating time """ + \
    f(LB.get("heating 4.5-60 s")) + r""" for 4.5--60~s and """ + f(LB.get("heating above 60 s")) + r""" above
60~s; the statistical part is the median ratio of the interval damage with and without it over 40 half splits.
\end{minipage}
\end{table}
"""

# ---------------------------------------------------------------------------------------------------------------------
# Table S2. Bandwidth of the density ratio (e3_bang_thong.py)
BW = jload("e3_bang_thong.json")


def ts2_rows():
    rows = []
    for d in sorted(BW, key=float):
        v = BW[d]
        rows.append(f"{100 * float(d):.0f} & {v['X_median_K']:.1f} & " + " & ".join(
            f"{v[f'p_test_anchors_{n}_bw_{b}']:.3f}" for n in (0, 20) for b in ("0.5", "1", "2")) + r" \\")
    return "\n".join(rows) if rows else r"-- & & & & & & & \\"


TS2 = r"""\begin{table}[htbp]
\centering
\caption{Test weight $p_{n_1+1}$ of the weighted strength quantile on the cruising mission (Fig.~\ref{fig:p3-F6}a) for
Gaussian kernels with half, one and two times Silverman's bandwidth, without and with 20 anchor tests at the median
loading summary. The quantile is infinite while the weight exceeds $\alpha_S=0.05$.}
\label{tab:S2}
\small
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cccccccc@{}}
\toprule
& & \multicolumn{3}{c}{No anchor tests} & \multicolumn{3}{c}{20 anchor tests} \\
\cmidrule(lr){3-5}\cmidrule(l){6-8}
Derating (\%) & Summary (K) & $\times0.5$ & $\times1$ & $\times2$ & $\times0.5$ & $\times1$ & $\times2$ \\
\midrule
""" + ts2_rows() + r"""
\bottomrule
\end{tabular*}
\end{table}
"""

# ---------------------------------------------------------------------------------------------------------------------
# Table S3. Temporal structure of the missions (experiment T)
TT = G6.get("T", {})


def ts3_rows():
    rows = []
    for tag, truth in (("T_boundary", "at the margins"), ("T_evidence", "evidence")):
        for dz, lab in (("resampled", "resampled days"), ("blocks", "consecutive dates"),
                        ("forward", "recorded before"), ("backward", "recorded after")):
            v = TT.get(f"{tag}|{dz}")
            if v is None:
                rows.append(f"{truth if dz == 'resampled' else ''} & {lab} & -- & -- & -- & -- & -- \\\\")
                continue
            rows.append(f"{truth if dz == 'resampled' else ''} & {lab} & {v['damage_ratio']:.3f} & {v['cov_paired']:.3f} "
                        f"({v['ci_paired'][0]:.3f}--{v['ci_paired'][1]:.3f}) & {v['cal_paired']:.3f} "
                        f"({v['ci_cal'][0]:.3f}--{v['ci_cal'][1]:.3f}) & {v['cov_union']:.3f} & {v['cal_union']:.3f} \\\\")
        rows.append(r"\addlinespace")
    return "\n".join(rows[:-1])


T_N = TT.get("T_boundary|blocks", {}).get("n", 400)
TS3 = r"""\begin{table}[htbp]
\centering
\caption{Temporal structure of the missions ($n_1=40$, $n_2=100$, $\alpha=0.10$; """ + \
    f"{TT.get('T_boundary|blocks', {}).get('reps', 200)}" + r""" repetitions at the margins and """ + \
    f"{TT.get('T_evidence|blocks', {}).get('reps', 400)}" + r""" under the evidence truth, from two seeds, each with
the outer and the inner module): coverage of the true damage and of the calibrated part by the paired and union
certificates of a new leg, with intervals of $\pm1.96$ standard errors over repetitions for the paired certificate (the
two modules of a repetition share its world, bench and recorded intervals). Intervals from resampled days, as in
the case study; from one vehicle day on each of 30 consecutive dates; and from consecutive dates with the coming
interval on the last 30 dates and every recorded interval before it (recorded before), or on the first 30 dates and
every recorded interval after it (recorded after). Coming damage: model damage of the coming interval over the median
recorded one. Exact level of the calibrated part: $37/41=0.902$.}
\label{tab:S3}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccccc@{}}
\toprule
& & & \multicolumn{2}{c}{Paired} & \multicolumn{2}{c}{Union} \\
\cmidrule(lr){4-5}\cmidrule(l){6-7}
Truth & Intervals from & \makecell{coming damage\\$\times$ recorded} & coverage & calibrated & coverage & calibrated \\
\midrule
""" + ts3_rows() + r"""
\bottomrule
\end{tabular*}
\end{table}
"""

# ---------------------------------------------------------------------------------------------------------------------
# Table S4. Sensitivity to the stated margin, warmer field interval, anchor tests
VARIANT = [("base", "stated margin (base)"), ("gamma 0", r"swing slope $\gamma=0$"),
           ("gamma 0.6", r"swing slope $\gamma=0.6$"), ("gamma 2.08", r"swing slope $\gamma=\beta=2.08$"),
           ("dT_T 68 K", r"$\Delta T_{\mathcal T}=68$~K"), ("gamma_t 0.29", r"heating-time slope $\gamma_t=0.29$"),
           ("gamma_t 1.0", r"heating-time slope $\gamma_t=1.0$"), ("t_on cap 60 s", r"heating time capped at 60~s"),
           ("t_on slope 0.27 above 60 s", r"slope 0.27 above 60~s"), ("E_a 0.111 eV", r"$E_a=0.111$~eV"),
           ("kappa_M 1", r"$\kappa_M=1$"), ("stat ellipsoid", r"statistical part: Scheff\'e ellipsoid"),
           ("stat none", "no statistical part"), ("short pulses counted at 0.15 s", r"pulses below 0.15~s counted"),
           ("short pulses credited, 0.59", r"pulses below 0.15~s counted, slope 0.59")]


def t4_margin():
    if not DET:
        return r"-- & & & & \\"
    rows = []
    b = DET["base"]
    for k, lab in VARIANT:
        v = DET.get(k)
        if v is None:
            continue
        q = v["certified_life_intervals_q10_q90"]
        cov = G6.get("D", {}).get(k, {}).get("coverage")
        rel = v["V_interval_median"] / b["V_interval_median"]
        rows.append(f"{lab} & {rel:.{1 if rel >= 10 else 2}f} & "
                    f"{v['certified_life_intervals_median']:.1f} & {q[0]:.1f}--{q[1]:.1f} & {f(cov, 3)} \\\\")
    return "\n".join(rows)


def t4_shift():
    E, rows = G6.get("E", {}), []
    for tag, truth in (("E_shift_evidence", "evidence"), ("E_shift", "at the margins")):
        for dl in ("10.0", "20.0"):
            v = E.get(f"{tag}|{dl}")
            if v is None:
                rows.append(f"ambient $+{float(dl):.0f}$~K & {truth} & -- & -- & -- \\\\")
                continue
            rows.append(f"ambient $+{float(dl):.0f}$~K & {truth} & {v['damage_ratio']:.3f} & {f(v.get('cert_ratio'), 3)} & "
                        f"{v['cov_resim']:.3f} \\\\")
    return "\n".join(rows)


def t4_anchor():
    N, rows = G6.get("N", {}), []
    for key, lab in (("stated", "stated rule"), ("300|3", "300~s, $n_a=3$"), ("300|5", "300~s, $n_a=5$"),
                     ("300|10", "300~s, $n_a=10$"), ("30+300|3", "30 and 300~s, $n_a=3$"),
                     ("30+300|5", "30 and 300~s, $n_a=5$"), ("30+300|10", "30 and 300~s, $n_a=10$")):
        cells = []
        for t in ("N_evidence", "N_boundary", "N_adverse_strict"):
            v = N.get(f"{t}|{key}|0.1")
            cells += ["--", "--"] if v is None else [f(v["coverage"], 3), f(v["ratio_to_stated"])]
        m = N.get(f"N_boundary|{key}|0.1", {})
        rows.append(f"{lab} & " + " & ".join(cells) + f" & {f(m.get('bounds_ok'), 3)} \\\\")
    return "\n".join(rows)


TS4 = r"""\begin{table}[htbp]
\centering
\caption{Sensitivity of the paired certificate to the stated margin and to a warmer field interval, and anchor tests that
replace the stated heating-time rule.}
\label{tab:S4}
\footnotesize
\textit{(a) Stated margin on the real bench data ($\alpha=0.10$)}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
Variant & \makecell{interval damage\\$\times$ base} & \makecell{certified reserve\\(intervals)} & 10--90\,\% &
\makecell{coverage, truth\\at the margins} \\
\midrule
""" + t4_margin() + r"""
\bottomrule
\end{tabular*}

\medskip
\textit{(b) Warmer field interval ($n_1=40$, $\alpha=0.10$, 100 repetitions of two modules)}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccc@{}}
\toprule
Shift & Truth & \makecell{true damage\\$\times$ unshifted} & \makecell{certificate, re-simulated\\$\times$ recorded} &
coverage \\
\midrule
""" + t4_shift() + r"""
\bottomrule
\end{tabular*}

\medskip
\textit{(c) Anchor tests in place of the stated heating-time rule ($n_1=40$, $\alpha=0.10$, 100 repetitions of two
modules)}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}
\toprule
& \multicolumn{2}{c}{Evidence truth} & \multicolumn{2}{c}{Truth at the margins} & \multicolumn{2}{c}{Adverse truth} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Anchor times & coverage & cert./stated & coverage & cert./stated & coverage & cert./stated & bounds hold \\
\midrule
""" + t4_anchor() + r"""
\bottomrule
\end{tabular*}

\smallskip
\begin{minipage}{\textwidth}\scriptsize
(a) Margined damage of a 30-day interval (critical module, median of 200 intervals, fit on all 34 modules) relative to
the base margin; certified reserve $R_c$ of a new leg (smaller module) in 30-day intervals over 40 half splits, whose
statistical part is wider than that of the full fit (Section~\ref{sec:realcert} gives the median over 200 random splits); coverage when the truth is at the margins and the certificate uses
a narrower variant ($n_1=40$). (b) Certificate: median ratio of the paired certificate on intervals re-simulated at the
field ambient to that on the recorded ones. (c) $n_a$ anchor tests per anchor time at 80~K swing, joint level 0.95;
bounds hold: share of repetitions in which every anchor bound covers the true heating-time effect (truth at the
margins).
\end{minipage}
\end{table}
"""

# ---------------------------------------------------------------------------------------------------------------------
# Table S5. Certificate-triggered derating over leg lives under the certificate policies (truth at the margins)
C = G6.get("C", {})
POLICY = {"pass": "pass/fail", "fixed20": r"fixed 20\,\%", "reserve": "reserve", "reserve_cap": r"reserve, cap",
          "reserve_budget": "reserve, late budget", "replace": "replacement only"}
METHOD = {"split0": "split", "union": "union", "paired": "paired", "reuse": "paired, reused data",
          "aci": r"adaptive$^{\dagger}$", "mc": "Monte Carlo", "bayes": "Bayes"}
LOOP = [("pass", ("union", "paired")), ("fixed20", ("union", "paired")),
        ("reserve", ("split0", "union", "paired", "reuse", "aci", "mc", "bayes")), ("reserve_cap", ("union", "paired")),
        ("reserve_budget", ("union", "paired")), ("replace", ("union", "paired"))]


def ts5_rows(truth):
    rows = []
    for st in ("cruise", "coast"):
        first_s = True
        for pol, meths in LOOP:
            for i, m in enumerate(meths):
                v = C.get(f"{truth}_{st}|{pol}|{m}")
                drive = ("cruising" if st == "cruise" else "coasting") if first_s else ""
                first_s = False
                lab = POLICY[pol] if i == 0 else ""
                if v is None:
                    rows.append(f"{drive} & {lab} & {METHOD[m]} & " + " & ".join(["--"] * 8) + r" \\")
                    continue
                rows.append(f"{drive} & {lab} & {METHOD[m]} & {v['intervals_mean']:.1f} & {f(v['harvest_ok'])} & "
                            f"{pct(v['derated'])} & {pct(v['mean_level'])} & {pct(v['false_of_derated'])} & "
                            f"{pct(v['late_pooled'], 1)} & {f(v['cov_pooled'], 3)} & {f(v['cov_life'], 3)} \\\\")
        rows.append(r"\addlinespace")
    return "\n".join(rows[:-1])


def ts5_note():
    # the certificates with a guarantee per decision: union and paired on data kept apart from the policy
    sel = [v for k, v in C.items() if isinstance(v, dict) and k.startswith(("boundary_cruise|", "boundary_coast|"))
           and k.endswith(("|union", "|paired")) and v.get("cov_pooled") is not None]
    if not sel:
        return ""
    idx = [v["cov_index_min"] for v in sel if v["cov_index_min"] is not None]
    fails = [v["failed"] for v in sel]
    return ((r" For the union and paired certificates, the lowest coverage at any decision index reached by at least 100"
             f" lives is {min(idx):.3f}," if idx else "") +
            f" and {100 * min(fails):.1f}--{100 * max(fails):.1f}\\,\\% of the lives fail.")


TS5 = r"""\begin{table}[htbp]
\centering
\caption{Certificate-triggered derating over leg lives under the certificate policies, truth at the margins (bench of 60
modules, 15 scores for the policy and 16 for the paired certificate, $n_2=50$ for each, $\alpha=0.10$, planned life 60
intervals, 300 lives per row, each with its own world, bench and field leg, common to all rows of a strategy). Damage at
replacement: mean true damage over the lives without failure. Derated: share of decisions with derating; false: share
of deratings that the true damage did not require; late: share of late sections; coverage per decision pooled over the
decisions and both modules, and share of lives in which every certificate of both modules covered.""" + ts5_note() + \
    r""" $^{\dagger}$Uses the truth; for analysis only.}
\label{tab:S5}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllcccccccc@{}}
\toprule
Driving & Policy & Certificate & \makecell{served\\intervals} & \makecell{damage at\\replacement}
& \makecell{derated\\(\%)} & \makecell{mean level\\(\%)} & \makecell{false\\(\%)} & \makecell{late\\(\%)} &
\makecell{coverage per\\decision} & \makecell{lives\\covered} \\
\midrule
""" + ts5_rows("boundary") + r"""
\bottomrule
\end{tabular*}
\end{table}
"""

# ---------------------------------------------------------------------------------------------------------------------
# Table S6. Stated components exceeded (Y), lot shift (L), harsher truths (A_adverse_strict_k0, A_beyond_k0)
Y = G6.get("Y", {})
L = G6.get("L", {})
Y_LAB = [("none", "none (truth at the margins)"), ("kappa", r"Miner factor 0.80"), ("thermal", "thermal scale 1.15"),
         ("heating", "heating-time slopes 0.6"), ("swing", "swing exponent below 26~K, $-0.8$"),
         ("kappa+thermal", "Miner factor and thermal scale"), ("kappa+thermal+heating", "and heating-time slopes"),
         ("all four", "all four")]
L_LAB = [("same", "same law"), ("mean -0.1", r"mean $-0.1$"), ("mean -0.2", r"mean $-0.2$"),
         ("scatter x1.5", r"scatter $\times1.5$"), ("scatter x2", r"scatter $\times2$")]
CN = ("U6 paired", "U5 union", "U2 Monte Carlo", "U3 Bayes")


def ts6_y():
    rows = []
    for k, lab in Y_LAB:
        v, w = Y.get(f"{k}|0.1"), Y.get(f"{k}|0.05")
        if v is None:
            rows.append(f"{lab} & " + " & ".join(["--"] * 6) + r" \\")
            continue
        rows.append(f"{lab} & {v['damage_ratio']:.2f} & " + " & ".join(
            f"{v['coverage'][c]:.3f} ({v['se'][c]:.3f})" if c == "U6 paired" else f"{v['coverage'][c]:.3f}" for c in CN) +
            f" & {w['coverage']['U6 paired']:.3f} \\\\")
    return "\n".join(rows)


def ts6_l():
    rows = []
    for k, lab in L_LAB:
        g = lambda nm, a="0.1": L.get(f"L_boundary|{k}|{a}|{nm}")
        if g("U6 paired") is None:
            rows.append(f"{lab} & " + " & ".join(["--"] * 8) + r" \\")
            continue
        cells = [f"{g(c)['calibrated']:.3f}" for c in CN] + [f"{g('U6 paired')['coverage']:.3f}"]
        for n in ("10", "20", "40"):
            v = g(f"recal {n}")
            cells.append(f"{v['calibrated']:.3f} ({f(v['width_vs_paired'])})")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def ts6_harsh():
    rows = []
    for tag, lab in (("A_adverse_strict_k0", "adverse"), ("A_beyond_k0", "beyond")):
        first = True
        for n1 in ("20", "40", "80"):
            for a in ("0.1", "0.05"):
                cells = []
                for c in ("U6 paired", "U5 union", "U2 Monte Carlo", "U3 Bayes"):
                    v = a_val(tag, n1, a, c)
                    cells.append("--" if v is None else (INF if v["finite"] < 0.5 else f(v["coverage"], 3)))
                o = a_val(tag, n1, a, "U6 paired")
                rows.append(f"{lab if first else ''} & {n1} & {a} & " + " & ".join(cells) +
                            f" & {'--' if o is None else f(o['calibrated'], 3)} & "
                            f"{'--' if o is None else f(o['ratio'])} \\\\")
                first = False
        rows.append(r"\addlinespace")
    return "\n".join(rows[:-1])


def ts6_count():
    rows = []
    for tag, lab in (("A_adverse_strict_k0", "adverse"), ("A_beyond_k0", "beyond")):
        v = VN.get(tag)
        if not v:
            rows.append(f"{lab} & -- & -- & -- & -- \\\\")
            continue
        cells = []
        for g in ("0", "1", "2", "3+"):
            x = v["coverage"].get(f"40|U6 paired|{g}")
            cells.append("--" if x is None else f"{x['covered'] / x['n']:.3f} ({x['n']})")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


TS6 = r"""\begin{table}[htbp]
\centering
\caption{(a) Stated components exceeded at once, from the truth at the margins ($n_1=40$, $n_2=100$, 200 repetitions
of 50 field legs): median true damage relative to no exceedance, and coverage of the true damage at $\alpha=0.10$, with
the standard error over repetitions for the paired certificate (also at $\alpha=0.05$); the rows below the single
components add them cumulatively. (b) Field lot whose log-strength law is moved or widened about its median against the
bench lot (truth at the margins, $n_1=40$, $n_2=100$, 200 repetitions of 200 field legs, $\alpha=0.10$): coverage of the
calibrated part, coverage of the true damage by the paired certificate, and the paired certificate recalibrated on
$n_{\mathrm{lot}}$ bench tests of the field lot, with its median width relative to the paired certificate of the bench
lot in parentheses. (c) Single certificates under the adverse and beyond truths ($n_2=100$, 200 repetitions of two
modules): coverage of the true damage, calibrated part and median ratio to the oracle of the paired certificate.
(d) Coverage of the paired certificate ($n_1=40$, $\alpha=0.10$) by the number of acting stated components that the world
and the module exceed, with the number of module records. All coverages pool the outer and the inner module.}
\label{tab:S6}
\footnotesize
\setlength{\tabcolsep}{3pt}
\textit{(a) Stated components exceeded}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
& & \multicolumn{4}{c}{Coverage, $\alpha=0.10$} & \\
\cmidrule(lr){3-6}
Exceeded & \makecell{true damage\\$\times$ none} & paired & union & \makecell{Monte\\Carlo} & Bayes &
\makecell{paired,\\$\alpha=0.05$} \\
\midrule
""" + ts6_y() + r"""
\bottomrule
\end{tabular*}

\medskip
\textit{(b) Lot shift}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}
\toprule
& \multicolumn{4}{c}{Calibrated part} & True damage & \multicolumn{3}{c}{Paired, recalibrated on $n_{\mathrm{lot}}$} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(l){7-9}
Field lot & paired & union & \makecell{Monte\\Carlo} & Bayes & paired & 10 & 20 & 40 \\
\midrule
""" + ts6_l() + r"""
\bottomrule
\end{tabular*}

\medskip
\textit{(c) Adverse and beyond truths}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}
\toprule
& & & \multicolumn{4}{c}{Coverage of the true damage} & \multicolumn{2}{c}{Paired} \\
\cmidrule(lr){4-7}\cmidrule(l){8-9}
Truth & $n_1$ & $\alpha$ & paired & union & \makecell{Monte\\Carlo} & Bayes & calibrated & ratio \\
\midrule
""" + ts6_harsh() + r"""
\bottomrule
\end{tabular*}

\medskip
\textit{(d) By the number of exceeded components}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
Truth & none & one & two & three or more \\
\midrule
""" + ts6_count() + r"""
\bottomrule
\end{tabular*}
\end{table}
"""

# ---------------------------------------------------------------------------------------------------------------------
# Table S7. Closed loop against the maintenance baselines, every truth and driving strategy, and the drift runs
T7_RUNS = [("reserve|paired", "paired, reserve rule"), ("reserve|union", "union, reserve rule"),
           ("reserve|mc", "Monte Carlo, reserve rule"), ("reserve_budget|paired", "paired, late budget"),
           ("reserve_budget|union", "union, late budget"),
           ("replace|paired", "paired, replacement only"), ("replace|union", "union, replacement only"),
           ("none|tbr60", "planned 60 intervals"),
           ("none|fix20_tbr60", r"20\,\% and planned 60"), ("none|cert_age", "commissioning certificate age"),
           ("none|mc_age", "Monte Carlo B10 age"), ("none|mc_opt10", r"Monte Carlo optimal age, $c_F=10$"),
           ("none|rtf", "run to failure")]
T7_GROUPS = [("boundary_cruise", "at the margins, cruising"), ("evidence_cruise", "evidence, cruising"),
             ("adverse_strict_cruise", "adverse, cruising"), ("beyond_cruise", "beyond, cruising"),
             ("boundary_coast", "at the margins, coasting"), ("evidence_coast", "evidence, coasting"),
             ("adverse_strict_coast", "adverse, coasting"), ("beyond_coast", "beyond, coasting"),
             ("drift", r"drift $\rho=0.2$, stated bound"), ("drift_allow", r"drift $\rho=0.2$, allowance 1.32"),
             ("drift04", r"drift $\rho=0.4$, stated bound"), ("drift04_allow", r"drift $\rho=0.4$, allowance 1.32")]
COST_KEYS = ("10|0.0", "50|0.0", "10|2.0", "50|2.0")


def ts7_rows(groups):
    rows = []
    for grp, glab in groups:
        first = True
        for run, lab in T7_RUNS:
            v = C.get(f"{grp}|{run}")
            if v is None:
                continue
            ci = v["failed_ci"]
            age = f" ({v['age_mean']:.0f})" if v.get("age_mean") is not None and run.startswith("none|") and \
                run not in ("none|tbr60", "none|fix20_tbr60", "none|rtf") else ""
            if first:
                rows.append(r"\multicolumn{10}{@{}l}{\textit{" + glab + r"}} \\")
            rows.append(f"{lab}{age} & {v['intervals_mean']:.1f} & {100 * v['failed']:.1f} "
                        f"({100 * ci[0]:.1f}--{100 * ci[1]:.1f}) & {f(v['cov_pooled'], 3)} & "
                        f"{100 * v['late_pooled']:.1f} & " +
                        " & ".join(f"{v['cost_per_100'][k]:.2f}" for k in COST_KEYS) + f" & {f(v['harvest_ok'])} \\\\")
            first = False
        if not first:
            rows.append(r"\addlinespace")
    return "\n".join(rows[:-1]) if rows else r"-- & & & & & & & & & & \\"


def ts7_age():
    A, rows = G6.get("AGE", {}), []
    for grp, glab in T7_GROUPS:
        b = A.get(f"{grp}|best_age")
        if not b:
            continue
        rows.append(f"{glab} & " + " & ".join(
            f"{b[r]['age']} & {100 * b[r]['failed']:.1f} & {b[r]['cost']:.2f}" for r in ("10", "50")) + r" \\")
    return "\n".join(rows) if rows else r"-- & & & & & & \\"


T7_HEAD = r"""\toprule
Policy (mean age) & \makecell{served\\intervals} & \makecell{failed (\%)} & \makecell{coverage per\\decision} &
\makecell{late\\(\%)} & \makecell{cost,\\$c_F=10$} & \makecell{cost,\\$c_F=50$} & \makecell{$c_F=10$,\\$c_L=2$} &
\makecell{$c_F=50$,\\$c_L=2$} & \makecell{damage at\\replacement} \\
\midrule
"""
TS7 = r"""\begin{table}[htbp]
\centering
\caption{Closed loop against the maintenance baselines (300 lives per truth and driving strategy, common to all rows of
a block, $\alpha=0.10$): served intervals, failed lives with Wilson 95\,\% interval, coverage per decision pooled over the
decisions and both modules, late sections, cost per 100 intervals in units of a planned replacement when a failure
costs $c_F$ and an interval with every section late costs $c_L$ (0 unless given), and mean true damage at replacement over
the lives without failure. Run to failure is censored at 240 intervals. (a) Cruising; (b) coasting; (c) drifting thermal
scale $1.10(1+\rho D^\star)$, truth at the margins, cruising; (d) cheapest fixed age without derating, from the
run-to-failure lives (true lives known, ex post).}
\label{tab:S7}
\scriptsize
\setlength{\tabcolsep}{2pt}
\textit{(a) Cruising}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}
""" + T7_HEAD + ts7_rows(T7_GROUPS[:4]) + r"""
\bottomrule
\end{tabular*}
\end{table}

\begin{table}[htbp]
\ContinuedFloat
\centering
\caption{(continued)}
\scriptsize
\setlength{\tabcolsep}{2pt}
\textit{(b) Coasting}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}
""" + T7_HEAD + ts7_rows(T7_GROUPS[4:8]) + r"""
\bottomrule
\end{tabular*}
\end{table}

\begin{table}[htbp]
\ContinuedFloat
\centering
\caption{(continued)}
\scriptsize
\setlength{\tabcolsep}{2pt}
\textit{(c) Drifting thermal scale}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}
""" + T7_HEAD + ts7_rows(T7_GROUPS[8:]) + r"""
\bottomrule
\end{tabular*}

\medskip
\textit{(d) Cheapest fixed age, true lives known}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
& \multicolumn{3}{c}{$c_F=10$} & \multicolumn{3}{c}{$c_F=50$} \\
\cmidrule(lr){2-4}\cmidrule(l){5-7}
Truth & age & failed (\%) & cost & age & failed (\%) & cost \\
\midrule
""" + ts7_age() + r"""
\bottomrule
\end{tabular*}
\end{table}
"""

# ---------------------------------------------------------------------------------------------------------------------
# Table S9. Information loop of the thermal bound (thu_nghiem_g7.py, g7_tong_hop.json; Section 6.8)
G7_LAB = {"stated": "stated bound 1.10", "stated_bal": "stated bound, balancing", "op": "operation only",
          "op_bal": "operation only, balancing", "pulse_gated": "pulses gated, reserve 10--30",
          "pulse_gated_min": "gated 10--30, at least 60", "pulse_gated_plan": "pulses gated, reserve 30--60",
          "pulse_gated_plan_min": "gated 30--60, at least 60", "pulse_always": "300 pulses"}


def ts9_lives():
    rows = []
    for tag, lab in (("L_evidence", "evidence"), ("L_boundary", "at the margins")):
        res = G7.get(tag, {})
        first = True
        for c, clab in G7_LAB.items():
            v = res.get(c)
            if v is None:
                continue
            rows.append(f"{lab if first else ''} & {clab} & {v['intervals_mean']:.1f} & {100 * v['failed']:.0f} & "
                        f"{100 * v['collapse']:.0f} & {v['cov_life']:.3f} & {v['th_cov'][0]:.2f}/{v['th_cov'][1]:.2f} & "
                        f"{v['omega'][1]:.3f} & {v['omega_max'][1]:.3f} & {v['pulses']:.0f} & "
                        f"{100 * v['pulse_model']:.1f}/{100 * v['pulse_true']:.1f} \\\\")
            first = False
        rows.append(r"\addlinespace")
    return "\n".join(rows[:-1]) if rows else r"-- & & & & & & & & & & \\"


def ts9_phase():
    rows = []
    ph = G7.get("L_evidence|phase", {})
    for c in ("pulse_gated_plan", "pulse_gated_plan_min", "op", "op_bal"):
        cells = []
        for age in ("start", "mid"):
            v = ph.get(f"{age}|{c}")
            if v is None:
                cells.append("--")
                continue
            fp = v["fixed_T2"]
            cells.append(", ".join(f"{1.645 / math.sqrt(x):.2f}" for x in sorted(fp, reverse=True)) if fp else "none")
        rows.append(f"{G7_LAB[c]} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def ts9_sens():
    S, rows = G7.get("sens", {}), []
    for name, lab in (("z of the thermal bound", "$z$"), ("drift of the thermal path per unit damage (boundary)",
                                                           r"drift $\rho$"),
                      ("minimum dwell pulses per interval", "minimum pulses")):
        keys = [k for k in S if k.startswith(name + "|")]
        for k in keys:
            _, val, c = k.split("|")
            v = S[k]
            rows.append(f"{lab} {val} & {G7_LAB.get(c, c)} & {v['intervals']:.1f} & {100 * v['failed']:.0f} & "
                        f"{v['cov_life']:.3f} & {v['th_cov'][0]:.2f}/{v['th_cov'][1]:.2f} & {v['omega']:.3f} & "
                        f"{v['omega_max']:.3f} & {100 * v['pulse']:.1f} \\\\")
    lam = G7.get("lambda", {})
    for k, v in lam.items():
        l_, c = k.split("|")
        rows.append(f"forgetting {l_} & {G7_LAB.get(c, c)} & {v['intervals']:.1f} & -- & -- & -- & -- & "
                    f"{v['omega_max']:.3f} & -- \\\\")
    return "\n".join(rows) if rows else r"-- & & & & & & & & \\"


TS9 = r"""\begin{table}[htbp]
\centering
\caption{Information loop of the thermal bound (Section~\ref{sec:infoloop}; bench of 60 modules, $n_2=50$,
$\alpha=0.10$, thermal scale of each module rising as $1+0.2D^\star$). (a) Leg lives per configuration (100 worlds per
truth): served intervals, failed and collapsed lives, share of lives in which every certificate covered, share of
intervals in which the bound covers the true thermal scale of the outer/inner module, mean and largest bound width of
the inner module (median over lives), test pulses per interval, and share of the pulses in the margined/true damage.
(b) Phase map of the information of the inner module, evidence truth: bound widths at the fixed points of the median map
for a new leg and at mid-life. (c) Sensitivity, continued on the next page.}
\label{tab:S9}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\textit{(a) Lives}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccccccccc@{}}
\toprule
Truth & Configuration & \makecell{served\\intervals} & \makecell{failed\\(\%)} & \makecell{collapsed\\(\%)} &
\makecell{lives\\covered} & \makecell{bound\\covers} & \makecell{width,\\mean} & \makecell{width,\\largest} &
\makecell{pulses per\\interval} & \makecell{pulse share\\(\%)} \\
\midrule
""" + ts9_lives() + r"""
\bottomrule
\end{tabular*}

\medskip
\textit{(b) Fixed points of the phase map (bound width)}\\[2pt]
\begin{tabular*}{0.7\textwidth}{@{\extracolsep{\fill}}lcc@{}}
\toprule
Configuration & new leg & mid-life \\
\midrule
""" + ts9_phase() + r"""
\bottomrule
\end{tabular*}
\end{table}

% (c) as a continuation without its own caption (19/10/2026): the whole table overran the page by 173 pt
\begin{table}[htbp]
\centering
\small Table~\ref{tab:S9} (continued). (c) Sensitivity on the same 30 worlds: confidence $z$, drift per unit damage,
minimum dwell-window pulses and forgetting factor.\par\medskip
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccccccc@{}}
\toprule
Value & Configuration & \makecell{served\\intervals} & \makecell{failed\\(\%)} & \makecell{lives\\covered} &
\makecell{bound\\covers} & \makecell{width,\\mean} & \makecell{width,\\largest} & \makecell{pulse share\\(\%)} \\
\midrule
""" + ts9_sens() + r"""
\bottomrule
\end{tabular*}
\end{table}
"""

# ---------------------------------------------------------------------------------------------------------------------
# Table S10. Test of re-simulation (experiment R)
RR = G6.get("R", {})
R_LAB = {"base": "base (control)", "in": "inner pair relieved", "out": "outer pair relieved",
         "coast": "coasting"}


def ts10_rows():
    rows = []
    for tag, lab in (("R_boundary", "at the margins"), ("R_evidence", "evidence")):
        rows.append(r"\multicolumn{8}{@{}l}{\textit{Truth " + lab + r"}} \\")
        for pat in ("base", "in", "out", "coast"):
            for mod in ("outer", "inner"):
                v = RR.get(f"{tag}|{pat}|{mod}|0.1")
                w = RR.get(f"{tag}|{pat}|{mod}|0.05")
                if v is None:
                    continue
                rows.append(f"{R_LAB[pat] if mod == 'outer' else ''} & {mod} & "
                            f"{v['damage_ratio']:.2f} & {v['V_ratio']:.2f} & {v['cov_recorded']:.3f} "
                            f"({v['ci_recorded'][0]:.3f}--{v['ci_recorded'][1]:.3f}) & {v['cov_resim']:.3f} & "
                            f"{f(None if w is None else w['cov_recorded'], 3)} & {f(None if w is None else w['cov_resim'], 3)} \\\\")
        rows.append(r"\addlinespace")
    return "\n".join(rows[:-1]) if rows else r"-- & & & & & & & & \\"


TS10 = r"""\begin{table}[htbp]
\centering
\caption{Test of re-simulation: the coming interval runs under a changed policy while the recorded intervals ran under
the base policy (cruising, base balancing pattern); paired certificate of each module on the recorded intervals as they
are and re-simulated under the changed policy ($n_1=40$, $n_2=100$, 200 repetitions). True damage: median over the
changed to the base policy; recorded damage: median of the re-simulated over the recorded margined damage; coverage of the
true damage with Wilson 95\,\% interval for the recorded intervals at $\alpha=0.10$.}
\label{tab:S10}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccc@{}}
\toprule
& & & & \multicolumn{2}{c}{$\alpha=0.10$} & \multicolumn{2}{c}{$\alpha=0.05$} \\
\cmidrule(lr){5-6}\cmidrule(l){7-8}
Changed policy & Module & \makecell{true damage\\$\times$ base} & \makecell{recorded damage\\$\times$ base} &
recorded & re-simulated & recorded & re-simulated \\
\midrule
""" + ts10_rows() + r"""
\bottomrule
\end{tabular*}
\end{table}
"""

for name, text in (("T1.tex", T1), ("T3.tex", T3), ("T5.tex", T5)):
    write(name, text)
SUPP = (("TS2.tex", TS2), ("TS3.tex", TS3), ("TS4.tex", TS4), ("TS5.tex", TS5), ("TS6.tex", TS6), ("TS7.tex", TS7),
        ("TS8.tex", TS8), ("TS9.tex", TS9), ("TS10.tex", TS10))
for name, text in SUPP:
    write(name, text)

# ---------------------------------------------------------------------------------------------------------------------
# Review document: final figures with the captions of the manuscript, then the tables. The captions are read from the
# manuscript (../P3_RESS/sec*.tex) when it is present, so that they cannot drift apart.
FIG_ORDER = [("F1", "fig:p3-F1"), ("F2", "fig:p3-F2"), ("F3", "fig:p3-F3"), ("F4", "fig:p3-F4"), ("F5", "fig:p3-F5"),
             ("F6", "fig:p3-F6"), ("F7", "fig:p3-F7"), ("F8", "fig:p3-F8"), ("F9", "fig:p3-F9")]


def manuscript_captions():
    """Figure captions of the manuscript (../P3_RESS/sec*.tex) by label; empty outside the project."""
    caps = {}
    for p in sorted(RESS.glob("sec*.tex")) if RESS.is_dir() else []:
        t = p.read_text(encoding="utf-8")
        for m in re.finditer(r"\\caption\{", t):
            depth, j = 1, m.end()
            while depth:
                depth += {"{": 1, "}": -1}.get(t[j], 0)
                j += 1
            lab = re.match(r"\s*\\label\{(fig:[^}]+)\}", t[j:])
            if lab:
                caps[lab.group(1)] = t[m.end():j - 1].strip()
    return caps


MS_CAPS = manuscript_captions()
fig_tex = "\n".join(
    "\\begin{figure}[p]\n\\centering\n\\makebox[\\textwidth][c]{\\includegraphics{" + name + ".pdf}}\n"
    "\\caption{" + MS_CAPS.get(lab, "(caption in the manuscript)") + "}\n\\label{" + lab + "}\n\\end{figure}\n"
    for name, lab in FIG_ORDER)

DOC = r"""\documentclass[a4paper,10pt]{article}
\usepackage[margin=12mm,top=15mm,bottom=15mm]{geometry}
\usepackage{newtxmath}
\usepackage[no-math]{fontspec}
\setmainfont{TeX Gyre Termes}
\usepackage{graphicx,booktabs,tabularx,array,makecell,caption}
\captionsetup{font=small,labelfont=bf,labelsep=period}
\usepackage[backend=biber,style=numeric-comp,sorting=none,giveninits=true,maxbibnames=5,doi=false,url=false,isbn=false,
  date=year]{biblatex}
\addbibresource{../tailieu.bib}
\DeclareSourcemap{\maps[datatype=bibtex]{\map{\step[fieldset=addendum,null]}}}
\renewcommand{\thefigure}{F\arabic{figure}}
\renewcommand{\thetable}{T\arabic{table}}
% labels of the manuscript that the captions cite
\makeatletter
\newcommand{\fakelabel}[2]{\expandafter\gdef\csname r@#1\endcsname{{#2}{}{}{}{}}}
\makeatother
\fakelabel{thm:p3-pair}{1}
\fakelabel{prop:p3-info}{9}
\fakelabel{sec:infoloop}{6.8}
\fakelabel{sec:sens}{6.9}
\fakelabel{sec:bench}{6.1}
\fakelabel{sec:certs}{2.4}
\title{Conformal mission reserve certificates: figures and tables}
\author{Overview of the figures and tables of the manuscript, with their captions}
\date{\today}
\begin{document}
\maketitle
\noindent Sources of the final figures and tables, all in \texttt{P3\_hinh\_bang/}:
\begin{itemize}\setlength{\itemsep}{0pt}
\item F1: \texttt{F1\_cau\_truc.tex} (TikZ, compiled with xelatex);
\item F2 and F4--F9: \texttt{P3\_mophong/hinh\_cuoi.py}; F3: \texttt{P3\_kiemtoan/hinh\_cuoi\_f3.py};
\item T1, T3 and T5: \texttt{P3\_mophong/bang\_cuoi.py}, one table environment per file, for \verb|\input| in the
manuscript; Tables S2--S10 of the Supplementary Material come from the same script.
\end{itemize}
Figures are vector PDF at print size (90, 140 or 190~mm wide, text 7--8~pt), with 300~dpi PNG previews. Every number
is read from the saved results; nothing is re-simulated. In the manuscript, F1--F9 are Figures 1, 3, 2, 4, 5, 7, 8, 9 and 6,
and T1, T5 and T3 are Tables 2--4. Since 18/10/2026 the case study uses the 3.3~kV converter and certificates per module.
\clearpage
""" + fig_tex + r"""
\clearpage
\input{T1}
\input{T3}
\input{T5}
\clearpage
\printbibliography
\end{document}
"""
write("P3_hinh_bang_cuoi.tex", DOC)

# Copies for the RESS manuscript in ../P3_RESS (Elsevier CAS template: float placement through the pos key)
if RESS.is_dir():
    for name, text in (("T1.tex", T1), ("T5.tex", T5)):
        (RESS / name).write_text(text.replace(r"\begin{table}[tbp]", r"\begin{table}[pos=!tbp]"), encoding="utf-8",
                                 newline="\n")
        print("written", RESS / name)
    # 20/10/2026 (word limit): the coverage and tightness of single certificates (formerly Table 4 of the manuscript)
    # is Table S28 of the Supplementary Material, at its end; its label stays tab:p3-T3, so the supplement refers to it
    (RESS / "TS28.tex").write_text(T3.replace(r"\begin{table}[tbp]", r"\begin{table}[htbp]"), encoding="utf-8",
                                   newline="\n")
    print("written", RESS / "TS28.tex")
    for name, text in SUPP:    # Supplementary Material (supp.tex, article class)
        (RESS / name).write_text(text, encoding="utf-8", newline="\n")
        print("written", RESS / name)
