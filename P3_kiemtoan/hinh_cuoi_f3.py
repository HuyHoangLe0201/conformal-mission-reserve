"""P3: final Figure F3 (empirical tests of the reference-scatter assumption, the stated margin and Miner's rule).

Same data and computations as section D of g3_tref_bien.py (draft F3_nhap), redrawn in the style of the final figures
(../P3_mophong/kieu_hinh.py) with four panels:
  (a) leave-one-out log-strength of the Hernes et al. (2021) modules against the swing, with the stated swing margin
      below 26 K and the statistical part of the model at the clipped field point;
  (b) life against heating time in tests that vary the heating time alone (Junghaenel and Scheuermann 2017, Table 1;
      Choi et al. 2017), normalized at 2 s, with the stated slope 0.5 and its sensitivity range;
  (c) residual log-lives of one module type in two campaigns (Choi et al. 2017 and 2018): the scatter differs;
  (d) Miner sum at failure under mixed loading (Hernes et al., PCIM 2021) against the constant-amplitude runs, with
      the conformal lower 10 % bound for kappa_M = 1 and 0.85.

Run: python -I hinh_cuoi_f3.py   -> ../P3_hinh_bang/F3.pdf, .png
"""
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "P3_mophong"))
import kieu_hinh as kh  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

rng = np.random.default_rng(20261016)
Z95 = 1.645
DT_T, TON_T, TON_FIELD, I_LOW = 26.0, 4.5, 4.5, 890.0


def read(name):
    with open(HERE / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def design(rows):
    return np.column_stack([np.ones(len(rows)), np.log([r["dT"] for r in rows]), np.log([r["ton"] for r in rows]),
                            np.log([r["I"] for r in rows])])


H = [dict(run=int(r["run"]), I=float(r["I_load_A"]), ton=float(r["t_on_s"]), dT=float(r["dT_C"]), Nf=float(r["Nf"]))
     for r in read("G2_hernes2021_sohoa.csv")]
runs = np.array([r["run"] for r in H])
X, y = design(H), np.log([r["Nf"] for r in H])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
res = y - X @ beta
dof = len(y) - X.shape[1]
cov = float(res @ res / dof) * np.linalg.inv(X.T @ X)
S = res / (1 - np.diag(X @ np.linalg.inv(X.T @ X) @ X.T))        # leave-one-out residuals

fig, axs = kh.figure(kh.W15, 112, 2, 2)
RUNCOL = {1: kh.BLUE, 2: kh.VERMILION, 3: kh.ORANGE, 4: kh.PURPLE, 5: kh.GREEN}

# (a) swing
ax = axs[0, 0]
dts = np.array([h["dT"] for h in H])
for r in range(1, 6):
    ax.scatter(dts[runs == r], S[runs == r], s=9, color=RUNCOL[r], label=f"run {r}", zorder=3)
xx = np.linspace(8, DT_T, 50)
for g, ls in ((0.6, ":"), (1.2, "--")):
    ax.plot(xx, -g * np.log(DT_T / xx), "k", ls=ls, lw=0.8, label=fr"swing margin, $\gamma$ = {g}")
grid = np.geomspace(8, 74, 60)
se = np.array([math.sqrt(float(v @ cov @ v)) for v in
               (np.array([1, math.log(d), math.log(TON_FIELD), math.log(I_LOW)]) for d in grid)])
ax.plot(grid, -Z95 * se, color=kh.GREY, ls="-.", lw=0.8, label="statistical part, 4.5 s, 890 A")
ax.axvline(DT_T, color=kh.GREY, lw=0.5)
ax.axhline(0, color=kh.GREY, lw=0.4)
ax.set_xscale("log")
ax.set_xticks([8, 10, 15, 20, 26, 40, 70], ["8", "10", "15", "20", "26", "40", "70"])
ax.minorticks_off()
ax.set_xlabel(r"junction swing $\Delta T$ (K)")
ax.set_ylabel("log-strength $S$ (leave-one-out)")
ax.set_ylim(-1.6, 0.7)
h_, l_ = ax.get_legend_handles_labels()
leg1 = ax.legend([h for h, l in zip(h_, l_) if l.startswith("run")], [l for l in l_ if l.startswith("run")],
                 loc="upper left", fontsize=6.5, ncol=3, handletextpad=0.2, columnspacing=0.6)
ax.add_artist(leg1)
ax.legend([h for h, l in zip(h_, l_) if not l.startswith("run")], [l for l in l_ if not l.startswith("run")],
          loc="lower right", fontsize=6.5)

# (b) heating time
ax = axs[0, 1]
SC = read("G10_junghaenel2017_bang1.csv")
mk = {("70", "1"): ("s", kh.VERMILION, "wire bond, 70 K"), ("70", "2"): ("D", kh.BLUE, "solder, 70 K"),
      ("110", "1"): ("s", kh.ORANGE, "wire bond, 110 K"), ("110", "2"): ("D", kh.SKY, "solder, 110 K")}
for (d, gr), (m_, c_, lab) in mk.items():
    pts = [(float(s["t_on_s"]), float(s["Nf"])) for s in SC if s["dTj_K"] == d and s["group"] == gr]
    ref = math.exp(np.mean([math.log(n) for t, n in pts if abs(t - 2) < 0.1]))
    ax.scatter([t for t, _ in pts], [n / ref for _, n in pts], marker=m_, s=9, color=c_, label=lab, zorder=3)
C17 = read("G2_choi2017_sohoa.csv")
c17t = np.array([float(c["t_dTj_s"]) for c in C17])
c17n = np.array([float(c["Nf"]) for c in C17])
ref17 = math.exp(np.mean(np.log(c17n[np.isclose(c17t, 2.0)])))
ax.scatter(c17t, c17n / ref17, marker="o", s=8, facecolors="none", edgecolors=kh.GREY, linewidths=0.6,
           label="IPM (Choi 2017)", zorder=3)
ax.axvspan(0.15, TON_T, color="0.93", zorder=0, lw=0)
ref_45 = (TON_T / 2) ** (-0.463)        # the stated rule starts at the longest tested heating time
tt = np.geomspace(TON_T, 120, 40)
for g, ls in ((0.29, ":"), (0.5, "--"), (1.0, "-.")):
    ax.plot(tt, ref_45 * (tt / TON_T) ** (-g), "k", ls=ls, lw=0.8, label=f"stated slope {g}" if g == 0.5
            else f"slope {g}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(0.15, 30)
ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
ax.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20], ["0.2", "0.5", "1", "2", "5", "10", "20"])
ax.minorticks_off()
ax.set_xlabel(r"heating time $t_{\mathrm{on}}$ (s)")
ax.set_ylabel(r"$N_f / N_f(2\,\mathrm{s})$")
h_, l_ = ax.get_legend_handles_labels()
is_line = ["slope" in l for l in l_]
leg_b = ax.legend([h for h, k in zip(h_, is_line) if k], [l for l, k in zip(l_, is_line) if k], loc="lower left",
                  fontsize=6.5)
ax.add_artist(leg_b)
ax.legend([h for h, k in zip(h_, is_line) if not k], [l for l, k in zip(l_, is_line) if not k], loc="upper left",
          bbox_to_anchor=(0.2, 1.0), fontsize=6.5, handletextpad=0.3)

# (c) one module type, two campaigns
ax = axs[1, 0]
xc, yc = np.log(c17t), np.log(c17n)
slc = stats.linregress(xc, yc)
r17 = yc - (slc.intercept + slc.slope * xc)
c18 = read("G2_choi2018_bang2.csv")
x18 = np.log([float(c["dTj_C"]) for c in c18])
y18 = np.log([float(c["Nf"]) for c in c18])
fail = np.array([c["failed"] == "1" for c in c18])
s18 = stats.linregress(x18[fail], y18[fail])
r18 = y18 - (s18.intercept + s18.slope * x18)
cond17 = sorted(set(c17t), reverse=True)
for i, t in enumerate(cond17):
    v = r17[np.isclose(c17t, t)]
    ax.scatter(np.full(v.size, i) + rng.uniform(-0.15, 0.15, v.size), v, s=8, color=kh.BLUE, zorder=3)
d18 = [70, 60, 40]
for j, d in enumerate(d18):
    sel = np.isclose(np.exp(x18), d)
    xs_ = np.full(sel.sum(), len(cond17) + 1 + j) + rng.uniform(-0.15, 0.15, sel.sum())
    ax.scatter(xs_[fail[sel]], r18[sel][fail[sel]], s=8, color=kh.VERMILION, zorder=3)
    ax.scatter(xs_[~fail[sel]], r18[sel][~fail[sel]], s=11, facecolors="none", edgecolors=kh.VERMILION, marker="^",
               linewidths=0.6, zorder=3)
ax.set_xticks(list(range(len(cond17))) + [len(cond17) + 1 + j for j in range(3)],
              [f"{t:g} s" for t in cond17] + [f"{d} K" for d in d18], rotation=45, fontsize=6.5)
ax.axhline(0, color=kh.GREY, lw=0.4)
ax.set_ylim(-1.72, 0.55)
ax.set_ylabel(r"residual of $\log N_f$")
ax.text(0.02, 0.97, "2017: heating time varied", transform=ax.transAxes, fontsize=6.5, color=kh.BLUE, va="top")
ax.text(0.98, 0.97, "2018: swing varied", transform=ax.transAxes, fontsize=6.5, color=kh.VERMILION, va="top",
        ha="right")
ax.text(0.02, 0.04, r"$\triangle$ suspended module (lower bound)", transform=ax.transAxes, fontsize=6.5)

# (d) Miner sum at failure
ax = axs[1, 1]
P = read("G2_hernes2021pcim_sohoa.csv")
Wt = {t: np.array([float(p["damage"]) for p in P if int(p["test"]) == t]) for t in range(1, 8)}
const = np.concatenate([Wt[t] for t in range(1, 6)])
qW = float(np.sort(const)[math.floor(0.10 * (const.size + 1)) - 1])
for t in range(1, 8):
    v = Wt[t]
    col = RUNCOL.get(t, kh.GREY)
    ax.scatter(np.full(v.size, t) + rng.uniform(-0.15, 0.15, v.size), v, s=9, color=col, zorder=3,
               marker="o" if t <= 5 else ("D" if t == 6 else "^"))
ax.axhline(qW, color="k", lw=0.8, label=f"lower 10 % bound, Miner factor 1 ({qW:.2f})")
ax.axhline(0.85 * qW, color="k", ls="--", lw=0.8, label=f"with Miner factor 0.85 ({0.85 * qW:.2f})")
ax.axvline(5.5, ymax=0.77, color=kh.GREY, lw=0.4)  # stops below the legend
ax.set_xticks(range(1, 8), ["1", "2", "3", "4", "5", "40 then\n70 K", "70 then\n40 K"], fontsize=6.5)
ax.set_xlabel("constant-amplitude runs 1-5;  mixed loading")
ax.set_ylabel("Miner sum at failure")
ax.set_ylim(0.65, 1.62)
ax.legend(loc="upper left", fontsize=6.5)

for a, l in zip(axs.flat, "abcd"):
    kh.label(a, l, x=-0.2)
fig.tight_layout(h_pad=1.2, w_pad=1.5)
kh.save(fig, "F3")
