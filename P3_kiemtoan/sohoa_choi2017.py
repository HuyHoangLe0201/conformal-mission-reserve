"""G2: per-module cycles to failure from Choi, Blaabjerg, Jorgensen (2017), IEEE TPEL 32(8):6434-6443, Fig. 7.

Fig. 7 has six Weibull probability plots, one per junction-temperature swing duration t_dTj = 1/f_out (600 V/30 A
transfer-molded IPM, dTj about 81 C, Tjm about 102 C). Each data point (blue dot) sits at x = cycles to failure (log
axis, 80k..300k).
The panels were extracted from the accepted manuscript into G2_nguon/choi2017_fig7_*.png.

Method: plot frame from its grey border lines (80k at the left edge, 300k at the right edge; the 100k gridline
falls where the log axis puts it); blue mask with legend and labels removed; dots found as 7x7 windows dense in blue
pixels, which the 2-px probability line never fills; dot centre -> cycles through the log axis. Two dots drawn under
a red confidence line turn purple and were located by eye on enlargements (source = "visual").
Check: 2-parameter Weibull refit (median-rank regression on X, as in Weibull++ defaults) compared with the beta and
eta printed in each panel.

Run: python sohoa_choi2017.py   (numpy, PyMuPDF). Writes G2_choi2017_sohoa.csv.
"""
import csv
import math
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())  # keeps the script runnable with python -I
import fitz  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
# dots drawn under a red confidence line turn purple and escape the blue mask; their x centres (pixels) were read
# on 8x enlargements of the panels
VISUAL = {2: [306.0], 4: [397.1]}
PANELS = {  # condition: (swing duration t_dTj in s, beta and eta printed in the panel, expected number of dots)
    1: (10.0, 13.52, 150943, 6), 2: (5.0, 17.22, 176376, 6), 3: (2.0, 24.05, 190998, 7),
    4: (1.0, 23.83, 201254, 8), 5: (0.8, 23.97, 211141, 6), 6: (0.59, 16.1, 235775, 6),
}


def load(path):
    pix = fitz.Pixmap(str(path))
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)


def frame(img):
    """Plot frame: the first and last groups of vertical lines (x > 30 px, past the rotated y label) that are
    non-white across most of a band through the plot area; the 100k gridline falls in between."""
    H, W, _ = img.shape
    nonwhite = img.sum(axis=2) < 720
    band = nonwhite[int(0.3 * H):int(0.6 * H)]
    cols = [x for x in range(30, W) if band[:, x].mean() > 0.6]
    groups = []
    for x in cols:
        if groups and x - groups[-1][-1] <= 1:
            groups[-1].append(x)
        else:
            groups.append([x])
    x0, x1 = float(np.mean(groups[0])), float(np.mean(groups[-1]))
    inner = nonwhite[:, int(x0) + 3:int(x1) - 2]
    rows = [y for y in range(H) if inner[y].mean() > 0.6]
    return x0, x1, min(rows), max(rows)


def dot_centres(mask, thr=22):
    """Centres of filled dots (about 7 px wide): 7x7 windows holding at least `thr` mask pixels, clustered,
    then refined by the centroid of mask pixels around the densest window. A 2-px line never reaches `thr`."""
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(mask.astype(np.int16), (7, 7)).sum(axis=(2, 3))
    clusters = []
    for r, c in np.argwhere(win >= thr):
        cy, cx, s = r + 3, c + 3, int(win[r, c])
        for cl in clusters:
            if abs(cl[0] - cy) <= 4 and abs(cl[1] - cx) <= 4:
                cl[2].append((cy, cx, s))
                break
        else:
            clusters.append([cy, cx, [(cy, cx, s)]])
    refined = []
    for _, _, members in clusters:
        by, bx, s = max(members, key=lambda m: m[2])
        ys, xs = np.nonzero(mask[by - 5:by + 6, bx - 5:bx + 6])
        refined.append((by - 5 + ys.mean(), bx - 5 + xs.mean(), s))
    # one dot can split into neighbouring clusters: keep the densest of centres closer than 4 px
    out = []
    for cy, cx, s in sorted(refined, key=lambda t: -t[2]):
        if all(abs(cy - oy) > 4 or abs(cx - ox) > 4 for oy, ox in out):
            out.append((cy, cx))
    return out


def weibull_rrx(t):
    """Median-rank regression on X (Benard ranks), the Weibull++ default for complete data."""
    t = np.sort(np.asarray(t, dtype=float))
    n = len(t)
    F = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
    y = np.log(-np.log(1 - F))
    x = np.log(t)
    b, a = np.polyfit(y, x, 1)  # x = a + b y  ->  beta = 1/b, eta = exp(a)
    return 1 / b, math.exp(a)


records = []
print(f"{'cond':>4} {'t_dTj':>5} {'dots':>4}/{'exp':<3} {'frame x':>11} {'beta fig':>8} {'beta refit':>10}"
      f" {'eta fig':>8} {'eta refit':>9}  Nf (cycles)")
for cond, (ton, beta_fig, eta_fig, n_exp) in PANELS.items():
    img = load(HERE / "G2_nguon" / f"choi2017_fig7_{cond}.png")
    x0, x1, y0, y1 = frame(img)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    blue = (b > r + 60) & (b > g + 40)
    blue[: y0 + 2, :] = False
    blue[y1 - 1:, :] = False
    blue[:, : int(x0) + 2] = False
    blue[:, int(x1) - 1:] = False
    # the legend (with a blue dot and a blue line sample) sits inside the frame: lower right in panel 1,
    # lower left in panels 2-6; the condition label sits at the upper left
    H, W, _ = img.shape
    if cond == 1:
        blue[int(0.49 * H):int(0.82 * H), int(0.67 * W):] = False
    else:
        blue[int(0.46 * H):int(0.82 * H), :int(0.42 * W)] = False
    blue[:int(0.25 * H), :int(0.45 * W)] = False
    found = [(x, "auto") for _, x in dot_centres(blue)]
    found += [(x, "visual") for x in VISUAL.get(cond, [])]
    found.sort()
    lg0, lg1 = math.log10(80e3), math.log10(300e3)
    nf = [(10 ** (lg0 + (x - x0) / (x1 - x0) * (lg1 - lg0)), src) for x, src in found]
    vals = [v for v, _ in nf]
    beta_r, eta_r = weibull_rrx(vals) if len(vals) >= 3 else (float("nan"), float("nan"))
    print(f"{cond:>4} {ton:>5} {len(vals):>4}/{n_exp:<3} {x0:>5}-{x1:<5} {beta_fig:>8.2f} {beta_r:>10.2f}"
          f" {eta_fig:>8,} {eta_r:>9,.0f}  " + ", ".join(f"{v:,.0f}" for v in vals))
    for v, src in nf:
        records.append(dict(condition=cond, t_dTj_s=ton, Nf=int(round(v, -2)), source=src))

# dTj and Tjm per condition from Table II of the paper
TABLE2 = {1: (80.8, 102.3), 2: (80.6, 102.5), 3: (82.0, 101.3), 4: (81.6, 101.5), 5: (81.8, 102.4), 6: (80.8, 102.0)}
for rec in records:
    rec["dTj_C"], rec["Tjm_C"] = TABLE2[rec["condition"]]
out = HERE / "G2_choi2017_sohoa.csv"
with open(out, "w", newline="\n", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=["condition", "t_dTj_s", "dTj_C", "Tjm_C", "Nf", "source"], lineterminator="\n")
    wr.writeheader()
    wr.writerows(records)
print("written", out.name, len(records), "rows")
