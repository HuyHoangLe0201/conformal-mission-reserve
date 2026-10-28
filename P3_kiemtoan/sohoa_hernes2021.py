"""G2: digitized per-module cycles to failure from Hernes et al. (2021), IET Power Electronics 14(7):1271-1283.

Source figure: "Recorded Nf versus dT for the five test runs" (the scatter without fitted curves), extracted from the
published PDF (CC BY 4.0) into G2_nguon/hernes2021_fig_Nf_dT.png. Axes are log-log; one MATLAB default colour per run.

Method:
  1. axis calibration from gridlines (dT = 30..100 on x, Nf = 1e5, 1e6, 1e7 on y), least squares in log scale;
  2. colour mask per run, 4-connected fragments (black marker outlines separate overlapping markers),
     centre of each marker from its un-occluded edges;
  3. three markers that show only thin slivers were located by eye on 6-7x enlargements (source = "visual").
Two markers (one in run 1, one in run 4) are fully hidden and cannot be recovered.

Check: the digitized extremes are compared with Table 5 of the paper (min/max of dT and Nf per run).

Run: python sohoa_hernes2021.py   (numpy, scipy, PyMuPDF). Writes G2_hernes2021_sohoa.csv.
"""
import csv
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())  # keeps the script runnable with python -I
import fitz  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).parent
IMG = HERE / "G2_nguon" / "hernes2021_fig_Nf_dT.png"

# test conditions, Table 2 of the paper (Tj,min = 60 C in all runs)
RUNS = {1: dict(I=1250, ton=0.75, dT_nom=70, n_eol=8), 2: dict(I=1250, ton=0.25, dT_nom=40, n_eol=8),
        3: dict(I=1250, ton=0.15, dT_nom=30, n_eol=6), 4: dict(I=890, ton=4.5, dT_nom=70, n_eol=8),
        5: dict(I=890, ton=1.1, dT_nom=40, n_eol=6)}
# Table 5: recorded span of dT [C] and Nf [cycles]
TABLE5 = {1: (65.4, 70.9, 351900, 528051), 2: (35.1, 40.3, 5562900, 7723300), 3: (25.9, 29.7, 18602000, 28932000),
          4: (68.2, 74.0, 101290, 153490), 5: (38.3, 41.4, 2341900, 3218636)}
COLORS = {1: (0, 114, 189), 2: (217, 83, 25), 3: (237, 177, 32), 4: (126, 47, 142), 5: (119, 172, 48)}
LEGEND = (720, 20, 970, 210)
# markers seen only as slivers, located on enlargements (pixel centres)
VISUAL = {1: [(772.5, 493.0)], 2: [(476.8, 216.3)]}

pix = fitz.Pixmap(str(IMG))
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)
H, W, _ = img.shape

XT = np.array([[30, 314.5], [40, 472.5], [50, 595.5], [60, 695.5], [70, 780.5], [80, 853.5], [100, 976.0]])
YT = np.array([[1e5, 630.5], [1e6, 397.5], [1e7, 165.0]])
bx, ax = np.polyfit(np.log10(XT[:, 0]), XT[:, 1], 1)
by, ay = np.polyfit(np.log10(YT[:, 0]), YT[:, 1], 1)


def to_data(x, y):
    return 10 ** ((x - ax) / bx), 10 ** ((y - ay) / by)


def fragments(mask):
    lab = -np.ones(mask.shape, dtype=int)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0] >= 0:
            continue
        stack, pts = [(y0, x0)], []
        lab[y0, x0] = len(out)
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] < 0:
                    lab[yy, xx] = len(out)
                    stack.append((yy, xx))
        out.append(np.array(pts))
    return out


rows = []
for run, colour in COLORS.items():
    mask = np.sqrt(((img - np.array(colour)) ** 2).sum(axis=2)) < 45
    x0, y0, x1, y1 = LEGEND
    mask[y0:y1, x0:x1] = False
    frags = [p for p in fragments(mask) if len(p) >= 15]
    side = 19  # interior side of a marker in pixels (modal value for every run)
    for p in frags:
        ys, xs = p[:, 0], p[:, 1]
        xa, xb, ya, yb = xs.min(), xs.max(), ys.min(), ys.max()

        def has(dx, dy):
            if dx:
                cols = range(xb + 2, xb + 6) if dx > 0 else range(xa - 5, xa - 1)
                return any(mask[yy, xx] for xx in cols for yy in range(ya, yb + 1) if 0 <= xx < W)
            rws = range(yb + 2, yb + 6) if dy > 0 else range(ya - 5, ya - 1)
            return any(mask[yy, xx] for yy in rws for xx in range(xa, xb + 1) if 0 <= yy < H)

        w, h = xb - xa + 1, yb - ya + 1
        cx = (xa + xb) / 2 if w >= side - 1 else (xa + (side - 1) / 2 if has(1, 0) and not has(-1, 0)
                                                  else xb - (side - 1) / 2 if has(-1, 0) and not has(1, 0)
                                                  else (xa + xb) / 2)
        cy = (ya + yb) / 2 if h >= side - 1 else (ya + (side - 1) / 2 if has(0, 1) and not has(0, -1)
                                                  else yb - (side - 1) / 2 if has(0, -1) and not has(0, 1)
                                                  else (ya + yb) / 2)
        rows.append((run, cx, cy, "auto"))
    for cx, cy in VISUAL.get(run, []):
        rows.append((run, cx, cy, "visual"))

records = []
for run, cx, cy, src in sorted(rows, key=lambda r: (r[0], -r[2])):
    dT, Nf = to_data(cx, cy)
    c = RUNS[run]
    records.append(dict(run=run, I_load_A=c["I"], t_on_s=c["ton"], Tjmin_C=60, dT_C=round(dT, 2),
                        Nf=int(round(Nf, -2)), source=src, x_px=round(cx, 1), y_px=round(cy, 1)))

out = HERE / "G2_hernes2021_sohoa.csv"
with open(out, "w", newline="\n", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
    wr.writeheader()
    wr.writerows(records)

print(f"axis fit: x = {ax:.2f} + {bx:.2f} log10(dT), y = {ay:.2f} + {by:.2f} log10(Nf)")
print(f"{'run':>3} {'found':>5}/{'EOL':<3} {'dT min-max (Table 5)':>28} {'Nf min-max (Table 5)':>44}"
      f" {'sd log10 Nf':>11} {'Weibull k':>9}")
resid = {}
for run in RUNS:
    rr = [r for r in records if r["run"] == run]
    dts = [r["dT_C"] for r in rr]
    nfs = np.array([r["Nf"] for r in rr], dtype=float)
    t = TABLE5[run]
    lg = np.log10(nfs)
    resid[run] = lg - lg.mean()
    k, _, _ = stats.weibull_min.fit(nfs, floc=0)
    print(f"{run:>3} {len(rr):>5}/{RUNS[run]['n_eol']:<3} {min(dts):6.1f}-{max(dts):5.1f} ({t[0]:.1f}-{t[1]:.1f})"
          f"   {nfs.min():>12,.0f}-{nfs.max():>12,.0f} ({t[2]:,}-{t[3]:,})"
          f" {lg.std(ddof=1):11.3f} {k:9.1f}")
fk = stats.fligner(*resid.values())
print(f"Fligner-Killeen test of equal log-scatter across runs: statistic {fk.statistic:.2f}, p = {fk.pvalue:.3f}")
pooled = np.concatenate(list(resid.values()))
print(f"pooled within-run sd of log10 Nf: {pooled.std(ddof=len(RUNS)):.3f} "
      f"(factor {10 ** (2 * pooled.std(ddof=len(RUNS))):.2f} for +-1 sd)")
