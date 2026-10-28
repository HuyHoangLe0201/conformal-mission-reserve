"""G2: per-module accumulated Miner damage at end of life from Hernes, D'Arco, Spro, Peftitsis (2021), PCIM Europe
digital days 2021, pp. 1202-1207, Fig. 4 ("Estimated accumulated damage versus recorded number of cycles for all DUTs").

Five constant-stress runs (tests 1-5: filled squares, one colour per run) and two mixed-stress runs on the same module
type: test 6 (40 K first, then 70 K) and test 7 (70 K first, then 40 K), whose end-of-life markers are gold diamonds
(test 6) and gold triangles (test 7). The authors compute the damage with their refitted lifetime model, Eq. (2); a value
of 1 means the module failed where the model predicts. For the constant runs the damage at failure is the strength
W = N_f / N_mod(x) of P3; for the mixed runs it is the Miner sum at failure.
The figure was extracted from the published PDF into G2_nguon/hernes2021pcim_fig4_dmg.png (VDE Verlag; licensed copy,
internal use only, not for redistribution).

Method: axis calibration from gridlines (damage 0..1.4 linear; N_f 0.01..100 million cycles, log); colour masks by
hue. Squares are found as dense 5 x 5 windows, which the 2-px ellipse outlines never fill; four blobs made of two
touching squares were split by reading the two squares off the mask (source = "visual"). Gold markers are the
connected components of the gold fill inside their black outline: a diamond sits at the centre of its fill, a
triangle 2.5 px above the last fill row (its base outline is one row below the fill and the marker is 8 px tall).
Gold markers are split into tests 6 and 7 by N_f (Table 2: test 6 below 0.9 million cycles, test 7 above 1.2 million).
One test-1 square is not visible (7/8); with the IET Power Electronics figure, the run-1 module with the same N_f as
another is the likely hidden one.
Checks: minimum and maximum of damage and N_f per run against Table 2; mean damage of tests 6 and 7 against the text
(0.85 and 0.97); number of modules outside 1.0 +- 20 % against the text (three in tests 1-5, one in test 6).

Run: python sohoa_hernes2021pcim.py   (numpy, PyMuPDF). Writes G2_hernes2021pcim_sohoa.csv.
"""
import csv
import math
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())  # keeps the script runnable with python -I
import fitz  # noqa: E402
import numpy as np  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

HERE = Path(__file__).parent
IMG = HERE / "G2_nguon" / "hernes2021pcim_fig4_dmg.png"

# runs: dT (K) of each phase, number of modules at end of life (Table 2), marker class
RUNS = {1: ("70", 8, "blue"), 2: ("40", 8, "red"), 3: ("30", 6, "orange"), 4: ("70", 8, "purple"), 5: ("40", 6, "green"),
        6: ("40 then 70", 7, "gold"), 7: ("70 then 40", 6, "gold")}
# Table 2: N_f min, N_f max, damage min, damage max at the recorded end of life
TABLE2 = {1: (351900, 528051, 0.82, 1.14), 2: (5562900, 7723300, 0.83, 1.24), 3: (18602000, 28932000, 0.82, 1.36),
          4: (101290, 153490, 0.91, 1.22), 5: (2341900, 3218636, 0.88, 1.11), 6: (236371, 875808, 0.75, 0.93),
          7: (1208124, 1559204, 0.80, 1.10)}
TEXT_MEAN = {6: 0.85, 7: 0.97}  # Section 7 of the paper; tests 1-5: 0.96-1.06
# hue windows (degrees), with saturation and value floors that drop the black outlines and the grey grid
HUES = {"red": (345, 10), "orange": (14, 33), "gold": (36, 55), "green": (135, 170), "blue": (190, 215),
        "purple": (250, 290)}
# blobs made of two touching squares: automatic centres inside the box are replaced by the two squares read off the
# mask (box x0, y0, x1, y1 and centres x, y in pixels)
SPLIT = {"blue": [((294, 80, 305, 92), [(298.0, 84.5), (300.5, 87.5)]),
                  ((278, 148, 288, 163), [(283.5, 153.0), (282.5, 158.5)])],
         "purple": [((215, 101, 225, 114), [(219.0, 105.0), (221.0, 110.0)])],
         "green": [((394, 137, 404, 151), [(398.5, 140.5), (398.5, 146.5)])]}

pix = fitz.Pixmap(str(IMG))
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(float) / 255
H, W, _ = img.shape


def gridlines():
    """Light-grey rows and columns that span most of the plot; each line is two pixels wide."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    lev = (r + g + b) / 3
    grey = (np.abs(r - g) < 0.05) & (np.abs(g - b) < 0.05) & (lev > 0.59) & (lev < 0.94)

    def centres(counts, frac, size):
        idx = [i for i in range(size) if counts[i] > frac * (W if size == H else H)]
        groups = []
        for i in idx:
            if groups and i - groups[-1][-1] <= 1:
                groups[-1].append(i)
            else:
                groups.append([i])
        return [float(np.mean(gr)) for gr in groups]

    return centres(grey.sum(axis=1), 0.5, H), centres(grey.sum(axis=0), 0.4, W)


rows, cols = gridlines()
assert len(rows) == 15 and len(cols) == 5, (rows, cols)
by, ay = np.polyfit(np.arange(14, -1, -1) / 10, rows, 1)  # y = ay + by * damage
bx, ax = np.polyfit(np.arange(-2, 3), cols, 1)  # x = ax + bx * log10(N_f / 1e6)
res_y = max(abs(ay + by * d - y) for d, y in zip(np.arange(14, -1, -1) / 10, rows))
res_x = max(abs(ax + bx * e - x) for e, x in zip(range(-2, 3), cols))


def hsv():
    mx, mn = img.max(axis=2), img.min(axis=2)
    d = mx - mn
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    h = np.zeros_like(mx)
    nz = d > 1e-9
    m = nz & (mx == r)
    h[m] = (60 * ((g[m] - b[m]) / d[m])) % 360
    m = nz & (mx == g) & (mx != r)
    h[m] = 60 * ((b[m] - r[m]) / d[m] + 2)
    m = nz & (mx == b) & (mx != r) & (mx != g)
    h[m] = 60 * ((r[m] - g[m]) / d[m] + 4)
    s = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0)
    return h, s, mx


HUE, SAT, VAL = hsv()


def mask(cls):
    lo, hi = HUES[cls]
    inhue = (HUE >= lo) | (HUE <= hi) if lo > hi else (HUE >= lo) & (HUE <= hi)
    m = inhue & (SAT > 0.35) & (VAL > 0.35)
    # keep the plot area only (between the outer gridlines)
    m[: int(rows[0]) - 2, :] = False
    m[int(rows[-1]) + 2:, :] = False
    m[:, : int(cols[0]) - 2] = False
    m[:, int(cols[-1]) + 2:] = False
    return m


def markers(m, k, thr):
    """Centres of dense k x k windows, clustered within 4 px and refined by the centroid of class pixels."""
    win = sliding_window_view(m.astype(np.int16), (k, k)).sum(axis=(2, 3))
    clusters = []
    for r_, c_ in np.argwhere(win >= thr):
        cy, cx, s = r_ + k // 2, c_ + k // 2, int(win[r_, c_])
        for cl in clusters:
            if abs(cl[0] - cy) <= 4 and abs(cl[1] - cx) <= 4:
                cl[2].append((cy, cx, s))
                break
        else:
            clusters.append([cy, cx, [(cy, cx, s)]])
    out = []
    for _, _, members in clusters:
        by_, bx_, s = max(members, key=lambda t: t[2])
        ys, xs = np.nonzero(m[by_ - 4:by_ + 5, bx_ - 4:bx_ + 5])
        cy, cx = by_ - 4 + ys.mean(), bx_ - 4 + xs.mean()
        if all(abs(cy - oy) > 4 or abs(cx - ox) > 4 for oy, ox, _ in out):
            out.append((cy, cx, s))
    return [(cy, cx) for cy, cx, _ in out]


def components(m, min_size=4):
    """4-connected components of a mask, as arrays of (y, x)."""
    lab = -np.ones(m.shape, dtype=int)
    out = []
    for y0, x0 in zip(*np.nonzero(m)):
        if lab[y0, x0] >= 0:
            continue
        stack, pts = [(y0, x0)], []
        lab[y0, x0] = len(out)
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and m[yy, xx] and lab[yy, xx] < 0:
                    lab[yy, xx] = len(out)
                    stack.append((yy, xx))
        out.append(np.array(pts))
    return [p for p in out if len(p) >= min_size]


def to_data(cx, cy):
    return 1e6 * 10 ** ((cx - ax) / bx), (cy - ay) / by


found = {}
for cls in sorted(set(c for _, _, c in RUNS.values())):
    if cls == "gold":
        found[cls] = []
        for p in components(mask(cls)):
            ys, xs = p[:, 0], p[:, 1]
            cx = (xs.min() + xs.max()) / 2
            nf, _ = to_data(cx, ys.mean())
            cy = ys.max() - 2.5 if nf > 1.0e6 else (ys.min() + ys.max()) / 2  # triangle (test 7) or diamond (test 6)
            found[cls].append((cx, cy, "auto"))
        continue
    pts = [(cx, cy, "auto") for cy, cx in markers(mask(cls), 5, 20)]
    for (x0, y0, x1, y1), centres in SPLIT.get(cls, []):
        pts = [p for p in pts if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1)] + [(x, y, "visual") for x, y in centres]
    found[cls] = pts

records = []
for run, (dt, n_eol, cls) in RUNS.items():
    for cx, cy, src in found[cls]:
        nf, dmg = to_data(cx, cy)
        if cls == "gold" and ((run == 6 and nf > 1.0e6) or (run == 7 and nf < 1.0e6)):
            continue
        records.append(dict(test=run, dT_K=dt, Nf=int(round(nf, -3)), damage=round(dmg, 3), source=src,
                            x_px=round(cx, 1), y_px=round(cy, 1)))
records.sort(key=lambda r: (r["test"], r["Nf"]))

out = HERE / "G2_hernes2021pcim_sohoa.csv"
with open(out, "w", newline="\n", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
    wr.writeheader()
    wr.writerows(records)

print(f"gridlines: max residual {res_y:.2f} px (damage), {res_x:.2f} px (log N_f); "
      f"1 px = {1 / abs(by):.4f} damage, {100 * (10 ** (1 / bx) - 1):.2f} % N_f")
print(f"{'test':>4} {'found':>5}/{'EoL':<3} {'damage min-max (Table 2)':>27} {'N_f min-max, million (Table 2)':>40}"
      f" {'mean':>6}")
outside = {}
for run in RUNS:
    rr = [r for r in records if r["test"] == run]
    d = np.array([r["damage"] for r in rr])
    nf = np.array([r["Nf"] for r in rr]) / 1e6
    t = TABLE2[run]
    outside[run] = int(np.sum((d < 0.8) | (d > 1.2)))
    note = f"  (text {TEXT_MEAN[run]:.2f})" if run in TEXT_MEAN else ""
    print(f"{run:>4} {len(rr):>5}/{RUNS[run][1]:<3} {d.min():5.2f}-{d.max():4.2f} ({t[2]:.2f}-{t[3]:.2f})"
          f"   {nf.min():9.3f}-{nf.max():7.3f} ({t[0] / 1e6:.3f}-{t[1] / 1e6:.3f}) {d.mean():6.3f}{note}")
print("modules outside 1.0 +- 20 %: tests 1-5:", sum(outside[r] for r in range(1, 6)), "(text 3); test 6:",
      outside[6], "(text 1); test 7:", outside[7], "(text 0)")
print("written", out.name, len(records), "rows")
