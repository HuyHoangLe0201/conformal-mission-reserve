"""G3: cycles to failure against load-pulse duration from Scheuermann, Junghaenel (2018), CIPS 2018, pp. 78-87,
Fig. 6 (dTj = 70 K, t_on 0.07-2 s) and Fig. 7 (dTj = 110 K, t_on 2-60 s), after Junghaenel, Scheuermann (2017).

Two module groups on a modified SKiM63 (no baseplate): group 1 has sintered chips, so it fails by Al wire-bond
lift-off only; group 2 has soldered chips with Al-clad Cu wires, so it fails by chip-solder fatigue only.

Both figures are vector graphics, so marker centres are read from the PDF drawing commands, not from pixels:
red squares (group 1) and dark-blue diamonds (group 2), markers inside the legend box dropped. Axes are calibrated on
the tick-label centres. In Fig. 6 this agrees with the plot-area rectangle within 0.1 pt, about 0.4 % in N_f.

Run: python sohoa_scheuermann2018.py [path to the PDF]   (PyMuPDF). Writes G3_scheuermann2018_ton.csv.
The PDF is the user's licensed copy in ../Tailieu_P3/ (VDE); it is read, not copied.
"""
import csv
import math
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())  # keeps the script runnable with python -I
import fitz  # noqa: E402

HERE = Path(__file__).parent
if len(sys.argv) > 1:
    PDF = Path(sys.argv[1])
else:
    PDF = next((HERE.parent / "Tailieu_P3").glob("ng*15-10/Scheuermann2018_*.pdf"))

RED, BLUE = (0.93, 0.2, 0.14), (0.23, 0.24, 0.59)
# figure: page index, clip box (pt), dTj, tick labels used for the axes (text: value)
FIGS = {6: dict(page=5, clip=(50, 280, 300, 500), dTj=70, xticks={"0,01": 0.01, "0,1": 0.1, "1": 1, "10": 10},
                yticks={"1E+05": 1e5, "1E+06": 1e6, "1E+07": 1e7}),
        7: dict(page=5, clip=(50, 500, 300, 720), dTj=110, xticks={"1": 1, "10": 10, "100": 100},
                yticks={"1E+04": 1e4, "1E+05": 1e5})}


def close(a, b, tol=0.02):
    return a is not None and all(abs(x - y) < tol for x, y in zip(a, b))


doc = fitz.open(str(PDF))
rows = []
for fig, f in FIGS.items():
    page = doc[f["page"]]
    clip = fitz.Rect(*f["clip"])
    xs, ys = [], []
    for b in page.get_text("dict", clip=clip)["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                t, (x0, y0, x1, y1) = s["text"].strip(), s["bbox"]
                if t in f["xticks"] and y0 > clip.y1 - 80:  # x tick labels sit under the plot
                    xs.append(((x0 + x1) / 2, math.log10(f["xticks"][t])))
                if t in f["yticks"] and x1 < clip.x0 + 60:  # y tick labels sit left of the plot
                    ys.append(((y0 + y1) / 2, math.log10(f["yticks"][t])))
    # least squares through the tick positions: position = a + b * log10(value)
    def fit(pts):
        n = len(pts)
        mx = sum(v for _, v in pts) / n
        mp = sum(p for p, _ in pts) / n
        b = sum((v - mx) * (p - mp) for p, v in pts) / sum((v - mx) ** 2 for _, v in pts)
        return mp - b * mx, b, max(abs(mp - b * mx + b * v - p) for p, v in pts)
    ax, bx, rx = fit(xs)
    ay, by, ry = fit(ys)
    legends = [fitz.Rect(d["rect"]) for d in page.get_drawings()
               if fitz.Rect(d["rect"]).intersects(clip) and close(d.get("color"), (0.46, 0.47, 0.47))]
    n_pts = {1: 0, 2: 0}
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"])
        if not r.intersects(clip) or r.width > 6 or r.height > 6:
            continue
        group = 1 if close(d.get("fill"), RED) else 2 if close(d.get("fill"), BLUE) else None
        if group is None:
            continue
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if any(L.contains(fitz.Point(cx, cy)) for L in legends):
            continue
        t_on = 10 ** ((cx - ax) / bx)
        nf = 10 ** ((cy - ay) / by)
        n_pts[group] += 1
        rows.append(dict(figure=fig, dTj_K=f["dTj"], group=group,
                         mechanism="wire-bond lift-off" if group == 1 else "chip-solder fatigue",
                         t_on_s=round(t_on, 3), Nf=int(round(nf, -2)), x_pt=round(cx, 2), y_pt=round(cy, 2)))
    print(f"Fig. {fig} (dTj {f['dTj']} K): axis residual {rx:.2f} pt (x), {ry:.2f} pt (y); "
          f"{bx:.2f} pt per decade of t_on, {abs(by):.2f} pt per decade of N_f; markers: wire bond {n_pts[1]}, "
          f"solder {n_pts[2]}")

rows.sort(key=lambda r: (r["figure"], r["group"], r["t_on_s"], r["Nf"]))
out = HERE / "G3_scheuermann2018_ton.csv"
with open(out, "w", newline="\n", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
    wr.writeheader()
    wr.writerows(rows)
for r in rows:
    print(f"  Fig. {r['figure']} {r['mechanism']:20s} t_on {r['t_on_s']:7.3f} s  N_f {r['Nf']:>10,}")
print("written", out.name, len(rows), "rows")
