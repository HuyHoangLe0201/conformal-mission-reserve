"""P3: graphical abstract for Reliability Engineering & System Safety (encouraged, uploaded as a separate file).

Three parts, from the saved results (no simulation):
  left    structure of the paired certificate: calibrated inputs (bench lives, mission records re-simulated under the
          deployed policy), stated inputs (thermal bound, lifetime margin), the certificate and the derating policy,
          with the two feedback paths of the paper (re-simulation, test pulses);
  middle  coverage per leg life under certificate-triggered derating, paired and union, cruising (filled) and coasting
          (open) (data of Fig. 4, g6_tong_hop.json, 300 lives per run, truth at the margins); the red line is the level
          1 - alpha of a single decision, which a whole life need not reach;
  right   limits of protective derating: certified damage against derating depth on the coasting mission without swing
          margin, with the stated swing margin and with a swing-margin slope equal to the lifetime exponent (data of
          Fig. 7b, e3_coast_base_delta.json; on the cruising mission, Fig. 7c, derating raises the damage at every
          slope).
Journal rules: at least 1328 pixels wide, longest side horizontal, readable at a width of 13 cm; TIFF, EPS, PDF or MS
Office. Drawn at 130 mm with 6.5-7.5 pt text, style of kieu_hinh.py.
Outputs: ../P3_hinh_bang/GA.pdf (vector), GA.png (300 dpi) and GA.tif (400 dpi, LZW).

Run: python -I hinh_ga.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import kieu_hinh as kh  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

G6 = json.loads((HERE / "g6_tong_hop.json").read_text(encoding="utf-8"))
E3 = json.loads((HERE / "e3_coast_base_delta.json").read_text(encoding="utf-8"))
plt = kh.plt
W_MM, H_MM = 130, 56
LIGHT = {kh.BLUE: "#E5F0F8", kh.ORANGE: "#FCF1DE", "k": "#F2F2F2", kh.GREY: "#F2F2F2"}


def box(ax, cx, cy, w, h, text, edge, size=6.6, weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                lw=0.7, edgecolor=edge, facecolor=LIGHT.get(edge, "white")))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, fontweight=weight, linespacing=1.15)


def arrow(ax, xy_from, xy_to, color="k", ls="-", rad=0.0):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color, ls=ls, mutation_scale=6,
                                shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def schematic(fig):
    """Left part, drawn in millimetres."""
    x0, w = 0.0, 0.405
    ax = fig.add_axes((x0, 0.0, w, 0.9))
    wmm, hmm = w * W_MM, 0.9 * H_MM
    ax.set_xlim(0, wmm)
    ax.set_ylim(0, hmm)
    ax.axis("off")
    bx, bw = 15.0, 23.5                      # input column; the feedback loop runs along its left side
    yb, ym, ys = 42.5, 30.0, 17.0            # bench, mission records, stated inputs
    box(ax, bx, yb, bw, 7.0, "Bench lives at\ntested conditions", kh.BLUE)
    box(ax, bx, ym, bw, 10.5, "Mission records,\nre-simulated under\nthe deployed policy", kh.BLUE)
    box(ax, bx, ys, bw, 7.0, "Thermal bound,\nlifetime margin", kh.ORANGE)
    cx, cw, cy, py = 42.5, 18.5, 36.0, 17.0
    box(ax, cx, cy, cw, 11.0, "Paired\ncertificate of the\nend-of-interval\ndamage", "k", weight="bold")
    box(ax, cx, py, cw, 7.0, "Derating\npolicy", kh.GREY)
    for y in (yb, ym, ys):
        arrow(ax, (bx + bw / 2, y), (cx - cw / 2, cy + (y - ym) * 0.25), color="0.25")
    arrow(ax, (cx, cy - 5.5), (cx, py + 3.5))
    # feedback of the policy, one dashed loop: re-simulation of the recorded inputs, test pulses for the thermal bound
    xl, yl = 1.3, 8.0
    ax.plot([cx, cx, xl, xl], [py - 3.5, yl, yl, ym], color=kh.GREY, ls="--", lw=0.7)
    arrow(ax, (xl, ym), (bx - bw / 2, ym), color=kh.GREY)
    arrow(ax, (xl, ys), (bx - bw / 2, ys), color=kh.GREY)
    ax.text((xl + cx) / 2, yl - 2.0, "feedback: re-simulation, test pulses", ha="center", va="center", fontsize=6.4,
            color=kh.GREY, style="italic")
    ax.text(xl, 1.6, "blue: conformal ranks    orange: stated confidence", ha="left", va="center", fontsize=6.4,
            color="0.3")
    fig.text(x0 + 0.01, 0.955, "Mission reserve certificate", fontsize=7.5, fontweight="bold", va="center")


def coverage(fig):
    """Middle part: coverage per leg life against the mean derating level (Fig. 4, paired and union; cruising filled,
    coasting open)."""
    f4 = G6["F4"]
    ax = fig.add_axes((0.495, 0.21, 0.205, 0.64))
    for st, fill in (("cruise", "full"), ("coast", "none")):
        for i, m in enumerate(("union", "paired")):
            col, name = kh.METHOD[m]
            pts = sorted(((v["mean_level"] or 0.0, v) for k, v in f4.items()
                          if k.startswith(st + "|") and k.endswith(f"|{m}")), key=lambda q: q[0])
            x = [p + 0.006 * (2 * i - 1) for p, _ in pts]
            y = [v["cov_life"] for _, v in pts]
            err = [[v["cov_life"] - v["ci"][0] for _, v in pts], [v["ci"][1] - v["cov_life"] for _, v in pts]]
            ax.errorbar(x, y, yerr=err, fmt="o", color=col, ms=2.4, lw=0.8, capsize=1.2, elinewidth=0.5,
                        fillstyle=fill, mew=0.6, label=name if st == "cruise" else None)
    ax.axhline(0.9, color=kh.VERMILION, lw=0.6)
    ax.text(0.41, 0.897, r"$1-\alpha$ per decision", fontsize=6.4, color=kh.VERMILION, ha="right", va="top")
    ax.set_ylim(0.88, 1.0)
    ax.set_yticks([0.9, 0.95, 1.0], ["0.90", "0.95", "1.00"])
    ax.set_xlim(-0.03, 0.42)
    ax.set_xticks([0, 0.2, 0.4], ["0", "0.2", "0.4"])
    ax.set_xlabel("mean derating level", fontsize=6.8, labelpad=1.5)
    ax.set_ylabel("leg lives covered", fontsize=6.8, labelpad=1.5)
    ax.tick_params(labelsize=6.6)
    ax.legend(loc="lower left", fontsize=6.4, ncol=2, handlelength=1.4, columnspacing=0.8,
              bbox_to_anchor=(0.0, 0.2))
    fig.text(0.445, 0.955, "Coverage under feedback", fontsize=7.5, fontweight="bold", va="center")


def collapse(fig):
    """Right part: certified damage against derating depth (Fig. 7b) for swing-margin slopes 0, 1.2 (stated), beta."""
    lv = [float(x) for x in E3["levels"]]
    ax = fig.add_axes((0.80, 0.21, 0.195, 0.64))
    lines = (("0.0", kh.BLUE, ":", "no swing margin"), ("1.2", kh.GREEN, "-", "stated swing margin"),
             ("2.08", kh.ORANGE, "--", r"swing slope $\gamma=\beta$"))
    for g, col, ls, name in lines:
        base = E3["b"]["0.0"][g]
        ax.plot([100 * d for d in lv], [E3["b"][str(d)][g] / base for d in lv], ls, color=col, marker="o", ms=2.2,
                lw=0.9, label=name)
    ax.axhline(1, color=kh.GREY, lw=0.5)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2], ["0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_xticks([0, 25, 50])
    ax.set_xlabel("derating depth (%)", fontsize=6.8, labelpad=1.5)
    ax.set_ylabel("certified damage, relative", fontsize=6.8, labelpad=1.5)
    ax.tick_params(labelsize=6.6)
    ax.set_ylim(0.25, 2.3)
    ax.legend(loc="lower left", fontsize=6.4, handlelength=1.4, handletextpad=0.4, borderaxespad=0.1)
    fig.text(0.745, 0.955, "Limits of protective derating", fontsize=7.5, fontweight="bold", va="center")
    ax.text(0.97, 0.97, "coasting", transform=ax.transAxes, fontsize=6.4, ha="right", va="top", color="0.3")
    fig.text(0.995, 0.03, "remedies: anchor tests, dwell-window pulses", fontsize=6.4, color="0.3", style="italic",
             ha="right", va="center")


def main():
    fig = plt.figure(figsize=(W_MM * kh.MM, H_MM * kh.MM))
    schematic(fig)
    coverage(fig)
    collapse(fig)
    kh.OUT.mkdir(exist_ok=True)
    # fixed canvas (no tight crop, which kieu_hinh sets by default): 130 mm is the width at which the journal displays it
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(kh.OUT / "GA.pdf")
        fig.savefig(kh.OUT / "GA.png", dpi=300)
        fig.savefig(kh.OUT / "GA.tif", dpi=400, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("written", kh.OUT / "GA.pdf", "GA.png", "GA.tif")


if __name__ == "__main__":
    main()
