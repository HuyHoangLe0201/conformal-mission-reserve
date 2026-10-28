"""Common style of the final figures of P3 (Elsevier column widths, Times-like fonts, colour-blind safe palette).

Use: import kieu_hinh as kh; fig, ax = kh.figure(kh.W2, 60, 1, 2); ...; kh.save(fig, "F2")
Outputs go to ../P3_hinh_bang/ as vector PDF and a 300 dpi PNG preview.
"""
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "P3_hinh_bang"
MM = 1 / 25.4
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM          # single, one-and-a-half and double column (inches)

# Okabe-Ito palette
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILION, PURPLE = ("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
                                                              "#0072B2", "#D55E00", "#CC79A7")
GREY = "#6E6E6E"
# one colour and name per certificate, in every figure. The keys are the labels U1-U10 of the code; the manuscript names
# the certificates without numbers, since U_i denotes the conformal scores there.
CERT = {"U1 box": (GREY, "box"), "U2 Monte Carlo": (SKY, "Monte Carlo"), "U3 Bayes": (YELLOW, "Bayes"),
        "U5 union": (ORANGE, "union"), "U6 paired": (BLUE, "paired"), "U9 PAC": (GREEN, "PAC"),
        "U6 anchored": (BLUE, "paired, ideal anchors"), "U10 oracle": (BLACK, "oracle")}
METHOD = {"union": (ORANGE, "union"), "paired": (BLUE, "paired"), "split0": (PURPLE, "split"),
          "reuse": (VERMILION, "paired, reused data"), "aci": (GREEN, "adaptive"), "mc": (SKY, "Monte Carlo"),
          "bayes": (YELLOW, "Bayes")}
# closed-loop runs of the cost comparison (policy|method of thu_nghiem_g6.py): colour, name, marker, line style
RUN = {"reserve|paired": (BLUE, "paired, reserve rule", "o", "-"),
       "reserve|mc": (SKY, "Monte Carlo, reserve rule", "D", "-"),
       "reserve_budget|paired": (GREEN, "paired, late budget", "P", "-"),
       "replace|paired": (PURPLE, "paired, replacement only", "^", "-"),
       "none|tbr60": (BLACK, "planned 60 intervals", "s", "--"),
       "none|fix20_tbr60": (GREY, "20 % derating, planned 60", "s", ":"),
       "none|cert_age": (ORANGE, "certificate age at commissioning", "v", "--"),
       "none|mc_age": (YELLOW, "Monte Carlo B10 age", "d", "--"),
       "none|rtf": (VERMILION, "run to failure", "x", "-.")}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5, "lines.linewidth": 1.0, "lines.markersize": 3.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "figure.dpi": 150,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E6E6E6", "grid.linewidth": 0.4, "axes.axisbelow": True})


def figure(width, height_mm, rows=1, cols=1, **kw):
    fig, ax = plt.subplots(rows, cols, figsize=(width, height_mm * MM), **kw)
    return fig, ax


def refline(ax, y, text, color=VERMILION, ha="right", va="bottom"):
    """Horizontal reference line with its name written just above it at the right (or left) end."""
    ax.axhline(y, color=color, lw=0.6)
    ax.text(0.99 if ha == "right" else 0.01, y, text, transform=ax.get_yaxis_transform(), ha=ha, va=va,
            fontsize=6.5, color=color)


def label(ax, letter, x=-0.16, y=1.02):
    """Panel letter, bold, above the upper-left corner."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("written", OUT / f"{name}.pdf")
