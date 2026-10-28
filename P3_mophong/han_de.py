"""P3 (19/10/2026, editor re-review): baseplate-solder fatigue from the case-temperature cycles (chu_ky_vo.py).

Thermal-cycling law of the solder between DCB substrate and baseplate (Infineon AN2019-05, PC and TC diagrams, Rev. 2.1,
2021, pp. 11-12 and Fig. 8): case-temperature cycles of several minutes fatigue this joint, and at a case swing of 80 K
a traction module with an AlSiC baseplate withstands 30,000 cycles, a module with a copper baseplate 3,000. The note's
worked example (p. 15) gives 75,000 cycles at 65 K and 650,000 at 40 K, exponents of 4.41 and 4.44 from 80 K; the lower
value is used, which lengthens no life at the small swings found here:
    N_f(Delta T_C) = 3.0e4 (80 K / Delta T_C)^4.4   (AlSiC; copper ten times fewer cycles).
Dwell: the case cycles of the missions are heated for minutes to hours, longer than the cycles of the curve, and solder
fatigue depends on the dwell (Engelmaier 1983); with the dwell rule, the life of a cycle heated longer than T_DW = 120 s
is divided by (t_on / T_DW)^0.5, the stated heating-time slope of the junction margin.

Miner damage per day of the outer (T1, D1) and inner (T2, D2) modules, averaged over the 4215 vehicle days, under
variants of the dwell rule, the thermal scale (1.10), the swing under the hottest chip against the lumped case node
(x1.3), the life (halved) and the baseplate (AlSiC or copper); then compared with the junction damage of the module,
that of its most damaged chip, margined or bare (phan_ra_hu_hai.json).

daily() gives the nominal solder damage with the dwell rule per module and day, the base of the solder channel of the
truth in thu_nghiem_g6.py (S runs).

Run: python han_de.py -> han_de.json, han_de.md
"""
import json
import site
import sys
from pathlib import Path

sys.path.insert(0, site.getusersitepackages())
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
N80, DT80, K_EXP = 3.0e4, 80.0, 4.4
T_DW, G_DW = 120.0, 0.5
# name, dwell rule, thermal scale, swing factor under the hottest chip, life divisor, baseplate factor (copper 10)
VARIANTS = (("nominal, AlSiC", False, 1.0, 1.0, 1.0, 1.0),
            ("with dwell, AlSiC", True, 1.0, 1.0, 1.0, 1.0),
            ("stacked with dwell, AlSiC", True, 1.10, 1.3, 2.0, 1.0),
            ("stacked with dwell, Cu", True, 1.10, 1.3, 2.0, 10.0))
MODS = {0: ("outer", ("T1", "D1")), 1: ("inner", ("T2", "D2"))}


def life(dT, ton, dwell):
    """Solder cycles to failure of case cycles of swing dT (K) and heating time ton (s), AlSiC baseplate."""
    N = N80 * (DT80 / np.maximum(dT, 1e-3)) ** K_EXP
    return N / np.maximum(ton / T_DW, 1.0) ** G_DW if dwell else N


def daily(strategy="cruise", dwell=True):
    """Nominal solder damage (AlSiC law, nominal thermal model, dwell rule) of each certified module on each vehicle
    day: [2, days], in the day order of the junction databases (chu_ky_<strategy>_d00_base.npz)."""
    z = np.load(HERE / f"chu_ky_vo_{strategy}.npz")
    cyc, n_days = z["cycles"].astype(float), len(z["day"])
    if not np.array_equal(z["day"], np.arange(n_days)):
        raise ValueError("case-cycle database out of day order")
    d = np.zeros((len(MODS), n_days))
    for j in MODS:
        c = cyc[cyc[:, 1] == j]
        np.add.at(d[j], c[:, 0].astype(int), c[:, 5] / life(c[:, 2], c[:, 4], dwell))
    return d


def main():
    P = json.loads((HERE / "phan_ra_hu_hai.json").read_text(encoding="utf-8"))
    out, md = {}, ["# Baseplate-solder fatigue from case-temperature cycles", ""]
    for st in ("cruise", "coast"):
        z = np.load(HERE / f"chu_ky_vo_{st}.npz")
        cyc, n_days = z["cycles"].astype(float), len(z["day"])
        res = {}
        md += [f"## {st} ({n_days} days)", "",
               "| module | variant | solder damage per day | x margined junction | x bare junction | solder life (years) |",
               "|---|---|---|---|---|---|"]
        for j, (mname, chips) in MODS.items():
            c = cyc[cyc[:, 1] == j]
            dT, ton, n = c[:, 2], c[:, 4], c[:, 5]
            jm = max(P[st]["chips"][ch]["margined_per_day"] for ch in chips)
            jb = max(P[st]["chips"][ch]["bare_per_day"] for ch in chips)
            daymax = np.array([dT[c[:, 0] == k].max() if np.any(c[:, 0] == k) else 0.0 for k in z["day"]])
            top = dT >= np.quantile(dT, 0.99)
            stats = dict(cycles_per_day=float(n.sum() / n_days), swing_daily_max_median=float(np.median(daymax)),
                         swing_daily_max_p99=float(np.quantile(daymax, 0.99)), swing_max=float(dT.max()),
                         ton_median_of_largest=float(np.median(ton[top])), ton_max=float(ton.max()),
                         ton_q10_of_largest=float(np.quantile(ton[top], 0.1)))
            rv = {}
            for name, dwell, s, f_sw, f_life, f_cu in VARIANTS:
                N = life(s * f_sw * dT, ton, dwell) / (f_life * f_cu)
                d = float(np.sum(n / N) / n_days)
                rv[name] = dict(per_day=d, vs_margined=d / jm, vs_bare=d / jb, life_years=1.0 / (365.0 * d))
                md.append(f"| {mname} | {name} | {d:.2e} | {d / jm:.3g} | {d / jb:.3g} | {1 / (365 * d):.3g} |")
            res[mname] = dict(stats=stats, junction_margined=jm, junction_bare=jb, variants=rv)
        for mname in res:
            s = res[mname]["stats"]
            md.append(f"\n{mname}: {s['cycles_per_day']:.0f} case cycles a day; largest swing of a day: median "
                      f"{s['swing_daily_max_median']:.1f} K, 99 % {s['swing_daily_max_p99']:.1f} K, maximum {s['swing_max']:.1f} K;"
                      f" heating time of the top 1 % swings: median {s['ton_median_of_largest']:.0f} s, 10 % "
                      f"{s['ton_q10_of_largest']:.0f} s; longest heating time {s['ton_max']:.0f} s")
        md.append("")
        out[st] = res
    (HERE / "han_de.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    (HERE / "han_de.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
