"""P3, third review (19/10/2026): where does the damage come from? Shares of the bare-model and margined damage of the
critical chip by swing and by heating time, for both driving strategies without derating, over all 4215 vehicle days.

Bare: the lifetime model fitted on all 34 Hernes modules, nominal thermal model, no stated margin (g5_ngan_sach 'bare').
Margined: upper thermal scale 1.10 and the stated margin of the case study without its statistical part (g5_ngan_sach
'base'; the statistical part scales whole intervals, not single cycles). Short pulses (t_on < 0.15 s) are out of scope in
both. Bins use the nominal swing and heating time of each cycle. The critical chip is the one with the largest margined
damage over all days.

Run: python phan_ra_hu_hai.py -> phan_ra_hu_hai.json
"""
import json
import site
import sys
from pathlib import Path

sys.path.insert(0, site.getusersitepackages())
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import thu_nghiem_g5 as g5  # noqa: E402
import tuoi_tho as tt  # noqa: E402

HERE = Path(__file__).resolve().parent
BARE = tt.Margin(stat="none", gamma=0.0, gamma_t=0.0, ton_rule="cap", Ea=0.0, kappa_M=1.0, short_rule="exclude")
BASE = tt.Margin(stat="none", short_rule="exclude")
DT_BINS = (0.0, 10.0, 26.0, 40.0, 70.0, np.inf)
TON_BINS = (0.15, 4.5, 60.0, 300.0, np.inf)


def label(edges, unit):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.append(f"{a:g}-{b:g} {unit}" if np.isfinite(b) else f">= {a:g} {unit}")
    return out


def per_cycle(cyc, mg, s, fitd):
    keep = (cyc["n"] > 0) & (cyc["ton"] >= tt.TON_LO)
    V, loga = tt.prep(cyc, mg, s)
    d = np.exp(loga - V @ fitd["b"])
    return d, cyc["dTheta"][keep], cyc["ton"][keep], cyc["theta_min"][keep] + cyc["T_amb"][keep]


def main():
    h = tt.hernes()
    fitd = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])
    out = {}
    for st in ("cruise", "coast"):
        pool = g5.DayPool(f"chu_ky_{st}_d00_base.npz", tt.Margin(stat="delta"))
        res = {}
        for j, name in enumerate(g5.CHIPS):
            fl = pool.flat[j]
            cyc = {k: fl[k] for k in ("dTheta", "theta_min", "ton", "n", "T_amb")}
            dm, dT, ton, tmin = per_cycle(cyc, BASE, g5.S_UP, fitd)
            db, _, _, _ = per_cycle(cyc, BARE, 1.0, fitd)
            res[name] = dict(
                margined_per_day=float(dm.sum() / pool.n), bare_per_day=float(db.sum() / pool.n),
                swing={lab: [float(db[(dT >= a) & (dT < b)].sum() / db.sum()), float(dm[(dT >= a) & (dT < b)].sum() / dm.sum())]
                       for lab, a, b in zip(label(DT_BINS, "K"), DT_BINS[:-1], DT_BINS[1:])},
                heating={lab: [float(db[(ton >= a) & (ton < b)].sum() / db.sum()),
                               float(dm[(ton >= a) & (ton < b)].sum() / dm.sum())]
                         for lab, a, b in zip(label(TON_BINS, "s"), TON_BINS[:-1], TON_BINS[1:])},
                cycles_per_day=float(np.sum(cyc["n"][(cyc["n"] > 0) & (cyc["ton"] >= tt.TON_LO)]) / pool.n),
                share_largest_cycle_of_day=None)
            # share of the margined damage carried by the largest-damage cycle of each day (the daily cold-start or
            # longest full-current cycle)
            off = fl["off"]
            keep_all = (fl["n"] > 0) & (fl["ton"] >= tt.TON_LO)
            dm_full = np.zeros(fl["n"].size)
            dm_full[keep_all] = dm
            top = sum(float(dm_full[off[k]:off[k + 1]].max()) for k in range(pool.n) if off[k + 1] > off[k])
            res[name]["share_largest_cycle_of_day"] = top / float(dm.sum())
        crit = max(res, key=lambda c: res[c]["margined_per_day"])
        out[st] = dict(critical_chip=crit, chips=res)
        print(st, "critical", crit, {c: round(res[c]["margined_per_day"], 6) for c in res})
        del pool
    (HERE / "phan_ra_hu_hai.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
