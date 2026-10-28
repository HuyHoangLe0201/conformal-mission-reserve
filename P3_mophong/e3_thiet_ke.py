"""P3 fourth round (20/10/2026): overlap collapse caused by protection, on converters whose junction swings are larger.

On the case-study converter the damage-weighted swing of an interval lies below or at the edge of the bench support
without any derating (e3: about 23 K cruising, 39 K coasting), so the weighted certificate is infinite before the
controller protects anything. Here the junction rise of the same missions is multiplied by a design factor (a smaller
cooler, or a heavier train with larger currents: swings 1.5 or 2 times those of the case), so that at full load the
loading lies inside the bench support; derating then moves it out. For each factor, driving strategy and derating
level: the damage-weighted mean log swing X of 200 intervals of the critical chip (stated margin, gamma = 1.2), its
median in kelvin, and the test weight p_{n+1} of the weighted certificate at the median X with 0, 5, 10 and 20 anchor
tests at that loading (alpha_S = 0.05), exactly as e3 of thu_nghiem_g5.py.

Run: python -I e3_thiet_ke.py  -> e3_thiet_ke.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import thu_nghiem_g6 as g6  # noqa: E402  (user site; WMI guard)
import numpy as np  # noqa: E402

import chung_chi as cc  # noqa: E402
import thu_nghiem_g5 as g5  # noqa: E402
import tuoi_tho as tt  # noqa: E402

FACTORS, ANCHORS, ALPHA_S, N_INT, SEED = (1.0, 1.5, 2.0), (0, 5, 10, 20), 0.05, 200, 20261020

g6.wait_memory(2.0, "e3_thiet_ke")
h = tt.hernes()
fitd = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])
mg = tt.Margin(stat="delta", gamma=1.2)
x_bench = np.log(h["dT"])
out = {"factors": FACTORS, "anchors": ANCHORS, "alpha_S": ALPHA_S, "bench_swing_K": [float(h["dT"].min()),
                                                                                       float(h["dT"].max())]}
for strategy in ("cruise", "coast"):
    for fac in FACTORS:
        rng = np.random.default_rng(SEED)
        days, res = None, {}
        for d in g5.LEVELS:
            pool = g5.DayPool(f"chu_ky_{strategy}_d{int(round(100 * d)):02d}_base.npz", mg, s_up=g5.S_UP * fac)
            if days is None:
                days = [pool.sample(rng) for _ in range(N_INT)]
            X = []
            for q in days:
                k = int(np.argmax(pool.model(q, fitd, mg)))
                Vc = np.vstack([pool.cells[k][i][0] for i in q])
                la = np.concatenate([pool.cells[k][i][1] for i in q])
                Vcc, lac = tt.compress(Vc, la)
                _, parts = tt.damage_from_prep(Vcc, lac, fitd, tt.Margin(stat="none", gamma=1.2), return_parts=True)
                X.append(float(np.sum(parts * Vcc[:, 1]) / parts.sum()))
            X = np.array(X)
            r = {"X_median_K": float(np.exp(np.median(X)))}
            for n_a in ANCHORS:
                xb = np.r_[x_bench, np.full(n_a, np.median(X))]
                w = cc.kde_ratio(xb, X)
                p = float(w(np.median(X))[0] / (w(xb).sum() + w(np.median(X))[0]))
                r[f"p_test_{n_a}"] = p
            res[f"{d:.1f}"] = r
            print(f"{strategy} x{fac}: derating {d:.1f}: X {r['X_median_K']:.1f} K, "
                  + ", ".join(f"p(n_a={n}) {r[f'p_test_{n}']:.3f}" for n in ANCHORS), flush=True)
            del pool
        out[f"{strategy}_x{fac}"] = res
(HERE / "e3_thiet_ke.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
print("written e3_thiet_ke.json")
