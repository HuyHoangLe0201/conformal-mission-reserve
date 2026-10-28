"""P3 fourth round (20/10/2026): collapse caused by protection, with anchor tests placed at the full-load field loading.

e3 (thu_nghiem_g5.py) and e3_thiet_ke.py place the anchor tests at the loading summary of each derating level. A bench
is built once, before the controller acts, so its anchors sit at the field loading the operator expects, that of full
load. Here n_a anchor tests are placed at the median damage-weighted log swing X of the non-derated intervals, and the
test weight p_{n+1} of the weighted certificate (alpha_S = 0.05, Gaussian kernel ratio as in e3) is evaluated at the
median X of every derating level: if the certificate is finite at full load but infinite once the controller derates,
the protective action itself has defeated it. Both missions, the case-study converter (design factor 1.0) and one with
1.5 times its junction swings.

Run: python -I e3_neo_co_dinh.py  -> e3_neo_co_dinh.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import thu_nghiem_g6 as g6  # noqa: E402
import numpy as np  # noqa: E402

import chung_chi as cc  # noqa: E402
import thu_nghiem_g5 as g5  # noqa: E402
import tuoi_tho as tt  # noqa: E402

FACTORS, ANCHORS, N_INT, SEED = (1.0, 1.5), (20, 40, 80), 200, 20261020
g6.wait_memory(2.0, "e3_neo_co_dinh")
h = tt.hernes()
fitd = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])
mg = tt.Margin(stat="delta", gamma=1.2)
x_bench = np.log(h["dT"])
out = {"anchors": ANCHORS, "alpha_S": 0.05}
for strategy in ("cruise", "coast"):
    for fac in FACTORS:
        rng = np.random.default_rng(SEED)
        days, Xs = None, {}
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
            Xs[d] = np.array(X)
            del pool
        x_a = float(np.median(Xs[0.0]))
        res = {}
        for d, X in Xs.items():
            r = {"X_median_K": float(np.exp(np.median(X)))}
            for n_a in ANCHORS:
                xb = np.r_[x_bench, np.full(n_a, x_a)]
                w = cc.kde_ratio(xb, X)
                xt = np.median(X)
                r[f"p_test_{n_a}"] = float(w(xt)[0] / (w(xb).sum() + w(xt)[0]))
            res[f"{d:.1f}"] = r
            print(f"{strategy} x{fac}: derating {d:.1f}: X {r['X_median_K']:.1f} K (anchors at {np.exp(x_a):.1f} K), "
                  + ", ".join(f"p(n_a={n}) {r[f'p_test_{n}']:.3f}" for n in ANCHORS), flush=True)
        out[f"{strategy}_x{fac}"] = dict(anchor_K=float(np.exp(x_a)), levels=res)
(HERE / "e3_neo_co_dinh.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
print("written e3_neo_co_dinh.json")
