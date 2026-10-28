"""P3 fourth round (20/10/2026, audit): sensitivity of the fixed-anchor result (e3_neo_co_dinh.py) to the kernel
estimate of the density ratio and to where the anchor tests are placed.

For both missions of the case-study converter: the damage-weighted log swings X of 200 intervals at every derating
level (as e3), and the test weight p_{n+1} of the weighted certificate at the median X of each level for
  - bandwidths of both kernels (bench and target) multiplied by 0.5, 1, 2 and 4 (Silverman's rule times the factor);
  - three anchor designs of the same size, 20 tests: all at the full-load loading ('full'), spread over the loadings of
    the six derating levels (three or four at each, 'spread'), and, for reference, 20 at the loading of each level
    (120 tests, 'each', the design of Fig. 8a).

Run: python -I e3_neo_bang_thong.py  -> e3_neo_bang_thong.json
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

MULT, N_INT, SEED = (0.5, 1.0, 2.0, 4.0), 200, 20261020
g6.wait_memory(2.0, "e3_neo_bang_thong")
h = tt.hernes()
fitd = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])
mg = tt.Margin(stat="delta", gamma=1.2)
x_bench = np.log(h["dT"])
sil = lambda x: 1.06 * max(np.std(x), 1e-3) * x.size ** (-0.2)
out = {"mult": MULT}
for strategy in ("cruise", "coast"):
    rng = np.random.default_rng(SEED)
    days, Xs = None, {}
    for d in g5.LEVELS:
        pool = g5.DayPool(f"chu_ky_{strategy}_d{int(round(100 * d)):02d}_base.npz", mg)
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
    med = {d: float(np.median(X)) for d, X in Xs.items()}
    levels = sorted(med)
    spread = [med[levels[i % len(levels)]] for i in range(20)]
    designs = {"full": [med[0.0]] * 20, "spread": spread}
    res = {}
    for m in MULT:
        for name in ("full", "spread", "each"):
            row = {}
            for d, X in Xs.items():
                anchors = designs[name] if name != "each" else [med[d]] * 20
                xb = np.r_[x_bench, anchors]
                w = cc.kde_ratio(xb, X, bw_cal=m * sil(xb), bw_tgt=m * sil(X))
                row[f"{d:.1f}"] = float(w(med[d])[0] / (w(xb).sum() + w(med[d])[0]))
            res[f"{name}|{m}"] = row
            print(f"{strategy} {name:6s} bw x{m}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items()), flush=True)
    out[strategy] = dict(X_median_K={f"{d:.1f}": float(np.exp(v)) for d, v in med.items()}, p_test=res)
(HERE / "e3_neo_bang_thong.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
print("written e3_neo_bang_thong.json")
