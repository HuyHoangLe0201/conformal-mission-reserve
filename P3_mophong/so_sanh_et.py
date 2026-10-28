"""P3 fourth round (20/10/2026): true damage of the electro-thermal response (dien_nhiet.py) against the case-study
response (losses at 150 C, linear network), interval by interval, and against the headroom of the stated thermal bound.

For each driving strategy, 500 random 30-day intervals and the truth at the stated values of every lifetime component
(boundary truth, median strength, Miner factor one): the ratio of the true damage per module (largest chip) computed on
the electro-thermal database to that on the case-study database, both at the nominal thermal scale 1.0; and the ratio
of the case-study damage at the stated upper scale 1.10 to that at 1.0, the headroom that Assumption A3 provides. A3
holds for the electro-thermal truth when the first ratio stays below the second.

Run: python -I so_sanh_et.py  -> so_sanh_et.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import thu_nghiem_g6 as g6  # noqa: E402
import numpy as np  # noqa: E402

import thu_nghiem_g5 as g5  # noqa: E402
import tuoi_tho as tt  # noqa: E402

h = tt.hernes()
b = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])["b"]
world = tt.World.draw(np.random.default_rng(0), b, tt.Scenario.boundary())
out = {}
for st in ("cruise", "coast"):
    g6.wait_memory(2.0, "so_sanh_et")
    lin = g5.DayPool(f"chu_ky_{st}_d00_base.npz", g6.BASE)
    et = g5.DayPool(f"chu_ky_{st}_d00_base_et.npz", g6.BASE)
    assert np.array_equal(lin.days, et.days)
    rng = np.random.default_rng(20261020)
    r_et, r_up = [], []
    for _ in range(500):
        q = lin.sample(rng)
        d_lin = lin.mod(lin.true(q, world, 0.0, 1.0, 1.0))
        d_et = et.mod(et.true(q, world, 0.0, 1.0, 1.0))
        d_up = lin.mod(lin.true(q, world, 0.0, 1.0, 1.10))
        r_et.append(d_et / d_lin)
        r_up.append(d_up / d_lin)
    r_et, r_up = np.array(r_et), np.array(r_up)
    th_lin, th_et = np.load(HERE / f"chu_ky_{st}_d00_base.npz")["theta_max"], \
        np.load(HERE / f"chu_ky_{st}_d00_base_et.npz")["theta_max"]
    out[st] = dict(ratio_et_q=[float(x) for x in np.quantile(r_et, (0.0, 0.1, 0.5, 0.9, 1.0))],
                   ratio_et_q_modules=[[float(x) for x in np.quantile(r_et[:, j], (0.1, 0.5, 0.9))]
                                       for j in range(r_et.shape[1])],
                   headroom_q=[float(x) for x in np.quantile(r_up, (0.0, 0.1, 0.5, 0.9, 1.0))],
                   share_et_above_headroom=float(np.mean(r_et > r_up)),
                   theta_max_median=[float(np.median(th_lin)), float(np.median(th_et))])
    print(st, json.dumps(out[st]), flush=True)
    del lin, et
(HERE / "so_sanh_et.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
print("written so_sanh_et.json")
