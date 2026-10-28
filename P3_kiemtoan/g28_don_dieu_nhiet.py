"""G28 (19/10/2026, structural audit): numerical check of the monotonicity that Assumption A3 needs in the case study.

Section 2.3 states that, under the scaling error model, the upper thermal model dominates when the margined damage is
non-decreasing in the thermal scale, and that with a statistical part bounding the damage of a whole interval this must
be checked numerically. Here the margined damage per chip of recorded 30-day intervals (first-order statistical part,
stated margin of the case study) is computed on a grid of thermal scales s = 0.90 ... 1.30 for both missions, without
derating and at 50 % derating, over random half splits of the Hernes bench; the check reports the smallest ratio of
the damage at one scale to the damage at the next lower scale (>= 1 means non-decreasing).

Run: python -I g28_don_dieu_nhiet.py  -> g28_don_dieu_nhiet.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
SIM = HERE.parent / "P3_mophong"
sys.path.insert(0, str(SIM))
import os  # noqa: E402

os.chdir(SIM)
import thu_nghiem_g6 as g6  # noqa: E402  (adds the user site; disables the WMI query of scipy)
import numpy as np  # noqa: E402

import thu_nghiem_g5 as g5  # noqa: E402
import tuoi_tho as tt  # noqa: E402

SCALES = np.round(np.arange(0.90, 1.3001, 0.025), 3)
N_INT, N_SPLITS, SEED = 20, 10, 20261019
g6.wait_memory(1.5, "g28")
h = tt.hernes()
rng = np.random.default_rng(SEED)
fits = []
for _ in range(N_SPLITS):
    fi, si = tt.split(h["run"], "half", rng)
    fits.append(tt.fit(h["dT"][fi], h["ton"][fi], h["I"][fi], h["Nf"][fi], h["run"][fi]))
out = {"scales": SCALES.tolist(), "n_intervals": N_INT, "n_splits": N_SPLITS, "cases": {}}
for strategy in ("cruise", "coast"):
    for d in (0, 50):
        pool = g5.DayPool(f"chu_ky_{strategy}_d{d:02d}_base.npz", g6.BASE)
        rec = [pool.sample(rng) for _ in range(N_INT)]
        worst, n_dec = np.inf, 0
        for fitd in fits:
            for idx in rec:
                D = np.array([g6.model_raw(pool, idx, fitd, g6.BASE, s=float(s)) for s in SCALES])  # [scales, chips]
                r = D[1:] / np.maximum(D[:-1], 1e-300)
                worst = min(worst, float(r.min()))
                n_dec += int((r < 1 - 1e-9).sum())
        key = f"{strategy}_d{d:02d}"
        out["cases"][key] = {"min_ratio_next_scale": worst, "decreasing_steps": n_dec,
                             "steps": int(N_SPLITS * N_INT * (len(SCALES) - 1) * len(pool.raw))}
        print(key, out["cases"][key], flush=True)
        del pool
(HERE / "g28_don_dieu_nhiet.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
print("written g28_don_dieu_nhiet.json")
