"""G28b (19/10/2026): the G28 check without cell compression, separating the stated margin from the statistical part.

G28 found steps where the margined damage of an interval falls by up to 2 % when the thermal scale rises by 0.025.
The compressed cycle cells carry an error of up to step * |b| / 2 per cycle (tuoi_tho.compress), so this run computes
the damage from the raw cycles (tt.prep + tt.damage_from_prep, no compression) with (a) the stated margin alone
(stat = 'none') and (b) the first-order statistical part of the case study (stat = 'delta'). It reports, per case, the
smallest ratio of the damage at a scale to that at the next lower scale, and the smallest ratio of the damage at the
upper scale 1.10 of the certificate to the largest damage at any scale in 0.90-1.10 (dominance needed by A3).

Run: python -I g28b_don_dieu_khong_nen.py  -> g28b_don_dieu_khong_nen.json
"""
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).parent
SIM = HERE.parent / "P3_mophong"
sys.path.insert(0, str(SIM))
os.chdir(SIM)
import thu_nghiem_g6 as g6  # noqa: E402
import numpy as np  # noqa: E402

import thu_nghiem_g5 as g5  # noqa: E402
import tuoi_tho as tt  # noqa: E402

SCALES = np.round(np.arange(0.90, 1.3001, 0.025), 3)
I110 = int(np.argmin(np.abs(SCALES - 1.10)))
N_INT, N_SPLITS, SEED = 10, 5, 20261019
g6.wait_memory(1.5, "g28b")
h = tt.hernes()
rng = np.random.default_rng(SEED)
fits = []
for _ in range(N_SPLITS):
    fi, si = tt.split(h["run"], "half", rng)
    fits.append(tt.fit(h["dT"][fi], h["ton"][fi], h["I"][fi], h["Nf"][fi], h["run"][fi]))
MG = {"stated": replace(g6.BASE, stat="none"), "delta": g6.BASE}
out = {"scales": SCALES.tolist(), "n_intervals": N_INT, "n_splits": N_SPLITS, "cases": {}}
for strategy in ("cruise", "coast"):
    for d in (0, 50):
        pool = g5.DayPool(f"chu_ky_{strategy}_d{d:02d}_base.npz", g6.BASE)
        rec = [pool.sample(rng) for _ in range(N_INT)]
        for name, mg in MG.items():
            step_min, dom_min, n_dec = np.inf, np.inf, 0
            for fitd in fits:
                for idx in rec:
                    D = []
                    for s in SCALES:
                        row = []
                        for rw in pool.raw:
                            cyc = {k: np.concatenate([rw[i][k] for i in idx])
                                   for k in ("dTheta", "theta_min", "ton", "n", "T_amb")}
                            row.append(tt.model_damage(cyc, fitd, mg, s=float(s)))
                        D.append(row)
                    D = np.array(D)                                      # [scales, chips]
                    r = D[1:] / np.maximum(D[:-1], 1e-300)
                    step_min = min(step_min, float(r.min()))
                    n_dec += int((r < 1 - 1e-9).sum())
                    dom = D[I110] / np.maximum(D[:I110 + 1].max(0), 1e-300)
                    dom_min = min(dom_min, float(dom.min()))
            key = f"{strategy}_d{d:02d}_{name}"
            out["cases"][key] = {"min_ratio_next_scale": step_min, "decreasing_steps": n_dec,
                                 "steps": int(N_SPLITS * N_INT * (len(SCALES) - 1) * len(pool.raw)),
                                 "min_ratio_1.10_to_max_below": dom_min}
            print(key, out["cases"][key], flush=True)
        del pool
(HERE / "g28b_don_dieu_khong_nen.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
print("written g28b_don_dieu_khong_nen.json")
