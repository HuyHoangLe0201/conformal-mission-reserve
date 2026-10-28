"""P3 fourth round (20/10/2026): certified reserve of a new leg on the real bench data when the certificate's thermal
model is the electro-thermal response (dien_nhiet.py) instead of the case-study response with losses at 150 C.

As vi_du_so.py: the Hernes runs split in half (16 fit, 18 scores), 40 random splits, 50 recorded 30-day intervals of each
mission, paired certificate at alpha = 0.10 for the coming interval, stated margin of the case study and the upper
thermal scale 1.10 over each thermal model; the reserve of the leg is the smaller of its two modules' reserves.

Run: python -I et_chung_chi.py  -> et_chung_chi.json
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
X = tt.features(h["dT"], h["ton"], h["I"])
out = {}
for st in ("cruise", "coast"):
    g6.wait_memory(2.0, "et_chung_chi")
    pools = {"linear 150 C": g5.DayPool(f"chu_ky_{st}_d00_base.npz", g6.BASE),
             "electro-thermal": g5.DayPool(f"chu_ky_{st}_d00_base_et.npz", g6.BASE)}
    rng = np.random.default_rng(20261020)
    rec = [pools["linear 150 C"].sample(rng) for _ in range(50)]
    res = {k: [] for k in pools}
    for _ in range(40):
        fi, si = tt.split(h["run"], "half", rng)
        fitd = tt.fit(h["dT"][fi], h["ton"][fi], h["I"][fi], h["Nf"][fi], h["run"][fi])
        S = np.log(h["Nf"][si]) - X[si] @ fitd["b"]
        for k, p in pools.items():
            V = np.array([p.model(q, fitd, g6.BASE) for q in rec])
            res[k].append(min(g5.reserve("paired", S, V[:, g], np.zeros(len(g)), 0.1)[1] for g in p.groups))
    out[st] = {k: dict(reserve_median=float(np.median(v)), reserve_q10_q90=[float(np.quantile(v, 0.1)),
                                                                          float(np.quantile(v, 0.9))])
               for k, v in res.items()}
    print(st, json.dumps(out[st]), flush=True)
    del pools
(HERE / "et_chung_chi.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
print("written et_chung_chi.json")
