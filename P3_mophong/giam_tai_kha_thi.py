"""P3 case study (G4): share of runs that still meet their recorded time when the drive is derated by d.

A derated drive (doan_tau.derated) has less tractive force and power; a run stays on time if its minimum time at the
(possibly lowered) line speed does not exceed the time used by the model. Runs are weighted by their count in
hsr_runs_model_<tag>.csv. Delay of a late run: minimum time minus the time used.

Run: python giam_tai_kha_thi.py <hsr_runs_model_tag.csv> -> giam_tai_kha_thi.json
"""
import collections
import csv
import json
import site
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

import doan_tau as dt  # noqa: E402

cnt = collections.Counter()
with open(sys.argv[1], encoding="utf-8") as f:
    for r in csv.DictReader(f):
        cnt[(float(r["L_km"]), float(r["T_model_min"]))] += 1
keys = list(cnt)
w = np.array([cnt[k] for k in keys], dtype=float)
out = {}
for d in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
    p = dt.derated(d)
    acc = dt.Acceleration(p)
    tf = np.array([dt.run_time(L * 1e3, p["v_line"], acc, p)[0] / 60 for L, _ in keys])
    T = np.array([T for _, T in keys])
    late = np.maximum(tf - T - 1e-3, 0.0)  # T_model is rounded to 0.001 min
    out[f"{d:.1f}"] = dict(line_speed_kmh=round(3.6 * p["v_line"], 1),
                          on_time_share=round(float(np.sum(w * (late <= 0.0)) / w.sum()), 4),
                          delay_min_mean=round(float(np.sum(w * late) / w.sum()), 3),
                          delay_min_p99=round(float(np.percentile(np.repeat(late, w.astype(int)), 99)), 2))
    print(d, out[f"{d:.1f}"])
(HERE / "giam_tai_kha_thi.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
