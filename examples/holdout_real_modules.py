"""Disjoint hold-out on the digitized bench modules of Hernes et al. (2021): Table S31 of the manuscript.

Run: python examples/holdout_real_modules.py    (from the repository root, after `pip install -e .`)
"""
import csv
from pathlib import Path

import numpy as np

from p3cert import holdout_coverage, loglinear_features, run_held_out

rows = list(csv.DictReader(open(Path(__file__).resolve().parent.parent / "P3_kiemtoan" / "G2_hernes2021_sohoa.csv",
                                encoding="utf-8")))
X = loglinear_features([float(r["dT_C"]) for r in rows], [float(r["t_on_s"]) for r in rows],
                       [float(r["I_load_A"]) for r in rows])
y = np.log([float(r["Nf"]) for r in rows])
runs = np.array([int(r["run"]) for r in rows])

print("random disjoint splits 12/12/10 (nominal 0.90 at alpha = 0.10, 0.80 at 0.20):")
for a, v in holdout_coverage(X, y, (12, 12, 10), reps=4000, seed=20261028).items():
    print(f"  alpha {a}: coverage {v['mean_coverage']:.3f}  (single splits {v['p05']:.2f}-{v['p95']:.2f})")

print("one whole run withheld (the model never saw that condition):")
for g, res in run_held_out(X, y, runs, reps=2000, seed=20261028).items():
    print(f"  run {g}: " + ", ".join(f"alpha {a}: {v['mean_coverage']:.3f}" for a, v in res.items()))
