"""P3 audit, 14/10/2026: paired joint score versus union-bound certificate.

Model (illustrative, not a calibrated module):
  S  : log-strength, S = log W with W ~ Weibull(shape, scale 1)  (the lifetime model predicts the scale)
  V  : margined model damage of one maintenance interval, V = v0 * exp(sigma_V * N(0,1))
  D0 : accumulated model damage at the decision instant (fixed)
  U  = exp(-S) * (D0 + V)  is the end-of-interval damage when the thermal bound and the margin hold exactly.

Checks:
  1. coverage, certified damage relative to the oracle quantile, and the fraction of infinite certificates,
     for the union certificate of the 06/10 outline (alpha_S = alpha_V = alpha/2) and the paired certificate;
  2. sample-size thresholds for a finite certificate (paired, union, PAC);
  3. training-conditional (PAC) version: fraction of calibration sets whose conditional coverage reaches 1 - alpha;
  4. conditioning on survival to the decision instant does not lower the coverage of the paired certificate.

Run: python kiem_ghepcap.py   (numpy, scipy). Writes results.json next to this file.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom

SEED = 20261014
rng = np.random.default_rng(SEED)
D0, V0 = 0.30, 0.30
REPS = 20000
N2 = 500
BIG = 2_000_000


def draw_s(size, shape):
    return np.log(rng.weibull(shape, size=size))


def draw_v(size, sigma_v):
    return V0 * np.exp(sigma_v * rng.standard_normal(size))


def union_cert(s, v, a_s, a_v):
    n1, n2 = s.shape[1], v.shape[1]
    k_s = math.floor(a_s * (n1 + 1))
    k_v = math.ceil((1 - a_v) * (n2 + 1))
    q_s = np.full(s.shape[0], -np.inf) if k_s == 0 else np.sort(s, axis=1)[:, k_s - 1]
    q_v = np.full(v.shape[0], np.inf) if k_v > n2 else np.sort(v, axis=1)[:, k_v - 1]
    return np.exp(-q_s) * (D0 + q_v)


def paired_cert(s, v_pair, a):
    m = s.shape[1]
    k = math.ceil((1 - a) * (m + 1))
    if k > m:
        return np.full(s.shape[0], np.inf)
    u = np.exp(-s) * (D0 + v_pair)
    return np.sort(u, axis=1)[:, k - 1]


def summary(cert, u_new, oracle):
    finite = cert[np.isfinite(cert)]
    return {
        "coverage": round(float(np.mean(u_new <= cert)), 4),
        "infinite_fraction": round(float(np.mean(~np.isfinite(cert))), 4),
        "median_ratio_to_oracle": (round(float(np.median(finite) / oracle), 3) if finite.size else None),
    }


def pac_index(m, a, delta):
    """Smallest k with P(Bin(m, 1-a) >= k) <= delta; None if no k <= m qualifies."""
    for k in range(1, m + 1):
        if binom.sf(k - 1, m, 1 - a) <= delta:
            return k
    return None


results = {"seed": SEED, "D0": D0, "v0": V0, "reps": REPS, "n2": N2, "comparison": [], "thresholds": [],
           "pac": [], "survival": []}

# 1. union versus paired
for shape, sigma_v in [(6.0, 0.15), (6.0, 0.40), (3.5, 0.15)]:
    big = np.exp(-draw_s(BIG, shape)) * (D0 + draw_v(BIG, sigma_v))
    for a in (0.10, 0.05):
        oracle = float(np.quantile(big, 1 - a))
        for n1 in (10, 20, 40, 80):
            s = draw_s((REPS, n1), shape)
            v = draw_v((REPS, N2), sigma_v)
            u_new = np.exp(-draw_s(REPS, shape)) * (D0 + draw_v(REPS, sigma_v))
            results["comparison"].append({
                "weibull_shape": shape, "sigma_V": sigma_v, "alpha": a, "n1": n1, "oracle": round(oracle, 4),
                "union_half_half": summary(union_cert(s, v, a / 2, a / 2), u_new, oracle),
                "paired": summary(paired_cert(s, v[:, :n1], a), u_new, oracle),
            })

# 2. thresholds for a finite certificate
for a in (0.20, 0.10, 0.05, 0.01):
    results["thresholds"].append({
        "alpha": a,
        "paired_min_m": math.ceil(1 / a - 1 - 1e-12),
        "union_min_n1_alpha_half": math.ceil(2 / a - 1 - 1e-12),
        "pac_min_m_delta_eq_alpha": math.ceil(math.log(a) / math.log(1 - a) - 1e-12),
    })

# 3. PAC version (shape 6, sigma_V 0.15)
shape, sigma_v = 6.0, 0.15
ref = np.sort(np.exp(-draw_s(BIG, shape)) * (D0 + draw_v(BIG, sigma_v)))
for a, delta in [(0.10, 0.10), (0.05, 0.05)]:
    for m in (40, 80, 160):
        k = pac_index(m, a, delta)
        if k is None:
            results["pac"].append({"alpha": a, "delta": delta, "m": m, "k": None})
            continue
        u = np.sort(np.exp(-draw_s((4000, m), shape)) * (D0 + draw_v((4000, m), sigma_v)), axis=1)[:, k - 1]
        cond_cov = np.searchsorted(ref, u, side="right") / ref.size
        results["pac"].append({
            "alpha": a, "delta": delta, "m": m, "k": k,
            "fraction_of_calibration_sets_with_conditional_coverage_ge_1_minus_alpha":
                round(float(np.mean(cond_cov >= 1 - a)), 4),
            "split_conformal_k": math.ceil((1 - a) * (m + 1)),
        })

# 4. survival conditioning (paired certificate, alpha 0.10, n1 = 40)
a, n1 = 0.10, 40
s = draw_s((REPS, n1), shape)
cert = paired_cert(s, draw_v((REPS, n1), sigma_v), a)
s_new = draw_s(REPS * 5, shape)
v_new = draw_v(REPS * 5, sigma_v)
for survive_q in (0.0, 0.1, 0.3):
    s0 = np.quantile(s_new, survive_q) if survive_q > 0 else -np.inf
    keep = np.flatnonzero(s_new > s0)[:REPS]
    u_new = np.exp(-s_new[keep]) * (D0 + v_new[keep])
    results["survival"].append({
        "removed_lower_fraction_of_strength": survive_q,
        "coverage_among_survivors": round(float(np.mean(u_new <= cert[: keep.size])), 4),
    })

out = Path(__file__).with_name("results.json")
out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
print(json.dumps(results, indent=1, ensure_ascii=False))
