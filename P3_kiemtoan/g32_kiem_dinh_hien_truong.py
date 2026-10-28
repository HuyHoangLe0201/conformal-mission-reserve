"""Power of a field trial of the replacement rule (G32, 28/10/2026). Exact binomial; pure Python, no simulation needed.

Observable in a trial: each of N legs is replaced at its first failing certificate, so the only observable event is a
failure in service before that replacement. The failure share p is compared with a target. Rates come from Table 4 of the
manuscript (paired certificate, replacement rule): 3.3 % (truth at the margins, cruising), 1.3 % (coasting), 11.7 % (beyond).
"""
import json
from math import comb

def cdf(k, n, p):
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1))

def sf(k, n, p):  # P(X >= k)
    return 1 - cdf(k - 1, n, p) if k > 0 else 1.0

def safety(n, p0, p1, a=0.05):
    """H0: p >= p0. Reject (show p < p0) if failures <= c, largest c with P_{p0}(X<=c) <= a. Power at p1."""
    c = -1
    while c + 1 <= n and cdf(c + 1, n, p0) <= a:
        c += 1
    return c, (cdf(c, n, p1) if c >= 0 else 0.0)

def violation(n, p0, p1, a=0.05):
    """H0: p <= p0. Reject (show p > p0) if failures >= c, smallest c with P_{p0}(X>=c) <= a. Power at p1."""
    c = 0
    while c <= n and sf(c, n, p0) > a:
        c += 1
    return c, (sf(c, n, p1) if c <= n else 0.0)

def need(fun, p0, p1, target=0.80, nmax=3000):
    for n in range(5, nmax):
        c, pw = fun(n, p0, p1)
        if pw >= target:
            return n, c, pw
    return None

out = {"safety": [], "violation": [], "power": []}
for p1 in (0.013, 0.033, 0.05):
    for p0 in (0.10, 0.20):
        r = need(safety, p0, p1)
        out["safety"].append({"p_true": p1, "p_target": p0, "N": r[0], "max_failures": r[1], "power": round(r[2], 3)})
for p0, p1 in ((0.033, 0.117), (0.013, 0.117), (0.05, 0.117), (0.033, 0.20)):
    r = need(violation, p0, p1)
    out["violation"].append({"p_stated": p0, "p_true": p1, "N": r[0], "min_failures": r[1], "power": round(r[2], 3)})
for n in (20, 50, 100, 200):
    row = {"N": n}
    c, pw = safety(n, 0.10, 0.033); row["safety_0.033_vs_0.10"] = round(pw, 3)
    c, pw = violation(n, 0.033, 0.117); row["violation_0.117_vs_0.033"] = round(pw, 3)
    out["power"].append(row)
json.dump(out, open("g32_kiem_dinh_hien_truong.json", "w"), indent=1)
for k, v in out.items():
    print(k)
    for r in v: print("  ", r)
