"""P3 (added 17/10/2026; since 18/10/2026 from the draws stored with each repetition, per module): results of the
adverse truths broken down by the number of stated components that each world and field module exceed.

Components that act on the missions (the activation energy does not: the damaging cycles stay below 60 C): swing
exponent below 26 K (true shift below -1.2) and the two heating-time slopes (4.5-60 s and above 60 s, above 0.5), drawn
per world; Miner factor (below 0.85) and thermal scale (above 1.10), drawn per module. A module record counts the
components that its world and its own draws exceed; a leg life counts a module component when either module exceeds it.

Run: python -I g6_vuot_nhieu.py  -> g6_vuot_nhieu.json
"""
import json
import site
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(site.getusersitepackages())     # numpy from the user site-packages, also under python -I
import numpy as np  # noqa: E402

from g6_tong_hop import binom_ci, load  # noqa: E402

STATIC = ("A_adverse_strict_k0", "A_beyond_k0")
LOOP = ("C_adverse_strict_cruise", "C_adverse_strict_coast", "C_beyond_cruise", "C_beyond_coast")
CERTS = ("U6 paired", "U5 union", "U2 Monte Carlo", "U3 Bayes")
GROUPS = ("0", "1", "2", "3+")


def world_count(w):
    return int(w["beta_shift"] < -1.2) + int(w["gamma_t"] > 0.5) + int(w["gamma_h"] > 0.5)


def module_count(fd, j):
    return int(fd["kappa"][j] < 0.85) + int(fd["s_th"][j] > 1.10)


def group(k):
    return str(k) if k < 3 else "3+"


OUT = {}
for tag in STATIC:
    rows = load(tag)
    if not rows:
        continue
    recs = [(world_count(r["world"]) + module_count(r["field"], j), r["world"]["Ea"] > 0.159, m)
            for r in rows for j, m in enumerate(r["modules"])]
    ks = np.array([x[0] for x in recs])
    res = OUT[tag] = dict(records=len(recs), worlds=len(rows), share_any=float(np.mean(ks > 0)),
                          share_any_with_Ea=float(np.mean((ks > 0) | np.array([x[1] for x in recs]))),
                          count_distribution={g: int(np.sum([group(k) == g for k in ks])) for g in GROUPS},
                          coverage={})
    print(f"{tag}: {len(rows)} worlds, {len(recs)} module records; at least one acting component exceeded "
          f"{res['share_any']:.2f} (with E_a {res['share_any_with_Ea']:.2f}); counts {res['count_distribution']}")
    D = np.array([x[2]["D_star"] for x in recs])
    for n1 in rows[0]["modules"][0]["n1"]:
        for c in CERTS:
            C = np.array([x[2]["n1"][n1]["certs"]["0.1"][c] for x in recs])
            for g in GROUPS:
                sel = np.array([group(k) == g for k in ks])
                if sel.any():
                    kk = int(np.sum(D[sel] <= C[sel]))
                    res["coverage"][f"{n1}|{c}|{g}"] = dict(n=int(sel.sum()), covered=kk,
                                                            ci=list(binom_ci(kk, int(sel.sum()))))
        print(f"  n1 {n1}: " + "; ".join(
            f"{c.split()[1]} " + ", ".join(f"{g}: {res['coverage'][f'{n1}|{c}|{g}']['covered']}/"
                                            f"{res['coverage'][f'{n1}|{c}|{g}']['n']}"
                                            for g in GROUPS if f"{n1}|{c}|{g}" in res["coverage"])
            for c in CERTS))
for tag in LOOP:
    rows = load(tag)
    if not rows:
        continue
    ks = np.array([world_count(r["world"]) + int(min(r["field"]["kappa"]) < 0.85) + int(max(r["field"]["s_th"]) > 1.10)
                   for r in rows])
    res = OUT[tag] = dict(lives=len(rows), count_distribution={g: int(np.sum([group(k) == g for k in ks]))
                                                               for g in GROUPS}, runs={})
    print(f"{tag}: {len(rows)} lives; counts {res['count_distribution']}")
    for run in rows[0]["runs"]:
        for g in GROUPS:
            sel = [i for i, k in enumerate(ks) if group(k) == g]
            if not sel:
                continue
            x = [rows[i]["runs"][run] for i in sel]
            F = int(sum(v["failed"] for v in x))
            T = float(np.mean([v["intervals"] for v in x]))
            bits = [v["bits"] for v in x if v.get("bits")]
            cov = (float(np.mean([(b >> j) & 1 for bl in bits for b in bl for j in range(2)])) if bits else None)
            res["runs"][f"{run}|{g}"] = dict(lives=len(x), failed=F, failed_ci=list(binom_ci(F, len(x))),
                                             intervals=T, cov_pooled=cov)
        print(f"  {run:22s} " + "; ".join(
            f"{g}: fail {res['runs'][f'{run}|{g}']['failed']}/{res['runs'][f'{run}|{g}']['lives']}, "
            f"T {res['runs'][f'{run}|{g}']['intervals']:.1f}"
            + ("" if res["runs"][f"{run}|{g}"]["cov_pooled"] is None
               else f", cov {res['runs'][f'{run}|{g}']['cov_pooled']:.3f}")
            for g in GROUPS if f"{run}|{g}" in res["runs"]))
(HERE / "g6_vuot_nhieu.json").write_text(json.dumps(OUT, indent=1) + "\n", encoding="utf-8", newline="\n")
