"""P3 fourth round (20/10/2026): summaries of the experiments of chay_vong4.sh and of e3_thiet_ke.py, together with
the calibrated-part coverage of the parametric certificates in the stored static runs (A), into vong4_tong_hop.json.

  static   per configuration, bench size, level and certificate: coverage of the true damage (D* <= C), of the
           calibrated part (U_new <= C: the field module's paired score at its true strength) and the median ratio of
           the certificate to the oracle quantile, pooled over repetitions and modules, with Wilson 95 % intervals
  closed   C-type runs (weak subpopulation, anchor tests): served intervals, failed lives, cost per 100 intervals for
           c_F = 10 (planned replacements), per run
  mission  M-type runs: the same, including the replanned ages after a detection delay

Run: python -I vong4_tong_hop.py
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
CF = 10.0


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def load(tag):
    f = HERE / f"g6_{tag}.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def static(tag):
    rows = load(tag)
    if not rows:
        return None
    acc = {}
    for rec in rows:
        for m in rec["modules"]:
            for n1, x in m["n1"].items():
                for a, cs in x["certs"].items():
                    orc = m["oracle"].get(a)
                    for c, v in list(cs.items()) + ([("U6 paired anchored", x["anchored"][a])]
                                                    if "anchored" in x else []):
                        s = acc.setdefault(n1, {}).setdefault(a, {}).setdefault(c, dict(n=0, cov=0, cal=0, ratio=[],
                                                                                         finite=0))
                        s["n"] += 1
                        s["cov"] += m["D_star"] <= v
                        if c != "U6 paired anchored":
                            s["cal"] += x["U_new"] <= v
                        s["finite"] += math.isfinite(v)
                        if orc and math.isfinite(v):
                            s["ratio"].append(v / orc)
    out = {}
    for n1, d in acc.items():
        for a, dd in d.items():
            for c, s in dd.items():
                r = sorted(s["ratio"])
                out.setdefault(n1, {}).setdefault(a, {})[c] = dict(
                    n=s["n"], coverage=s["cov"] / s["n"], coverage_ci=wilson(s["cov"], s["n"]),
                    calibrated=None if c == "U6 paired anchored" else s["cal"] / s["n"],
                    calibrated_ci=None if c == "U6 paired anchored" else wilson(s["cal"], s["n"]),
                    finite=s["finite"] / s["n"], ratio_median=r[len(r) // 2] if r else None)
    return out


def cost(runs, cf=CF):
    """Cost per 100 intervals in planned replacements: each life ends with one replacement (planned, or a failure at
    c_F), renewal-reward over lives."""
    n = sum(x["intervals"] for x in runs)
    c = sum(cf if x["failed"] else 1.0 for x in runs)
    return 100.0 * c / max(n, 1)


def closed(tag):
    rows = load(tag)
    if not rows:
        return None
    keys = rows[0]["runs"].keys()
    out = {}
    for k in keys:
        rs = [r["runs"][k] for r in rows]
        nf = sum(x["failed"] for x in rs)
        cov = [x.get("coverage") for x in rs if x.get("coverage") is not None]
        out[k] = dict(lives=len(rs), served=sum(x["intervals"] for x in rs) / len(rs), failed=nf / len(rs),
                      failed_ci=wilson(nf, len(rs)), cost10=cost(rs), cost50=cost(rs, 50.0),
                      coverage=sum(cov) / len(cov) if cov else None)
    return out


def mission(tag):
    rows = load(tag)
    if not rows:
        return None
    out = {}
    for k in rows[0]["runs"]:
        rs = [r["runs"][k] for r in rows]
        nf = sum(x["failed"] for x in rs)
        bits = [b for x in rs for b in (x.get("bits") or [])]
        out[k] = dict(lives=len(rs), served=sum(x["intervals"] for x in rs) / len(rs), failed=nf / len(rs),
                      failed_ci=wilson(nf, len(rs)), cost10=cost(rs), cost50=cost(rs, 50.0),
                      coverage=(sum(bin(b).count("1") for b in bits) / (2 * len(bits))) if bits else None)
    return out


J = {"static": {}, "closed": {}, "mission": {}}
for tag in ("A_boundary_k0", "A_evidence_k0", "P_boundary_weak", "P_evidence_weak", "P_boundary_a001",
            "P_boundary_weak_a001", "E_boundary_et_cruise", "E_boundary_et_coast", "E_evidence_et_cruise",
            "E_evidence_et_coast"):
    s = static(tag)
    if s:
        J["static"][tag] = s
for tag in ("C_boundary_cruise_weak", "C_boundary_coast_weak",
            *(f"V_{sc}_{st}_{an}" for sc in ("evidence", "boundary") for st in ("cruise", "coast")
              for an in ("ideal", "real")),
            *(f"C_{sc}_{st}" for sc in ("evidence", "boundary") for st in ("cruise", "coast"))):
    c = closed(tag)
    if c:
        J["closed"][tag] = c
for tag in ("M_boundary_cruise_coast_dl", "M_boundary_cruise_coast_weak"):
    m = mission(tag)
    if m:
        J["mission"][tag] = m
e3 = HERE / "e3_thiet_ke.json"
if e3.exists():
    J["e3_design"] = json.loads(e3.read_text(encoding="utf-8"))
(HERE / "vong4_tong_hop.json").write_text(json.dumps(J, indent=1), encoding="utf-8", newline="\n")

# console summary
for tag, s in J["static"].items():
    print(f"== {tag}")
    for n1, d in s.items():
        for a, dd in d.items():
            print(f"  n1 {n1:>3s} a {a:>5s}: " + "; ".join(
                f"{c.split(' ', 1)[-1]} {x['coverage']:.3f}/{'-' if x['calibrated'] is None else format(x['calibrated'], '.3f')}"
                f" r{'-' if x['ratio_median'] is None else format(x['ratio_median'], '.2f')}" for c, x in dd.items()))
for grp in ("closed", "mission"):
    for tag, d in J[grp].items():
        print(f"== {tag}")
        for k, x in d.items():
            print(f"  {k:28s} served {x['served']:6.1f} failed {100 * x['failed']:5.1f} % cost {x['cost10']:6.2f}/{x['cost50']:6.2f}"
                  + ("" if x["coverage"] is None else f" cov {x['coverage']:.3f}"))
