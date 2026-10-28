"""P3: where do the negative results come from? (19/10/2026)

1. Code check. Theorem 1(a) makes the calibrated part exact: P(U_new <= C_P) lies in [1 - alpha, 1 - alpha + 1/(m+1)]
   with m paired scores. Across every static run, the observed share is compared with the exact level
   ceil((1 - alpha)(m + 1)) / (m + 1); z uses the standard error over repetitions (both modules of a repetition share
   its world). A coding error in the scores or the ranks would show as a systematic shift.
2. Decomposition of the misses. U_new = exp(-S_new) (D0 + V_new) is the margined score of the field module at its
   true strength, so D* <= U_new is the event that the stated components (margin, thermal bound, Miner factor) hold
   for that module. A miss D* > C_P is then either a calibrated miss (U_new > C_P while the stated event holds) or a
   stated-event violation (D* > U_new). Theorem 1 bounds only the first by alpha; the second is the stated level.

Run: python chan_doan_am.py -> chan_doan_am.md
"""
import json
import math
import site
import sys
from pathlib import Path

sys.path.insert(0, site.getusersitepackages())
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
MD = []


def p(s=""):
    MD.append(s)
    print(s)


def load(tag):
    f = HERE / f"g6_{tag}.jsonl"
    rows = {}
    for x in f.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            rows.setdefault(r["rep"], r)
    return [rows[k] for k in sorted(rows)]


def exact(m, a):
    k = math.ceil((1 - a) * (m + 1) - 1e-12)
    return k / (m + 1) if k <= m else 1.0


def records_A(tag, n1, a):
    """Per repetition: list of (U_new, C_P, D*) for the modules (experiments A/Q)."""
    out = []
    for r in load(tag):
        rec = []
        for x in r["modules"]:
            y = x["n1"][n1]
            rec.append((y["U_new"], y["certs"][a]["U6 paired"], x["D_star"], min(y["n_scores"], 100)))
        out.append(rec)
    return out


def records_T(tag, design):
    out = []
    for r in load(tag):
        out.append([(x["designs"][design]["U_new"], x["designs"][design]["paired"]["0.1"],
                     x["designs"][design]["D_star"], 40) for x in r["modules"]])
    return out


def records_U(tag, n1, a):
    out = []
    for r in load(tag):
        out.append([(x["n1"][n1]["U_new"], x["n1"][n1]["certs"][a]["paired"], x["D_star"],
                     min(x["n1"][n1]["n_scores"], 100)) for x in r["modules"]])
    return out


def rep_stats(recs, f):
    v = np.array([np.mean([f(*t) for t in rec]) for rec in recs], float)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


rows = []
for tag in ("A_boundary_k0", "A_evidence_k0", "A_adverse_strict_k0", "A_beyond_k0", "Q_boundary_int", "Q_evidence_int",
            "Q_boundary_lutz", "Q_boundary_nc", "Q_evidence_nc", "Q_boundary_th", "Q_evidence_th", "Q_boundary_seq",
            "Q_evidence_seq"):
    if not (HERE / f"g6_{tag}.jsonl").exists():
        continue
    n1s = list(load(tag)[0]["modules"][0]["n1"])
    for n1 in n1s:
        for a in ("0.1", "0.05"):
            recs = records_A(tag, n1, a)
            if not math.isfinite(recs[0][0][1]):
                continue
            rows.append((tag, n1, a, recs))
for tag in ("T_boundary", "T_evidence", "T_evidence_b"):
    for d in ("resampled", "blocks", "forward", "backward"):
        rows.append((tag, d, "0.1", records_T(tag, d)))
for tag in ("U_boundary", "U_evidence"):
    for n1 in ("10", "20", "40", "80"):
        for a in ("0.1", "0.05"):
            recs = records_U(tag, n1, a)
            if math.isfinite(recs[0][0][1]):
                rows.append((tag, n1, a, recs))

p("## 1. Calibrated part against the exact level of Theorem 1(a)\n")
p("| run | n1 or design | alpha | scores m | exact level | observed | z |")
p("|---|---|---|---|---|---|---|")
zs = []
for tag, n1, a, recs in rows:
    m = recs[0][0][3]
    lev = exact(m, float(a))
    obs, se = rep_stats(recs, lambda U, C, D, m_: float(U <= C))
    z = (obs - lev) / se if se > 0 else 0.0
    zs.append(z)
    p(f"| {tag} | {n1} | {a} | {m} | {lev:.3f} | {obs:.3f} | {z:+.2f} |")
zs = np.array(zs)
p(f"\n{len(zs)} cells: mean z {zs.mean():+.2f} (0 if unbiased; cells overlap, so not independent), "
  f"sd {zs.std(ddof=1):.2f}, below -2: {int(np.sum(zs < -2))}, above +2: {int(np.sum(zs > 2))}; "
  f"expected beyond +-2 by chance: about {0.046 * len(zs):.1f}")

DEC = {}
p("\n## 2. Misses of the paired certificate: calibrated miss or stated-event violation\n")
p("| run | n1 | alpha | miss rate D*>C_P | stated event violated D*>U_new | misses with stated event held |")
p("|---|---|---|---|---|---|")
for tag, n1, a, recs in rows:
    if not tag.startswith(("A_", "Q_")) or a != "0.1":
        continue
    flat = [t for rec in recs for t in rec]
    miss = np.array([D > C for U, C, D, _ in flat])
    viol = np.array([D > U for U, C, D, _ in flat])
    held = np.array([(D > C) and (D <= U) for U, C, D, _ in flat])
    p(f"| {tag} | {n1} | {a} | {miss.mean():.3f} | {viol.mean():.3f} | {held.mean():.3f} |")
    DEC[f"{tag}|{n1}|{a}"] = dict(miss=float(miss.mean()), violated=float(viol.mean()), held_miss=float(held.mean()))

SUM = dict(cells=int(len(zs)), mean_z=float(zs.mean()), sd_z=float(zs.std(ddof=1)), below=int(np.sum(zs < -2)),
           above=int(np.sum(zs > 2)))
(HERE / "chan_doan_am.json").write_text(json.dumps(dict(calibrated=SUM, misses=DEC), indent=1), encoding="utf-8",
                                        newline="\n")
(HERE / "chan_doan_am.md").write_text("\n".join(MD) + "\n", encoding="utf-8", newline="\n")
