"""P3 (19/10/2026): multi-split paired certificate on the real bench data of Hernes et al. (2021), cruising mission.

A single random split of the bench into fitting and scoring modules makes the certificate depend on the split (Section
6.3: median 0.030 over 200 splits, 10-90 % range 0.016-0.058). Multi-split conformal prediction (Solari and
Djordjilovic 2022) aggregates the splits: by Markov's inequality, the median over B splits of certificates each at level
a covers with probability at least 1 - 2a, so a = alpha/2 per split gives 1 - alpha without any assumption on how the
splits depend on each other. With the 'half' scheme (18 scores) the paired certificate at 0.05 is infinite (it needs
19), so the guaranteed multi-split certificate uses the 'two' scheme (two fitting modules per run, 24 scores).

The recorded intervals are those of vi_du_so.py (the same 50 cruising intervals); two independent sets of B = 199
splits per scheme show how much the multi-split certificate still depends on the draw of the splits. The certificate of
a module is aggregated per module, and that of the leg is the larger of its two modules.

Run: python -I chia_nhieu.py -> chia_nhieu.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import thu_nghiem_g6 as g6  # noqa: E402  (also disables the WMI query of scipy on Python 3.12)
import numpy as np  # noqa: E402

import chung_chi as cc  # noqa: E402
import thu_nghiem_g5 as g5  # noqa: E402
import tuoi_tho as tt  # noqa: E402

N2, SEED_REC, B, SEEDS = 50, 20261018, 199, (20261019, 20261020)
ALPHAS = (0.10, 0.05)
BASE = g6.BASE


def main():
    g6.wait_memory(1.5, "chia_nhieu")
    h = tt.hernes()
    X = tt.features(h["dT"], h["ton"], h["I"])
    pool = g5.DayPool("chu_ky_cruise_d00_base.npz", BASE)
    groups = pool.groups
    rng = np.random.default_rng(SEED_REC)
    rec = [pool.sample(rng) for _ in range(N2)]                  # as in vi_du_so.py
    out = dict(n_recorded=N2, B=B, seeds=list(SEEDS), alphas=list(ALPHAS), schemes={})
    for scheme in ("half", "two"):
        res = {}
        for seed in SEEDS:
            rs = np.random.default_rng(seed)
            C = {a: [] for a in ALPHAS}                              # [split][module]
            for _ in range(B):
                fi, si = tt.split(h["run"], scheme, rs)
                fitd = tt.fit(h["dT"][fi], h["ton"][fi], h["I"][fi], h["Nf"][fi], h["run"][fi])
                S = np.log(h["Nf"][si]) - X[si] @ fitd["b"]
                m = min(len(S), N2)
                V = np.array([pool.model(q, fitd, BASE) for q in rec[:m]])
                for a in ALPHAS:
                    C[a].append([cc.u6_paired(S[:m], V[:, g], np.zeros(len(g)), a) for g in groups])
            r = dict(n_scores=int(m))
            for a in ALPHAS:
                c = np.array(C[a])                                   # [B, K]
                leg = c.max(1)
                fin = np.isfinite(leg)
                multi_mod = np.median(c, 0)                          # per module, over the splits
                r[f"{a}"] = dict(finite_share=float(fin.mean()),
                                 single_leg_median=float(np.median(leg)),
                                 single_leg_q10_q90=[float(np.quantile(leg, q)) for q in (0.1, 0.9)],
                                 single_leg_min_max=[float(leg.min()), float(leg.max())],
                                 multi_modules=[float(v) for v in multi_mod],
                                 multi_leg=float(multi_mod.max()))
            res[str(seed)] = r
        out["schemes"][scheme] = res
        for seed in SEEDS:
            for a in ALPHAS:
                v = res[str(seed)][f"{a}"]
                print(f"{scheme} seed {seed} alpha {a}: finite {v['finite_share']:.2f}, single median "
                      f"{v['single_leg_median']:.4f} (10-90 % {v['single_leg_q10_q90'][0]:.4f}-"
                      f"{v['single_leg_q10_q90'][1]:.4f}), multi-split leg {v['multi_leg']:.4f}", flush=True)
    (HERE / "chia_nhieu.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
