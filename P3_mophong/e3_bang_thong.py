"""P3 case study: sensitivity of the test weight of the weighted certificate (Fig. 6a, experiment E3 of thu_nghiem_g5.py)
to the kernel bandwidth of the density-ratio estimate (added 17/10/2026, internal review).

E3 estimates the ratio of the target to the bench density of the loading summary X (damage-weighted mean ln dT of an
interval's cycles, critical chip) with Gaussian kernels and Silverman's bandwidth 1.06 sd n^(-1/5). This script rebuilds the
same 200 intervals (same seed and sampling as e3), recomputes X at every derating level, checks that a bandwidth factor of
1 reproduces the stored test weights (e3_cruise_base_delta.json), and gives the test weight for bandwidth factors 0.5 and
2 (both densities), without and with 20 anchor tests at the median target summary.

Run: python -I e3_bang_thong.py -> e3_bang_thong.json and a printed table
"""
import json
import site
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

import chung_chi as cc  # noqa: E402
import thu_nghiem_g5 as g5  # noqa: E402
import tuoi_tho as tt  # noqa: E402

FACTORS = (0.5, 1.0, 2.0)
ANCHORS = (0, 20)


def silverman(x):
    x = np.asarray(x, float)
    return 1.06 * max(np.std(x), 1e-3) * x.size ** (-0.2)


def main(n_int=200, seed=20261019, strategy="cruise", pattern="base", alpha_S=0.05):
    stored = json.loads((HERE / f"e3_{strategy}_{pattern}_delta.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    h = tt.hernes()
    fitd = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])
    mg = tt.Margin(stat="delta", gamma=1.2)
    out, days = {}, None
    for d in g5.LEVELS:
        f = HERE / f"chu_ky_{strategy}_d{int(round(100 * d)):02d}_{pattern}.npz"
        pool = g5.DayPool(f.name, mg)
        if days is None:
            # e3 draws the days once, from the first pool it loads (gamma 0, level 0); the pools of all levels hold the
            # same vehicle days, so the positions are the same
            days = [pool.sample(rng) for _ in range(n_int)]
        X = []
        for q in days:
            k = int(np.argmax(pool.model(q, fitd, mg)))
            Vc = np.vstack([pool.cells[k][i][0] for i in q])
            la = np.concatenate([pool.cells[k][i][1] for i in q])
            _, parts = tt.damage_from_prep(*tt.compress(Vc, la), fitd, tt.Margin(stat="none", gamma=1.2),
                                           return_parts=True)
            Vcc, _ = tt.compress(Vc, la)
            X.append(float(np.sum(parts * Vcc[:, 1]) / parts.sum()))
        X = np.array(X)
        res = {"X_median_K": float(np.exp(np.median(X)))}
        for n_a in ANCHORS:
            x_b = np.r_[np.log(h["dT"]), np.full(n_a, np.median(X))]
            for fac in FACTORS:
                w = cc.kde_ratio(x_b, X, bw_cal=fac * silverman(x_b), bw_tgt=fac * silverman(X))
                wb, wt = w(x_b), w(np.median(X))[0]
                res[f"p_test_anchors_{n_a}_bw_{fac:g}"] = float(wt / (wb.sum() + wt))
        ref = stored["a"][str(d)]
        res["check_bw1_matches_e3"] = all(abs(res[f"p_test_anchors_{n}_bw_1"] - ref[f"p_test_anchors_{n}"]) < 1e-9
                                          for n in ANCHORS)
        out[str(d)] = res
        print(f"derating {d:.1f}: X {res['X_median_K']:.1f} K; " + "; ".join(
            f"n_a={n}: " + ", ".join(f"bw x{fac:g} {res[f'p_test_anchors_{n}_bw_{fac:g}']:.3f}" for fac in FACTORS)
            for n in ANCHORS) + f"; matches e3: {res['check_bw1_matches_e3']}", flush=True)
    (HERE / "e3_bang_thong.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
