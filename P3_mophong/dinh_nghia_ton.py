"""P3 (19/10/2026, referee request): sensitivity of the certificate to the definition of the heating time of a cycle.

The case study times the heating of a cycle from the last upward crossing of the mid level of its swing to the peak
(rainflow_ton.heating_time). Two other definitions are compared on 600 evenly spaced vehicle days of each driving
strategy: from the valley (level 0, the rise time of the half cycle, which charges an envelope cycle with its hours of
rise) and from three quarters of the swing (level 0.75, shorter). Databases: chu_ky_<strategy>_d00_base_t0.npz and
_t75.npz, built by nhiem_vu.py with P3_TON_LEVEL=0.0 / 0.75 and n_days = 600 (the same days as the base database at
those positions).

For 200 coming 30-day intervals resampled from these days, with the real Hernes fit and the stated margin of the case
study: the margined damage of the leg (larger module) relative to the mid-level definition (median, 10-90 %), and the
certified reserve of a new leg over 40 half splits (paired certificate, alpha = 0.10, 50 recorded intervals).

Run: python -I dinh_nghia_ton.py -> dinh_nghia_ton.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import thu_nghiem_g6 as g6  # noqa: E402  (also disables the WMI query of scipy on Python 3.12)
import numpy as np  # noqa: E402

import thu_nghiem_g5 as g5  # noqa: E402
import tuoi_tho as tt  # noqa: E402

N_INT, N_REC, N_SPLITS, ALPHA, SEED = 200, 50, 40, 0.10, 20261019
LEVELS = (("t0", "from the valley (half-cycle rise)"), ("t75", "from three quarters of the swing"))


def main():
    g6.wait_memory(2.0, "dinh_nghia_ton")
    h = tt.hernes()
    X = tt.features(h["dT"], h["ton"], h["I"])
    f_full = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])
    out = {}
    for st in ("cruise", "coast"):
        base = g5.DayPool(f"chu_ky_{st}_d00_base.npz", g6.BASE)
        pools = {k: g5.DayPool(f"chu_ky_{st}_d00_base_{k}.npz", g6.BASE) for k, _ in LEVELS}
        days = pools["t0"].days
        for k, _ in LEVELS:
            if not np.array_equal(pools[k].days, days):
                raise ValueError("sensitivity databases on different days")
        rng = np.random.default_rng(SEED)
        ints = [rng.integers(0, len(days), 30) for _ in range(N_INT)]        # positions in the 600-day pools
        rec = ints[:N_REC]
        splits = [tt.split(h["run"], "half", rng) for _ in range(N_SPLITS)]
        fits = [(tt.fit(h["dT"][fi], h["ton"][fi], h["I"][fi], h["Nf"][fi], h["run"][fi]), si) for fi, si in splits]

        def leg_damage(pool, q, fitd, pos_map=None):
            idx = q if pos_map is None else pos_map[q]
            return float(pool.mod(pool.model(idx, fitd, g6.BASE)).max())

        def reserve(pool, pos_map=None):
            life = []
            for fitd, si in fits:
                S = np.log(h["Nf"][si]) - X[si] @ fitd["b"]
                V = np.array([pool.model(q if pos_map is None else pos_map[q], fitd, g6.BASE) for q in rec])
                life.append(min(g5.reserve("paired", S, V[:len(S), g], np.zeros(len(g)), ALPHA)[1]
                                for g in pool.groups))
            return float(np.median(life))

        ref = np.array([leg_damage(base, q, f_full, days) for q in ints])
        res = dict(days=int(len(days)), reserve_mid=reserve(base, days))
        for k, name in LEVELS:
            v = np.array([leg_damage(pools[k], q, f_full) for q in ints])
            r = v / ref
            res[k] = dict(definition=name, damage_ratio_median=float(np.median(r)),
                          damage_ratio_q10_q90=[float(np.quantile(r, 0.1)), float(np.quantile(r, 0.9))],
                          reserve=reserve(pools[k]))
            print(f"{st} {k}: damage x {np.median(r):.3f} ({np.quantile(r, 0.1):.3f}-{np.quantile(r, 0.9):.3f}), "
                  f"reserve {res[k]['reserve']:.1f} against {res['reserve_mid']:.1f}", flush=True)
        out[st] = res
    (HERE / "dinh_nghia_ton.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
