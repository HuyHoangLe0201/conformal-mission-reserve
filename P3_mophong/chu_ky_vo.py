"""P3 (19/10/2026, editor re-review): case-temperature cycles of the certified modules, for baseplate-solder fatigue.

For every vehicle day of nhiem_vu.py (same days, same order, same operating points and chip losses), the rise of the
case (baseplate) temperature of a module above the ambient is the part of the junction rise below the chips: the two
module branches (case to heat sink, heat sink to water) driven by the module losses, plus the water loop driven by the
losses of the three legs (nhiet.junction_rise without the chip Foster network). Its rainflow cycles, with heating
times, are stored per day for the outer module (T1 with D1, module 0) and the inner module (T2 with D2, module 1):
  (day, module, range dTheta_C, valley Theta_C,min above ambient, heating time, count).
Like the junction cycles, they are rises above the ambient, so a thermal scale and a lifetime law apply afterwards.

Run: python chu_ky_vo.py <hsr_runs_tag.csv> <strategy> [workers]  -> chu_ky_vo_<strategy>.npz (no derating, base pattern)
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import nhiem_vu as nv  # noqa: E402  (sets the thread limits and the WMI guard before numpy and scipy are used)
import numpy as np  # noqa: E402

MODS = (0, 1)


def case_day(k):
    import ho_so
    import nhiet
    import rainflow_ton as rf
    W = nv._W
    day, model, tab, mix = W["days"][k], W["model"], W["tables"], W["mix"]
    ser = ho_so.day_series(day, model)
    h = float(ser["t"][1] - ser["t"][0])
    I, m = ser["I_pk"], ser["m"]
    sign = np.where(ser["cosphi"] < 0, -1, 1)
    p = tab.losses(I, m, sign, mix)
    P = np.r_[p, np.zeros((int(nv.COOL_S / h), p.shape[1]))]
    pm = np.stack([P[:, tab.mod_of == j].sum(1) for j in range(int(tab.mod_of.max()) + 1)], 1)
    tm = sum(nhiet.first_order(pm, R, tau, h) for R, tau in nhiet.MODULE_BRANCHES)
    tw = nhiet.first_order(nhiet.N_LEGS * P.sum(1), tab.r_rad, nhiet.TAU_RAD, h)
    rows = []
    for j in MODS:
        r, v, ton, cnt = rf.rainflow(tm[:, j] + tw, h)
        keep = r >= rf.GATE
        n = int(keep.sum())
        rows.append(np.c_[np.full(n, k), np.full(n, j), r[keep], v[keep], ton[keep], cnt[keep]])
    return np.vstack(rows).astype(np.float32), (k, float((tm[:, 0] + tw).max()), float((tm[:, 1] + tw).max()))


def build(runs_path, strategy, workers=6):
    t0 = time.time()
    nv._init(runs_path, strategy, 0.0, "base")
    idx = list(range(len(nv._W["days"])))
    with mp.Pool(workers, initializer=nv._init, initargs=(runs_path, strategy, 0.0, "base")) as pool:
        out = pool.map(case_day, idx, chunksize=8)
    cyc = np.vstack([o[0] for o in out])
    meta = np.array([o[1] for o in out])
    np.savez_compressed(HERE / f"chu_ky_vo_{strategy}.npz", cycles=cyc, day=meta[:, 0].astype(int),
                        theta_c_max=meta[:, 1:], strategy=strategy, modules=np.array(MODS))
    print(f"chu_ky_vo_{strategy}: {len(idx)} days, {len(cyc):,} cycle rows, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv
    build(a[1], a[2], int(a[3]) if len(a) > 3 else 6)
