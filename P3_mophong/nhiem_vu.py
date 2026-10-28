"""P3 case study (G5): cycle database of the vehicle days of G4, for one driving strategy, derating level and loss
pattern.

For every vehicle day (ho_so.vehicle_days on hsr_runs_<tag>.csv): inverter operating points every 0.5 s
(ho_so.day_series), chip losses of one NPC leg (ton_hao tables), junction rise above the ambient temperature of the
day (nhiet.junction_rise; the day starts at ambient after the night and ends after three hours of cooling), rainflow
cycles with heating times (rainflow_ton), and fundamental-frequency ripple cycles superposed on the slow history.

Five chips are stored, T1, T2, D1, D2 and D5: the leg is symmetric, so T4, T3, D4, D3 and D6 have the same cycles.
The ripple is the periodic steady state at the instantaneous output frequency (quasi-stationary); in the first
periods after a start it over-states the excursions, and its valley is floored at ambient.
Every cycle is stored as (day, chip, range dTheta, valley Theta_min above ambient, heating time, count, kind):
  kind 0  rainflow cycle of the slow (period-averaged) junction temperature;
  kind 1  ripple cycle with heating time 1/(2 f_e) >= TON_MIN (0.15 s, the shortest tested heating time);
  kind 2  ripple cycle with a shorter heating time, outside the reference set in the direction of shorter pulses;
          aggregated in bins of range, valley and heating time (count-weighted means), kept for the sensitivity study.
Temperatures are rises above the ambient temperature of the day, so a thermal scale factor s, a shift of ambient and
any lifetime model can be applied afterwards: Delta T = s dTheta, T_min = T_amb + s Theta_min.

Run: python nhiem_vu.py <hsr_runs_tag.csv> <strategy> <derating> <pattern> [n_days] [workers]
  -> chu_ky_<strategy>_d<derating*100>_<pattern>.npz
"""
import math
import multiprocessing as mp
import os
import platform
import site
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one thread per worker process
if hasattr(platform, "_wmi_query"):
    # Python 3.12: importing scipy queries WMI through platform, and the WMI client (fastprox.dll) crashed a run under
    # load on 16/10/2026 (0xc0000409); without WMI, platform falls back to sys.getwindowsversion()
    def _no_wmi(*_a):
        raise OSError("WMI queries disabled")
    platform._wmi_query = _no_wmi

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

TON_MIN = 0.15
COOL_S = 3 * 3600
RIP_MIN = 0.1        # K, ripple below this is not counted (as the hysteresis gate of the slow history)
CHIPS = (0, 1, 4, 5, 8)   # T1, T2, D1, D2, D5: the leg is symmetric (T4 = T1, T3 = T2, D4 = D1, D3 = D2, D6 = D5)
_W = {}


def _init(runs_path, strategy, derating, pattern):
    import doan_tau as dt_
    import ho_so
    import nhiet
    trips = ho_so.load_trips(runs_path)
    _W["days"] = ho_so.vehicle_days(trips)
    p = dt_.derated(derating) if derating > 0 else dt_.P
    _W["model"] = ho_so.RunModel(strategy, p)
    _W["tables"] = nhiet.Tables()
    _W["mix"] = {pattern: 1.0}


def _bin_ripple(dT, vmin, ton, n):
    """Aggregate many small ripple cycles: log bins of range (0.15) and heating time (0.25), 10 K bins of the valley."""
    if dT.size == 0:
        return np.zeros((0, 4))
    kb = (np.floor(np.log(dT) / 0.15) * 1_000_000 + np.floor(vmin / 10.0) * 1000
          + np.floor(np.log(ton) / 0.25)).astype(np.int64)
    u, inv = np.unique(kb, return_inverse=True)
    w = np.bincount(inv, weights=n)
    g = lambda x: np.bincount(inv, weights=n * x) / w
    return np.c_[np.exp(g(np.log(dT))), g(vmin), np.exp(g(np.log(ton))), w]


def simulate_day(k):
    import ho_so
    import nhiet
    import rainflow_ton as rf
    day, model, tab, mix = _W["days"][k], _W["model"], _W["tables"], _W["mix"]
    ser = ho_so.day_series(day, model)
    h = float(ser["t"][1] - ser["t"][0])
    I, m, fe = ser["I_pk"], ser["m"], ser["f_e"]
    sign = np.where(ser["cosphi"] < 0, -1, 1)
    p = tab.losses(I, m, sign, mix)
    n_cool = int(COOL_S / h)
    th = nhiet.junction_rise(np.r_[p, np.zeros((n_cool, p.shape[1]))], h, tab)
    rip = tab.ripple(I, m, sign, fe, mix)
    T_amb = float(np.nanmedian(ser["ambient_C"]))
    rows = []
    for c in CHIPS:
        r, v, ton, cnt = rf.rainflow(th[:, c], h)
        keep = r >= rf.GATE
        rows.append(np.c_[np.full(keep.sum(), k), np.full(keep.sum(), c), r[keep], v[keep], ton[keep], cnt[keep],
                          np.zeros(keep.sum())])
        sel = (rip[:, c] >= RIP_MIN) & (fe > 0)
        dT = rip[sel, c]
        vmin = np.maximum(th[:len(I)][sel, c] - dT / 2, 0.0)   # quasi-stationary ripple cannot dip below ambient
        ton_r = 1.0 / (2.0 * fe[sel])
        n = fe[sel] * h
        lo = ton_r >= TON_MIN
        rows.append(np.c_[np.full(lo.sum(), k), np.full(lo.sum(), c), dT[lo], vmin[lo], ton_r[lo], n[lo],
                          np.ones(lo.sum())])
        b = _bin_ripple(dT[~lo], vmin[~lo], ton_r[~lo], n[~lo])
        rows.append(np.c_[np.full(len(b), k), np.full(len(b), c), b, np.full(len(b), 2.0)])
    meta = (k, T_amb, float(th.max()), float(np.trapezoid(p.sum(1), dx=h)) / 3.6e6)
    return np.vstack(rows).astype(np.float32), meta


def build(runs_path, strategy, derating, pattern, n_days=None, workers=10):
    t0 = time.time()
    _init(runs_path, strategy, derating, pattern)
    n_all = len(_W["days"])
    idx = list(range(n_all)) if not n_days else list(np.linspace(0, n_all - 1, n_days).round().astype(int))
    with mp.Pool(workers, initializer=_init, initargs=(runs_path, strategy, derating, pattern)) as pool:
        out = pool.map(simulate_day, idx, chunksize=8)
    cyc = np.vstack([o[0] for o in out])
    meta = np.array([o[1] for o in out])
    days = _W["days"]
    info = dict(date=np.array([days[i]["date"] for i in idx]), unit=np.array([days[i]["unit"] for i in idx]),
                trains=np.array([" ".join(x["train"] for x in days[i]["trips"]) for i in idx]),
                n_runs=np.array([sum(len(x["runs"]) for x in days[i]["trips"]) for i in idx]))
    tag = f"chu_ky_{strategy}_d{int(round(100 * derating)):02d}_{pattern}"
    lv = float(os.environ.get("P3_TON_LEVEL", "0.5"))   # heating-time definition (rainflow_ton.LEVEL, 19/10/2026)
    if lv != 0.5:
        tag += f"_t{int(round(100 * lv))}"
    np.savez_compressed(HERE / f"{tag}.npz", cycles=cyc, day=meta[:, 0].astype(int), T_amb=meta[:, 1],
                        theta_max=meta[:, 2], E_leg_kWh=meta[:, 3], strategy=strategy, derating=derating,
                        pattern=pattern, names=np.array(_W["tables"].names), **info)
    print(f"{tag}: {len(idx)} days, {len(cyc):,} cycle rows, {time.time() - t0:.0f} s")
    return tag


if __name__ == "__main__":
    a = sys.argv
    build(a[1], a[2], float(a[3]), a[4], int(a[5]) if len(a) > 5 and a[5] != "all" else None,
          int(a[6]) if len(a) > 6 else 10)
