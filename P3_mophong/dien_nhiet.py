"""P3 fourth round (20/10/2026): an electro-thermal truth for the thermal bound (Assumption A3).

The case study computes the junction temperature with chip losses from the device curves at a fixed 150 C and a linear
thermal network (ton_hao.py, nhiet.py); the certificate takes 1.10 times that response as its upper thermal model.
Here the true response of the same vehicle days is computed by a different model:
  - the chip losses follow the instantaneous junction temperature: conduction from the on-state curves interpolated
    between 25, 125 and 150 C (datasheet, linh_kien_hv.py), switching energies interpolated between 125 and 150 C and
    extrapolated linearly below 125 C, where the datasheet gives no curve (floored at half the 125 C value);
  - the junction-to-case part of the thermal impedance (the chip Foster network) grows with the junction temperature as
    the thermal conductivity of silicon falls, by ((T_j + 273.15) / (T_REF + 273.15))^RTH_EXP with T_REF = 25 C, so
    that the datasheet impedance holds at 25 C and is exceeded at every operating temperature (a harsh choice), applied
    quasi-statically to the chip part of the rise;
  - three fixed-point iterations between losses and temperature on every day, starting from the 150 C losses.
The ripple at the output frequency is scaled per chip by the ratio of its losses at the instantaneous temperature to
those at 150 C. Every other step (operating points, module and water branches, rainflow with heating times, ripple
cycles, short-pulse bins) is that of nhiem_vu.py, so the database has the layout of the case-study databases and
DayPool reads it.

Run: python -I dien_nhiet.py tables                   -> bang_ton_hao_T.npz (base pattern, T = 25 ... 150 C)
     python -I dien_nhiet.py build <strategy> [workers] -> chu_ky_<strategy>_d00_base_et.npz
"""
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import nhiem_vu as nv  # noqa: E402  (thread limits, WMI guard, user site)
import numpy as np  # noqa: E402

TS = (25.0, 50.0, 75.0, 100.0, 125.0, 150.0)
RTH_EXP, T_REF, N_IT = 1.3, 25.0, 3
_W = {}


def tables(path=HERE / "bang_ton_hao_T.npz"):
    """Base-pattern chip losses P[nT, 2, nI, nM, NCHIP] on the grid of ton_hao.py at the temperatures TS."""
    import linh_kien_hv as lk
    import ton_hao as th

    def ext(table):
        def f(i, T, V):
            v = lk._lookup(lk.HV_IE, table, T, i)
            return np.maximum(v, 0.5 * lk._lookup(lk.HV_IE, table, 125.0, i)) * (V / lk.HV_VREF) ** lk.KV_HV
        return f
    lk.hv_eon, lk.hv_eoff, lk.hv_err = ext(lk.HV_EON), ext(lk.HV_EOFF), ext(lk.HV_ERR)
    P = np.zeros((len(TS), 2, len(th.GRID_I), len(th.GRID_M), th.NCHIP))
    for ti, T in enumerate(TS):
        for si, s in enumerate(th.SIGNS):
            for ii, I in enumerate(th.GRID_I):
                if I == 0:
                    continue
                for mi, m in enumerate(th.GRID_M):
                    L, _ = th.loss_angle(I, max(m, 1e-3), s * th.COSPHI, T=T)
                    P[ti, si, ii, mi] = L[th.NOFF // 2].mean(0)      # base pattern: centred offset at every angle
    np.savez_compressed(path, P=P, T=np.array(TS), I=th.GRID_I, M=th.GRID_M)
    ii, mi = int(np.argmin(abs(th.GRID_I - 554))), len(th.GRID_M) - 1
    for ti, T in enumerate(TS):
        print(f"T {T:5.0f} C: motoring 554 A, m 0.95, leg loss {P[ti, 0, ii, mi].sum():7.1f} W "
              f"(T1 {P[ti, 0, ii, mi, 0]:.0f}, D1 {P[ti, 0, ii, mi, 4]:.0f} W)", flush=True)


def _interp_table(tab, Pt, I, m, sign):
    """Losses [n, NCHIP] of one table Pt[2, nI, nM, NCHIP] at the operating points (as nhiet.Tables.losses)."""
    i0, m0, wi, wm = tab._weights(I, m)
    s = np.where(sign >= 0, 0, 1)
    v = ((1 - wi) * (1 - wm))[:, None] * Pt[s, i0, m0] + (wi * (1 - wm))[:, None] * Pt[s, i0 + 1, m0] \
        + ((1 - wi) * wm)[:, None] * Pt[s, i0, m0 + 1] + (wi * wm)[:, None] * Pt[s, i0 + 1, m0 + 1]
    return np.where((I > 0)[:, None], v, 0.0)


def _rise_parts(p, dt, tab):
    """Chip part and module-plus-water part of the junction rise for losses p (nhiet.junction_rise, split)."""
    import nhiet
    chip = np.zeros_like(p)
    for Rn, tau in zip(nhiet.tables_foster_R(), nhiet.tables_foster_tau()):
        chip += nhiet.first_order(p, 1.0, tau, dt) * (tab.rth * Rn)[None, :]
    pm = np.stack([p[:, tab.mod_of == j].sum(1) for j in range(int(tab.mod_of.max()) + 1)], 1)
    tm = sum(nhiet.first_order(pm, R, tau, dt) for R, tau in nhiet.MODULE_BRANCHES)
    tw = nhiet.first_order(nhiet.N_LEGS * p.sum(1), tab.r_rad, nhiet.TAU_RAD, dt)
    return chip, tm[:, tab.mod_of] + tw[:, None]


def _init(runs_path, strategy):
    nv._init(runs_path, strategy, 0.0, "base")
    d = np.load(HERE / "bang_ton_hao_T.npz")
    _W["PT"], _W["T"] = d["P"], d["T"]


def simulate_day_et(k):
    import ho_so
    import rainflow_ton as rf
    day, model, tab, mix = nv._W["days"][k], nv._W["model"], nv._W["tables"], nv._W["mix"]
    PT, TT = _W["PT"], _W["T"]
    ser = ho_so.day_series(day, model)
    h = float(ser["t"][1] - ser["t"][0])
    I, m, fe = ser["I_pk"], ser["m"], ser["f_e"]
    sign = np.where(ser["cosphi"] < 0, -1, 1)
    n, n_cool = len(I), int(nv.COOL_S / h)
    T_amb = float(np.nanmedian(ser["ambient_C"]))
    p150 = tab.losses(I, m, sign, mix)
    PTs = np.stack([_interp_table(tab, PT[ti], I, m, sign) for ti in range(len(TT))])      # [nT, n, NCHIP]
    pad = lambda x: np.r_[x, np.zeros((n_cool, x.shape[1]))]
    chip, rest = _rise_parts(pad(p150), h, tab)
    th = chip + rest
    for _ in range(N_IT):
        Tj = np.clip(T_amb + th[:n], TT[0], TT[-1])
        fi = np.interp(Tj, TT, np.arange(len(TT)))
        i0 = np.minimum(fi.astype(int), len(TT) - 2)
        w = fi - i0
        p = (1 - w) * np.take_along_axis(PTs, i0[None], 0)[0] + w * np.take_along_axis(PTs, i0[None] + 1, 0)[0]
        chip, rest = _rise_parts(pad(p), h, tab)
        Tj_all = T_amb + chip + rest
        th = chip * ((Tj_all + 273.15) / (T_REF + 273.15)) ** RTH_EXP + rest
    ratio = np.where(p150 > 0, p / np.maximum(p150, 1e-9), 1.0)
    rip = tab.ripple(I, m, sign, fe, mix) * ratio
    rows = []
    for c in nv.CHIPS:
        r, v, ton, cnt = rf.rainflow(th[:, c], h)
        keep = r >= rf.GATE
        rows.append(np.c_[np.full(keep.sum(), k), np.full(keep.sum(), c), r[keep], v[keep], ton[keep], cnt[keep],
                          np.zeros(keep.sum())])
        sel = (rip[:, c] >= nv.RIP_MIN) & (fe > 0)
        dT = rip[sel, c]
        vmin = np.maximum(th[:n][sel, c] - dT / 2, 0.0)
        ton_r = 1.0 / (2.0 * fe[sel])
        cnt_r = fe[sel] * h
        lo = ton_r >= nv.TON_MIN
        rows.append(np.c_[np.full(lo.sum(), k), np.full(lo.sum(), c), dT[lo], vmin[lo], ton_r[lo], cnt_r[lo],
                          np.ones(lo.sum())])
        b = nv._bin_ripple(dT[~lo], vmin[~lo], ton_r[~lo], cnt_r[~lo])
        rows.append(np.c_[np.full(len(b), k), np.full(len(b), c), b, np.full(len(b), 2.0)])
    meta = (k, T_amb, float(th.max()), float(np.trapezoid(p.sum(1), dx=h)) / 3.6e6)
    return np.vstack(rows).astype(np.float32), meta


def build(runs_path, strategy, workers=8):
    t0 = time.time()
    _init(runs_path, strategy)
    idx = list(range(len(nv._W["days"])))
    with mp.Pool(workers, initializer=_init, initargs=(runs_path, strategy)) as pool:
        out = pool.map(simulate_day_et, idx, chunksize=8)
    cyc = np.vstack([o[0] for o in out])
    meta = np.array([o[1] for o in out])
    days = nv._W["days"]
    info = dict(date=np.array([days[i]["date"] for i in idx]), unit=np.array([days[i]["unit"] for i in idx]),
                trains=np.array([" ".join(x["train"] for x in days[i]["trips"]) for i in idx]),
                n_runs=np.array([sum(len(x["runs"]) for x in days[i]["trips"]) for i in idx]))
    tag = f"chu_ky_{strategy}_d00_base_et"
    np.savez_compressed(HERE / f"{tag}.npz", cycles=cyc, day=meta[:, 0].astype(int), T_amb=meta[:, 1],
                        theta_max=meta[:, 2], E_leg_kWh=meta[:, 3], strategy=strategy, derating=0.0, pattern="base",
                        names=np.array(nv._W["tables"].names), **info)
    print(f"{tag}: {len(idx)} days, {len(cyc):,} cycle rows, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "tables":
        tables()
    else:
        build("hsr_runs_jinghu.csv", a[2], int(a[3]) if len(a) > 3 else 8)
