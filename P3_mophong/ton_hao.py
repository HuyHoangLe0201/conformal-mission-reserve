"""P3 case study (G5): chip losses of one Si-NPC leg and the loss patterns available to thermal balancing.

Leg: three-level NPC with CM1000HC-66R single-switch modules (3.3 kV / 1 kA, the class of the bench modules;
linh_kien_hv.py), Vdc = 3000 V, ten chips: IGBTs T1-T4, their diodes D1-D4 and the clamp diodes D5, D6, in six modules
[T1 D1], [T2 D2], [T3 D3], [T4 D4], [D5], [D6] (one module per switch position; a clamp diode is taken as a module of the
same diode chip, as an IGBT module with its gate held off would be). The loss engine follows the P4 code of the project (carrier-averaged losses over the fundamental angle,
zero-sequence offset as the redundancy; P4_code/sim/tonhao.py), restricted to the NPC leg and rewritten here so that P3
stands alone.

Model choices (stated in the case study):
  - FS-MPC is represented by its averaged effect: losses of a carrier-based modulation at the average switching
    frequency FSW = 750 Hz per device (the 1.7 kV design of the project switched at 1500 Hz, the limit of its FS-MPC;
    3.3 kV devices switch at lower frequencies because their switching energies are four to eight times larger).
  - Device curves are taken at a fixed 150 C. The losses then do not depend on temperature, the thermal model is
    linear, and the scaling error model of Remark P3.7 holds exactly (the upper junction temperature is an affine image
    of the true one); the fixed 150 C curves over-state the losses at lower temperatures.
  - Thermal balancing (Novak et al. 2021 use the redundant states of an FS-MPC) acts through the zero-sequence offset,
    chosen per 10-degree bin from NOFF values across the feasible band, with a pattern repeated every 120 degrees (the
    offset is common to the three legs) and zero average neutral-point current over the fundamental period. The NP
    voltage ripple is not constrained here (a simplification against P4).
  - Three patterns per operating point: 'base' (centred offset), 'out' (minimum loss of the outer IGBTs T1+T4), 'in'
    (minimum loss of the inner IGBTs T2+T3). A balancing law mixes them; losses are linear in the pattern weights, so a
    mix keeps the NP constraint.

Output tables (build_tables): average chip losses and the peak-to-peak junction ripple at the output frequency, on a
grid of peak current, modulation index and sign of the power factor (|cos phi| = 0.95, as in doan_tau.py).

Run: python ton_hao.py -> bang_ton_hao.npz, ton_hao_log.txt (printed checks)
The 1.7 kV design used until 17/10/2026 (CM600DX-34T dual modules, 1800 V, 1500 Hz) is kept in luu_1700V_20261018/.
"""
import math
import site
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

import doan_tau  # noqa: E402
import linh_kien_hv as lk  # noqa: E402
import nhiet  # noqa: E402

VDC = doan_tau.P["Vdc"]
FSW = 750.0
T_CURVE = 150.0
COSPHI = 0.95
NTH = 360
NBIN = 36
NB3 = NBIN // 3
NOFF = 5
TH = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
BIN = (np.arange(NTH) * NBIN) // NTH
CLAMP_TOL = 1e-9
NHARM = 12

ROLES = [("T", 1), ("T", 2), ("T", 3), ("T", 4), ("D", 1), ("D", 2), ("D", 3), ("D", 4), ("D", 5), ("D", 6)]
CHIP = {r: i for i, r in enumerate(ROLES)}
NAMES = [f"{r[0]}{r[1]}" for r in ROLES]
NCHIP = len(ROLES)
MODULES = [[("T", 1), ("D", 1)], [("T", 2), ("D", 2)], [("T", 3), ("D", 3)], [("T", 4), ("D", 4)], [("D", 5)], [("D", 6)]]
MOD_OF = np.array([next(j for j, ms in enumerate(MODULES) if r in ms) for r in ROLES])
NMOD = len(MODULES)
RTH_JC = np.array([lk.HV_RTH_JC_IGBT if r[0] == "T" else lk.HV_RTH_JC_FWD for r in ROLES])
FOSTER_R, FOSTER_TAU = lk.HV_FOSTER_R, lk.HV_FOSTER_TAU
# conduction paths per level and current sign, commutation pairs (transistor, diode) per level pair and sign
COND = {"P": {+1: [("T", 1), ("T", 2)], -1: [("D", 1), ("D", 2)]},
        "N": {+1: [("D", 3), ("D", 4)], -1: [("T", 3), ("T", 4)]},
        "O": {+1: [("D", 5), ("T", 2)], -1: [("T", 3), ("D", 6)]}}
COM = {"PO": {+1: (("T", 1), ("D", 5)), -1: (("T", 3), ("D", 1))},
       "ON": {+1: (("T", 2), ("D", 4)), -1: (("T", 4), ("D", 6))}}
TARGETS = {"out": [CHIP[("T", 1)], CHIP[("T", 4)]], "in": [CHIP[("T", 2)], CHIP[("T", 3)]]}
PATTERNS = ("base", "out", "in")


def offsets(m):
    """Leg-a reference and the NOFF zero-sequence offsets across the feasible band, per angle (normalized to Vdc/2)."""
    A = 2 * m / np.sqrt(3)
    u = np.stack([A * np.sin(TH - k * 2 * np.pi / 3) for k in range(3)])
    lo, hi = -1 - u.min(0), 1 - u.max(0)
    hi = np.maximum(hi, lo)
    d = np.linspace(-1, 1, NOFF)
    return u[0], (lo + hi)[None, :] / 2 + d[:, None] * (hi - lo)[None, :] / 2


def loss_angle(I_pk, m, cosphi, fsw=FSW, vdc=VDC, T=T_CURVE):
    """Carrier-averaged chip losses L[NOFF, NTH, NCHIP] (W) and leg-a neutral-point current inp[NOFF, NTH] (A)."""
    phi = math.acos(float(np.clip(cosphi, -1, 1)))
    i = I_pk * np.sin(TH - phi)
    sgn = np.where(i >= 0, 1, -1)
    ai = np.abs(i)
    ua, v0 = offsets(m)
    vsw = vdc / 2
    vT, vD = lk.hv_vce(ai, T), lk.hv_vf(ai, T)
    eT = lk.hv_eon(ai, T, vsw) + lk.hv_eoff(ai, T, vsw)
    eD = lk.hv_err(ai, T, vsw)
    L = np.zeros((NOFF, NTH, NCHIP))
    inp = np.zeros((NOFF, NTH))
    for r in range(NOFF):
        u = np.clip(ua + v0[r], -1, 1)
        dP, dN = np.clip(u, 0, 1), np.clip(-u, 0, 1)
        duty = {"P": dP, "N": dN, "O": 1 - dP - dN}
        acc = np.zeros((NTH, NCHIP))
        for lvl in ("P", "N", "O"):
            for s in (1, -1):
                msk = (sgn == s) * duty[lvl]
                for role in COND[lvl][s]:
                    acc[:, CHIP[role]] += msk * (vT if role[0] == "T" else vD) * ai
        act = {"PO": (u > CLAMP_TOL) & (u < 1 - CLAMP_TOL), "ON": (u < -CLAMP_TOL) & (u > -1 + CLAMP_TOL)}
        for pair in ("PO", "ON"):
            for s in (1, -1):
                msk = act[pair] & (sgn == s)
                rT, rD = COM[pair][s]
                acc[:, CHIP[rT]] += msk * fsw * eT
                acc[:, CHIP[rD]] += msk * fsw * eD
        L[r] = acc
        inp[r] = duty["O"] * i
    return L, inp


def bin_mean(x, axis):
    x = np.moveaxis(x, axis, -1)
    out = x.reshape(*x.shape[:-1], NBIN, NTH // NBIN).mean(-1)
    return np.moveaxis(out, -1, axis)


def pattern_weights(L, inp, m):
    """Weights w[NB3, NOFF] of the three patterns ('base', 'out', 'in'), repeated every 120 degrees."""
    Lb, ib = bin_mean(L, 1), bin_mean(inp, 1)               # [NOFF, NBIN, NCHIP], [NOFF, NBIN]
    rep = lambda a: a[:, :NB3] + a[:, NB3:2 * NB3] + a[:, 2 * NB3:]
    W = {}
    wb = np.zeros((NB3, NOFF))
    wb[:, NOFF // 2] = 1.0
    W["base"] = wb
    A_eq = np.zeros((NB3 + 1, NB3 * NOFF))
    for b in range(NB3):
        A_eq[b, b * NOFF:(b + 1) * NOFF] = 1.0
    A_eq[NB3] = rep(ib).T.ravel()                            # zero average NP current, [b, r] order
    b_eq = np.r_[np.ones(NB3), 0.0]
    for name, chips in TARGETS.items():
        c = rep(Lb[:, :, chips].sum(-1)).T.ravel()
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        W[name] = res.x.reshape(NB3, NOFF) if res.status == 0 else wb.copy()
    return W


def pattern_losses(L, inp, w):
    """Angle-resolved chip losses p[NTH, NCHIP] of a pattern, their average and the average NP current."""
    wt = w[BIN % NB3]                                        # [NTH, NOFF]
    p = np.einsum("tr,rtk->tk", wt, L)
    i_np = np.einsum("tr,rt->t", wt, inp)
    return p, p.mean(0), float(i_np.mean())


_H = np.arange(1, NHARM + 1)
_E = np.exp(1j * np.outer(_H, np.linspace(0, 2 * np.pi, 72, endpoint=False)))     # [NH, 72]


def ripple_p2p(p, fe):
    """Peak-to-peak junction ripple of every chip, [len(fe), NCHIP], at the output frequencies fe, from the periodic
    part of the losses p[NTH, NCHIP] through the chip Foster network and the module case-sink and sink-water branches
    (thermal constants of nhiet.py)."""
    fe = np.atleast_1d(np.asarray(fe, dtype=float))
    H = np.fft.rfft(p, axis=0)[1:NHARM + 1] / NTH * 2       # [NH, NCHIP] complex amplitudes
    Hm = np.stack([H[:, MOD_OF == j].sum(1) for j in range(NMOD)], 1)[:, MOD_OF]
    w = 2 * np.pi * fe[:, None] * _H[None, :]                # [NF, NH]
    Zc = (RTH_JC[None, None, None, :] * FOSTER_R[None, None, :, None]
          / (1 + 1j * w[:, :, None, None] * FOSTER_TAU[None, None, :, None])).sum(2)   # [NF, NH, NCHIP]
    Zm = sum(R / (1 + 1j * w * tau) for R, tau in nhiet.MODULE_BRANCHES)               # [NF, NH]
    A = H[None] * Zc + Hm[None] * Zm[:, :, None]
    T = np.real(np.einsum("fhk,hq->fqk", A, _E))
    return T.max(1) - T.min(1)


GRID_I = np.arange(0.0, 610.0, 10.0)
GRID_M = np.round(np.arange(0.0, 0.951, 0.05), 3)
GRID_FE = np.geomspace(0.25, 400.0, 33)
SIGNS = (1, -1)


def build_tables(path=HERE / "bang_ton_hao.npz"):
    nI, nM, nF = len(GRID_I), len(GRID_M), len(GRID_FE)
    P = np.zeros((len(PATTERNS), 2, nI, nM, NCHIP))
    R = np.zeros((len(PATTERNS), 2, nI, nM, nF, NCHIP))
    NP = np.zeros((len(PATTERNS), 2, nI, nM))
    for si, s in enumerate(SIGNS):
        for ii, I in enumerate(GRID_I):
            if I == 0:
                continue
            for mi, m in enumerate(GRID_M):
                L, inp = loss_angle(I, max(m, 1e-3), s * COSPHI)
                W = pattern_weights(L, inp, m)
                for pi, name in enumerate(PATTERNS):
                    p, pav, inav = pattern_losses(L, inp, W[name])
                    P[pi, si, ii, mi] = pav
                    NP[pi, si, ii, mi] = inav
                    R[pi, si, ii, mi] = ripple_p2p(p, GRID_FE)
    np.savez_compressed(path, P=P, R=R, NP=NP, I=GRID_I, M=GRID_M, FE=GRID_FE, patterns=np.array(PATTERNS),
                        names=np.array(NAMES), mod_of=MOD_OF, rth_jc=RTH_JC, fsw=FSW, vdc=VDC, t_curve=T_CURVE)
    return P, R, NP


if __name__ == "__main__":
    import time
    I_RAT = doan_tau.P["I_rated"]
    t0 = time.time()
    P, R, NP = build_tables()
    print(f"tables built in {time.time() - t0:.0f} s; grid I {GRID_I[0]:.0f}-{GRID_I[-1]:.0f} A, m {GRID_M[0]}-{GRID_M[-1]}, "
          f"fe {GRID_FE[0]}-{GRID_FE[-1]:.0f} Hz")
    for si, s in enumerate(SIGNS):
        ii, mi = int(np.argmin(abs(GRID_I - I_RAT))), len(GRID_M) - 1
        for pi, name in enumerate(PATTERNS):
            p = P[pi, si, ii, mi]
            print(f"{'motoring' if s > 0 else 'braking '} I {GRID_I[ii]:.0f} A, m {GRID_M[mi]}, {name:4s}: leg loss "
                  f"{p.sum():7.1f} W; " + " ".join(f"{n} {v:5.0f}" for n, v in zip(NAMES, p))
                  + f"; NP mean {NP[pi, si, ii, mi]:+.2e} A")
    # efficiency check at the rated point: three legs against the electrical power of one inverter
    p_leg = np.interp(I_RAT, GRID_I, P[0, 0, :, -1].sum(-1))
    p_el = 1.5 * (0.95 * VDC / math.sqrt(3)) * I_RAT * COSPHI
    print(f"rated point: inverter loss {3 * p_leg / 1e3:.2f} kW against {p_el / 1e3:.0f} kW, efficiency "
          f"{100 * (1 - 3 * p_leg / p_el):.2f} %")
    for fe in (0.5, 2.0, 3.3, 10, 50, 250):
        fi = int(np.argmin(abs(GRID_FE - fe)))
        r = R[0, 0, int(np.argmin(abs(GRID_I - I_RAT))), -1, fi]
        print(f"ripple at {GRID_FE[fi]:6.2f} Hz ({I_RAT:.0f} A, m 0.95, base): " + " ".join(f"{n} {v:4.1f}" for n, v in zip(NAMES, r)))
    # at start-up the modulation index is small: ripple at low speed and full current
    for m_ in (0.05, 0.2):
        mi = int(np.argmin(abs(GRID_M - m_)))
        for fe in (0.5, 1.0, 2.0, 3.3):
            fi = int(np.argmin(abs(GRID_FE - fe)))
            r = R[0, 0, int(np.argmin(abs(GRID_I - I_RAT))), mi, fi]
            print(f"ripple at {GRID_FE[fi]:5.2f} Hz, m {GRID_M[mi]}, {I_RAT:.0f} A: T1 {r[0]:4.1f} T2 {r[1]:4.1f} D5 {r[8]:4.1f}; "
                  f"mean losses T1 {P[0, 0, int(np.argmin(abs(GRID_I - I_RAT))), mi, 0]:.0f} W, "
                  f"T2 {P[0, 0, int(np.argmin(abs(GRID_I - I_RAT))), mi, 1]:.0f} W, D5 {P[0, 0, int(np.argmin(abs(GRID_I - I_RAT))), mi, 8]:.0f} W")
