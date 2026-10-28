"""P3 case study (G5): linear thermal model of one NPC leg and its coolant, for one vehicle day.

Junction temperature of chip k above the ambient temperature of the day:
    Theta_k = own Foster network (Rth_jc * normalized R, tau; CM1000HC-66R, linh_kien_hv.py)
            + module branches of the module that holds k (case to sink, sink to water)
            + water rise above ambient (one cooling loop per inverter, driven by the loss of the three legs).
Every branch is first order and driven by piecewise-constant losses, so the update over a step is exact. The losses do
not depend on temperature (ton_hao.py), the network is linear, and Theta scales exactly with a common thermal scale
factor: the scaling error model of Remark P3.7 with the ambient temperature of the day as the constant reference.

Values (assumptions of the case study, as in P2 and P4 of the project where they exist):
  R_CS = 9.0 mK/W per module (datasheet, grease), TAU_CS = 1 s;
  R_HW = 10 mK/W per module, TAU_HW = 5 s (P2/P4 assumption, not measured; kept from the 1.7 kV design, whose dual
  module of about half the base-plate area carried two switch positions);
  water: DT_RAD_RATED = 20 K above ambient at continuous rated inverter loss, TAU_RAD = 600 s (radiator loop).
The fundamental-frequency ripple is computed separately (ton_hao.ripple_p2p) and superposed as cycles.
"""
import math
import site
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

R_CS, TAU_CS = 9.0e-3, 1.0
R_HW, TAU_HW = 10e-3, 5.0
MODULE_BRANCHES = ((R_CS, TAU_CS), (R_HW, TAU_HW))
DT_RAD_RATED, TAU_RAD = 20.0, 600.0
N_LEGS = 3


class Tables:
    """Loss and ripple tables of ton_hao.py with interpolation on (I, m, f_e)."""

    def __init__(self, path=HERE / "bang_ton_hao.npz"):
        d = np.load(path)
        self.P, self.R, self.I, self.M, self.FE = d["P"], d["R"], d["I"], d["M"], d["FE"]
        self.patterns = [str(x) for x in d["patterns"]]
        self.names = [str(x) for x in d["names"]]
        self.mod_of, self.rth = d["mod_of"], d["rth_jc"]
        self.lnfe = np.log(self.FE)
        # rated inverter loss (motoring, rated current, m 0.95, base pattern) sets the radiator resistance
        import doan_tau
        self.p_inv_rated = N_LEGS * float(np.interp(doan_tau.P["I_rated"], self.I, self.P[0, 0, :, -1].sum(-1)))
        self.r_rad = DT_RAD_RATED / self.p_inv_rated

    def _weights(self, I, m):
        fi = np.clip(np.interp(I, self.I, np.arange(len(self.I))), 0, len(self.I) - 1 - 1e-9)
        fm = np.clip(np.interp(m, self.M, np.arange(len(self.M))), 0, len(self.M) - 1 - 1e-9)
        i0, m0 = fi.astype(int), fm.astype(int)
        return i0, m0, fi - i0, fm - m0

    def losses(self, I, m, sign, mix):
        """Chip losses [n, NCHIP] for a pattern mix {name: weight} (weights sum to one)."""
        i0, m0, wi, wm = self._weights(I, m)
        s = np.where(sign >= 0, 0, 1)
        out = np.zeros((len(I), self.P.shape[-1]))
        for name, w in mix.items():
            if w == 0:
                continue
            T = self.P[self.patterns.index(name)]
            v = ((1 - wi) * (1 - wm))[:, None] * T[s, i0, m0] + (wi * (1 - wm))[:, None] * T[s, i0 + 1, m0] \
                + ((1 - wi) * wm)[:, None] * T[s, i0, m0 + 1] + (wi * wm)[:, None] * T[s, i0 + 1, m0 + 1]
            out += w * v
        return np.where((I > 0)[:, None], out, 0.0)

    def ripple(self, I, m, sign, fe, mix):
        """Peak-to-peak junction ripple [n, NCHIP] at the output frequency (0 where fe is below the table)."""
        i0, m0, wi, wm = self._weights(I, m)
        s = np.where(sign >= 0, 0, 1)
        lf = np.log(np.maximum(fe, 1e-9))
        ff = np.clip(np.interp(lf, self.lnfe, np.arange(len(self.FE))), 0, len(self.FE) - 1 - 1e-9)
        f0 = ff.astype(int)
        wf = ff - f0
        out = np.zeros((len(I), self.R.shape[-1]))
        for name, w in mix.items():
            if w == 0:
                continue
            T = self.R[self.patterns.index(name)]
            acc = 0.0
            for di, a in ((0, 1 - wi), (1, wi)):
                for dm, b in ((0, 1 - wm), (1, wm)):
                    for df, c in ((0, 1 - wf), (1, wf)):
                        acc = acc + (a * b * c)[:, None] * T[s, i0 + di, m0 + dm, f0 + df]
            out += w * acc
        ok = (I > 0) & (fe >= self.FE[0])
        return np.where(ok[:, None], out, 0.0)


def first_order(p, R, tau, dt):
    """Exact response of R/(1 + s tau) to losses held constant over each step, starting from zero."""
    a = math.exp(-dt / tau)
    return lfilter([R * (1 - a)], [1.0, -a], p, axis=0)


def junction_rise(p, dt, tables, scale=1.0):
    """Junction temperature above ambient Theta[n, NCHIP] for chip losses p[n, NCHIP] (one leg; the other two legs carry
    the same mean losses and load the common water loop). scale multiplies every thermal resistance."""
    th = np.zeros_like(p)
    for Rn, tau in zip(tables_foster_R(), tables_foster_tau()):
        th += first_order(p, 1.0, tau, dt) * (tables.rth * Rn)[None, :]
    pm = np.stack([p[:, tables.mod_of == j].sum(1) for j in range(int(tables.mod_of.max()) + 1)], 1)
    tm = sum(first_order(pm, R, tau, dt) for R, tau in MODULE_BRANCHES)
    tw = first_order(N_LEGS * p.sum(1), tables.r_rad, TAU_RAD, dt)
    return scale * (th + tm[:, tables.mod_of] + tw[:, None])


def tables_foster_R():
    import linh_kien_hv as lk
    return lk.HV_FOSTER_R


def tables_foster_tau():
    import linh_kien_hv as lk
    return lk.HV_FOSTER_TAU
