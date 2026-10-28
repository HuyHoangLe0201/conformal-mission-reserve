"""Device model of the CM1000HC-66R (Mitsubishi Electric, 3300 V / 1000 A single-switch HVIGBT module with
anti-parallel diode, AlSiC base plate; datasheet HVM-1061-B, December 2012), the 3.3 kV / 1 kA class of the bench
modules of Hernes et al. (2021). Datasheet downloaded 17/10/2026 from mitsubishielectric.com into
../Tailieu_P3/datasheet/cm1000hc-66r_e.pdf (read only).

The curves are read from the vector paths of the PDF (pages 4 and 5): solid curves (25 C) point by point along the path,
dashed curves (125 C, 150 C) from the dash centres, separated into the two temperatures by iterative least-squares fits
of V = a + b sqrt(I) + c I + d I^2 (rms residual 0.010 V), switching energies per path. Values marked "table" are the
printed typical values at 1000 A, VCC = 1800 V, against which self_check() compares the curves. Below about 200 A the
on-state curves at 25 C and 125 C cross, where the separation of the dashed curves is least certain; the model uses
the 150 C curves only (ton_hao.py), so the crossing does not enter the losses.
"""
import site
import sys

sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

# ----------------------------------------------------------------------------------- IGBT
# Collector-emitter saturation voltage, VGE = 15 V (p. 4, top right; table p. 2: 2.45 / 3.10 / 3.25 V at 1000 A)
HV_I = np.array([0, 100, 200, 300, 400, 600, 800, 1000, 1200, 1600.0])
HV_VCE = {   # T [degC] -> V
    25:  np.array([0, 1.262, 1.492, 1.654, 1.773, 2.044, 2.256, 2.470, 2.651, 3.030]),
    125: np.array([0, 1.292, 1.622, 1.879, 2.100, 2.482, 2.822, 3.141, 3.448, 4.050]),
    150: np.array([0, 1.359, 1.703, 1.971, 2.201, 2.601, 2.956, 3.290, 3.611, 4.243]),
}
# Free-wheel diode forward voltage, VGE = 0 V (p. 4, bottom right; table p. 3: 2.15 / 2.30 / 2.25 V at 1000 A)
HV_IF = np.array([0, 100, 200, 400, 600, 800, 1000, 1200, 1600.0])
HV_VF = {
    25:  np.array([0, 1.151, 1.339, 1.607, 1.818, 1.991, 2.165, 2.317, 2.615]),
    125: np.array([0, 0.935, 1.171, 1.524, 1.811, 2.067, 2.304, 2.531, 2.962]),
    150: np.array([0, 0.884, 1.122, 1.473, 1.759, 2.014, 2.251, 2.477, 2.914]),
}
# Half-bridge switching energies per pulse [J], VCC = 1800 V, RG(on) 2.4 ohm, RG(off) 8.4 ohm, Ls 150 nH (p. 5; table
# p. 2-3 at 1000 A, IEC 60747 definition: Eon 1.95 / 2.15, Eoff 1.80 / 1.90, Erec 1.35 / 1.55 J at 125 / 150 C)
HV_IE = np.array([0, 100, 200, 400, 600, 800, 1000, 1200, 1600.0])
HV_EON = {125: np.array([0, 0.297, 0.481, 0.822, 1.198, 1.592, 2.037, 2.519, 3.697]),
          150: np.array([0, 0.334, 0.524, 0.897, 1.297, 1.719, 2.167, 2.690, 3.920])}
HV_EOFF = {125: np.array([0, 0.355, 0.557, 0.897, 1.223, 1.519, 1.814, 2.074, 2.630]),
           150: np.array([0, 0.370, 0.592, 0.963, 1.297, 1.608, 1.890, 2.172, 2.741])}
HV_ERR = {125: np.array([0, 0.370, 0.524, 0.814, 1.037, 1.186, 1.297, 1.408, 1.481]),
          150: np.array([0, 0.466, 0.668, 1.001, 1.223, 1.408, 1.519, 1.592, 1.630])}
HV_VREF = 1800.0
# Thermal (table p. 3, maximum values; Foster table p. 7, normalized to Rth(j-c), one curve for IGBT and diode)
HV_RTH_JC_IGBT = 12.0e-3     # K/W
HV_RTH_JC_FWD = 22.5e-3      # K/W
HV_RTH_CS_MODULE = 9.0e-3    # K/W, case to heat sink per module, grease 1 W/(m K), typical
HV_FOSTER_R = np.array([0.0096, 0.1893, 0.4044, 0.3967])   # normalized, sum 1
HV_FOSTER_TAU = np.array([0.0001, 0.0058, 0.0602, 0.3512])  # s

# Voltage scaling exponent of switching energy, E ~ (V / Vref)^KV: an assumption (no curve in the datasheet), as in the
# 1.7 kV model of the project.
KV_HV = 1.0

TABLE = {   # printed typical values at 1000 A (VCC = 1800 V for the energies)
    "VCE": {25: 2.45, 125: 3.10, 150: 3.25}, "VF": {25: 2.15, 125: 2.30, 150: 2.25},
    "Eon": {125: 1.95, 150: 2.15}, "Eoff": {125: 1.80, 150: 1.90}, "Erec": {125: 1.35, 150: 1.55},
}


def _interp_T(table, T):
    """Linear interpolation (and extrapolation) in temperature between tabulated curves."""
    Ts = sorted(table)
    T = float(T)
    if T <= Ts[0]:
        a, b = Ts[0], Ts[1]
    elif T >= Ts[-1]:
        a, b = Ts[-2], Ts[-1]
    else:
        a = max(t for t in Ts if t <= T); b = min(t for t in Ts if t > T)
    w = (T - a) / (b - a)
    return (1 - w) * np.asarray(table[a]) + w * np.asarray(table[b])


def _lookup(x, table, T, i):
    return np.interp(np.abs(i), x, _interp_T(table, T))


def hv_vce(i, T): return _lookup(HV_I, HV_VCE, T, i)
def hv_vf(i, T): return _lookup(HV_IF, HV_VF, T, i)
def hv_eon(i, T, V): return _lookup(HV_IE, HV_EON, max(T, 125), i) * (V / HV_VREF) ** KV_HV
def hv_eoff(i, T, V): return _lookup(HV_IE, HV_EOFF, max(T, 125), i) * (V / HV_VREF) ** KV_HV
def hv_err(i, T, V): return _lookup(HV_IE, HV_ERR, max(T, 125), i) * (V / HV_VREF) ** KV_HV


def self_check():
    """Compare the digitized curves with the printed table values (anchors) at 1000 A."""
    rows = [(f"VCEsat 1000A {T}C", hv_vce(1000, T), v) for T, v in TABLE["VCE"].items()]
    rows += [(f"VEC 1000A {T}C", hv_vf(1000, T), v) for T, v in TABLE["VF"].items()]
    for name, f in (("Eon", hv_eon), ("Eoff", hv_eoff), ("Erec", hv_err)):
        rows += [(f"{name} 1000A {T}C [J]", f(1000, T, HV_VREF), v) for T, v in TABLE[name].items()]
    rows += [("Foster sum", HV_FOSTER_R.sum(), 1.0)]
    return [(n, float(v), ref, 100 * (float(v) - ref) / ref) for n, v, ref in rows]


if __name__ == "__main__":
    print(f"{'anchor':24s} {'model':>8s} {'table':>8s} {'err %':>7s}")
    for n, v, ref, e in self_check():
        print(f"{n:24s} {v:8.3f} {ref:8.3f} {e:7.2f}")
