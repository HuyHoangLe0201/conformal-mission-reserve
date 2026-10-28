"""P3 case study (G5): rainflow counting of a junction-temperature history with the heating time of every cycle.

Counting: turning points with a hysteresis gate (reversals smaller than GATE are dropped), then the three-point
rainflow of ASTM E1049 (full cycles from the stack, half cycles from the residue).

Heating time of a cycle with valley v and peak p: the time from the last upward crossing of the mid level (v + p) / 2
before the peak to the peak, with linear interpolation between samples. For the rectangular power pulse of a
power-cycling test this is the pulse duration minus the short time to reach mid level; for an envelope cycle of a
mission (the day's minimum against its maximum) it is the time spent rising through the upper half of the swing,
not the hours between the morning start and the peak. The duration of the half cycle, used in P4, would charge such a
cycle with a heating time of hours.

Run: python rainflow_ton.py   (self-test on synthetic histories)
"""
import os
import site
import sys

sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

GATE = 0.1   # K
# level of the swing from which the heating time is timed (19/10/2026, sensitivity to the definition): 0.5 is the
# mid level of the case study; P3_TON_LEVEL=0.25 times the rise from a quarter of the swing (longer heating times),
# 0.75 from three quarters (shorter)
LEVEL = float(os.environ.get("P3_TON_LEVEL", "0.5"))


def turning_points(x, gate=GATE):
    """Indices of the turning points of x after a hysteresis gate; keeps the first and last samples."""
    d = np.diff(x)
    nz = np.flatnonzero(d != 0)
    if nz.size == 0:
        return np.array([0, len(x) - 1])
    s = np.sign(d[nz])
    rev = nz[np.flatnonzero(s[1:] != s[:-1])] + 1          # candidate extrema (sample index)
    cand = np.r_[0, rev, len(x) - 1]
    out = [cand[0]]
    direction = 0
    ext = cand[0]
    for i in cand[1:]:
        v = x[i]
        if direction == 0:
            if abs(v - x[out[-1]]) >= gate:
                direction = 1 if v > x[out[-1]] else -1
                ext = i
            continue
        if (direction > 0 and v >= x[ext]) or (direction < 0 and v <= x[ext]):
            ext = i                                          # the current excursion goes on
        elif abs(v - x[ext]) >= gate:
            out.append(ext)                                  # reversal confirmed
            direction = -direction
            ext = i
    out.append(ext)
    if out[-1] != len(x) - 1:
        out.append(len(x) - 1)
    return np.array(sorted(set(out)))


def heating_time(x, dt, i_peak, v_valley, level=None):
    """Time from the last upward crossing of the mid level (or of v + level (p - v)) before i_peak to i_peak."""
    lv = LEVEL if level is None else level
    mid = 0.5 * (x[i_peak] + v_valley) if lv == 0.5 else v_valley + lv * (x[i_peak] - v_valley)
    w = 64
    while True:
        lo = max(0, i_peak - w)
        seg = x[lo:i_peak + 1]
        below = np.flatnonzero(seg <= mid)
        if below.size:
            j = lo + below[-1]
            x0, x1 = x[j], x[j + 1]
            frac = (mid - x0) / (x1 - x0) if x1 != x0 else 0.0
            return (i_peak - j - frac) * dt
        if lo == 0:
            return i_peak * dt
        w *= 4


def rainflow(x, dt, gate=GATE):
    """Cycles of x: arrays (range, valley, heating time, count)."""
    tp = turning_points(x, gate)
    rng, val, ton, cnt = [], [], [], []

    def add(a, b, c):
        lo, hi = (a, b) if x[a] < x[b] else (b, a)
        rng.append(x[hi] - x[lo])
        val.append(x[lo])
        ton.append(heating_time(x, dt, hi, x[lo]))
        cnt.append(c)

    stack = []
    for i in tp:
        stack.append(i)
        while len(stack) >= 3:
            X = abs(x[stack[-1]] - x[stack[-2]])
            Y = abs(x[stack[-2]] - x[stack[-3]])
            if X < Y:
                break
            if len(stack) == 3:
                add(stack[0], stack[1], 0.5)
                stack.pop(0)
            else:
                add(stack[-3], stack[-2], 1.0)
                last = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(last)
    for a, b in zip(stack[:-1], stack[1:]):
        add(a, b, 0.5)
    return np.array(rng), np.array(val), np.array(ton), np.array(cnt)


if __name__ == "__main__":
    dt = 0.5
    # power-cycling test: 20 rectangular pulses of 4.5 s on, 9 s off through tau = 0.5 s
    t = np.arange(0, 20 * 13.5, dt)
    p = ((t % 13.5) < 4.5).astype(float)
    a = np.exp(-dt / 0.5)
    from scipy.signal import lfilter
    x = 40 * lfilter([1 - a], [1, -a], p)
    r, v, ton, c = rainflow(x, dt)
    big = r > 30
    print(f"pulse test: {c.sum():.1f} cycles for 20 pulses (ASTM gives half cycles for a constant-amplitude history), "
          f"range {np.average(r[big], weights=c[big]):.2f} K, heating time {np.average(ton[big], weights=c[big]):.2f} s "
          f"(pulse 4.5 s; mid level reached after tau ln 2 = 0.35 s)")
    # mission-like day: slow envelope, two runs with an acceleration peak and a braking peak
    x = np.r_[np.zeros(10), np.linspace(0, 45, 400), np.full(2000, 35.0), np.linspace(35, 42, 200),
              np.linspace(42, 12, 60), np.full(240, 12.0), np.linspace(12, 50, 400), np.full(1800, 33.0),
              np.linspace(33, 0, 600)]
    r, v, ton, c = rainflow(x, dt)
    for k in np.argsort(-r):
        print(f"  range {r[k]:5.1f} K, valley {v[k]:5.1f}, heating time {ton[k]:7.1f} s, count {c[k]}")
    # hysteresis gate: noise of 0.05 K on a ramp gives no cycle
    x = np.linspace(0, 10, 1000) + 0.05 * np.random.default_rng(1).standard_normal(1000)
    r, v, ton, c = rainflow(x, dt)
    print(f"noisy ramp: {len(r)} cycles, largest {r.max():.2f} K (gate {GATE} K)")
