"""P3 case study (G4): train dynamics, driving strategy and inverter operating point for one run between two stops.

Train: the programme's 8-car EMU assumption, with the same values as the P4 simulation (P4_mophong/README.md) and the
650 kW model of the project (CoSoToanHoc.tex). It is coded again here so that P3 stands alone. Every value is an
assumption typical of a 350 km/h EMU with 16 PMSM of 650 kW (a CR400-class unit: 8 cars, 16 motors of 625-650 kW),
fed in 1C2M configuration (one inverter for the two motors of a bogie, eight inverters; until 17/10/2026 the model had
1C1M at 1800 V for the 1.7 kV design): mass 480 t, rotating-mass factor 1.06, starting acceleration 0.5 m/s^2, service braking 0.5 m/s^2,
Davis resistance 6.4 + 0.044 v + 0.00055 v^2 kN (v in km/h).

Driving strategies. The timetables carry slack, so a run can be driven in many ways; two bracket the inverter load.
  'cruise'  full tractive effort up to a cruise speed v_c, constant v_c, service braking to a stop at the next station;
            v_c is the lowest speed that meets the run time. Shortest time at full current, lowest cruise power.
  'coast'   full effort up to the line speed, cruise at the line speed, coast, service braking; the coasting point is
            set to meet the run time (when coasting from the line speed straight after accelerating is still too fast,
            the top speed is lowered). Longest time at full current, highest cruise power, and a coasting phase without
            current inside the run.
Energy-efficient driving (accelerate, cruise, coast, brake) lies between the two. A run that needs more than the line
speed is flagged by the caller and driven at the line speed with either strategy.

Electrical point of one inverter (the P4 formulas): torque per motor from the wheel force, voltage proportional to
speed below the base speed and at its limit above (field weakening), current from the electrical power; the inverter
current is that of its N_PAR motors in parallel. DC link 3000 V for the 3.3 kV modules of ton_hao.py.
"""
import math
import site
import sys

sys.path.append(site.getusersitepackages())  # keeps the module usable with python -I
import numpy as np  # noqa: E402

P = dict(
    n_motor=16,            # PMSM of the train
    n_par=2,               # motors per inverter (1C2M)
    P_rated=650e3,         # W, electrical, per motor
    M=480e3,               # kg, loaded train mass
    xi=1.06,               # rotating-mass factor
    a0=0.5,                # m/s^2, starting acceleration (adhesion and comfort limit)
    b_service=0.5,         # m/s^2, service deceleration
    davis=(6.4e3, 0.044e3, 0.00055e3),  # R = A + B v + C v^2, N with v in km/h
    v_line=350 / 3.6,      # m/s, line speed (Beijing-Shanghai high-speed line)
    r_w=0.43,              # m, wheel radius
    G=2.7,                 # gear ratio
    pole_pairs=3,
    eta_m=0.96, eta_g=0.975,
    Vdc=3000.0, m_rated=0.95, cosphi=0.95,
    v_fade=7 / 3.6,        # m/s, electric braking fades out below this speed
    dt=0.5,                # s, output time step
)


# rated peak current of one inverter: N_PAR motors at rated power, rated modulation and power factor (554 A)
P["I_rated"] = P["n_par"] * P["P_rated"] / (1.5 * P["m_rated"] * P["Vdc"] / math.sqrt(3) * P["cosphi"])


def resistance(v, p=P):
    A, B, C = p["davis"]
    vk = 3.6 * np.asarray(v, dtype=float)
    return A + B * vk + C * vk ** 2


def limits(p=P):
    """Wheel power limit, maximum tractive force and base speed."""
    eta = p["eta_m"] * p["eta_g"]
    Pw_max = p["n_motor"] * p["P_rated"] * eta
    F_max = p["xi"] * p["M"] * p["a0"] + float(resistance(0.0, p))
    return Pw_max, F_max, Pw_max / F_max


def f_trac(v, p=P):
    Pw_max, F_max, _ = limits(p)
    return np.minimum(F_max, Pw_max / np.maximum(v, 1e-6))


def top_speed(p=P):
    """Speed at which the full tractive force equals the running resistance."""
    lo, hi = 1.0, 200.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(f_trac(mid, p)) > float(resistance(mid, p)):
            lo = mid
        else:
            hi = mid
    return lo


def derated(d, p=P):
    """Parameters of a drive derated by the fraction d: current limit, hence tractive force and power, scaled by
    1 - d at equal voltage. The line speed is capped at 98 % of the top speed when derating makes it unreachable."""
    q = dict(p)
    q["P_rated"] = p["P_rated"] * (1 - d)
    _, F_max, _ = limits(p)
    q["a0"] = ((1 - d) * F_max - float(resistance(0.0, p))) / (p["xi"] * p["M"])
    q["v_line"] = min(p["v_line"], 0.98 * top_speed(q))
    return q


class Acceleration:
    """Full-effort acceleration from standstill to the line speed, integrated once on a fine grid, so that time and
    distance to reach any speed come from interpolation."""

    def __init__(self, p=P, h=0.01):
        v_end = p["v_line"]
        vs, ts, xs = [0.0], [0.0], [0.0]
        v = t = x = 0.0
        mass = p["xi"] * p["M"]
        while v < v_end - 1e-9:
            a = (float(f_trac(v, p)) - float(resistance(v, p))) / mass
            if a <= 1e-6:
                raise ValueError(f"line speed {3.6 * v_end:.0f} km/h is not reachable: acceleration vanishes at "
                                 f"{3.6 * v:.1f} km/h")
            dv = min(h * a, v_end - v)
            dt_ = dv / a
            x += (v + 0.5 * dv) * dt_
            v += dv
            t += dt_
            vs.append(v)
            ts.append(t)
            xs.append(x)
        self.v, self.t, self.x = np.array(vs), np.array(ts), np.array(xs)

    def t_of(self, v):
        return np.interp(v, self.v, self.t)

    def x_of(self, v):
        return np.interp(v, self.v, self.x)

    def v_of_t(self, t):
        return np.interp(t, self.t, self.v)


class Coasting:
    """Coasting (no tractive or braking force) from the line speed down to 0.5 m/s, integrated once in speed steps. The
    equation is autonomous, so coasting between any two speeds is a difference of the stored time and distance."""

    def __init__(self, p=P, h=0.01, v_end=0.5):
        v, t, x = p["v_line"], 0.0, 0.0
        vs, ts, xs = [v], [t], [x]
        mass = p["xi"] * p["M"]
        while v > v_end + 1e-12:
            a = float(resistance(v, p)) / mass
            dv = min(h, v - v_end)
            dt_ = dv / a
            x += (v - 0.5 * dv) * dt_
            v -= dv
            t += dt_
            vs.append(v)
            ts.append(t)
            xs.append(x)
        self.v, self.t, self.x = np.array(vs[::-1]), np.array(ts[::-1]), np.array(xs[::-1])

    def t_from(self, v):
        return np.interp(v, self.v, self.t)

    def x_from(self, v):
        return np.interp(v, self.v, self.x)

    def v_after(self, v1, tau):
        """Speed after coasting for tau seconds from speed v1."""
        return np.interp(self.t_from(v1) + tau, self.t[::-1], self.v[::-1])


def run_time_coast(L, v_top, v_b, acc, coast, p=P):
    """Run time with full effort to v_top, cruise, coasting from v_top to v_b, braking; None if the distance is short."""
    b = p["b_service"]
    x_a, t_a = acc.x_of(v_top), acc.t_of(v_top)
    x_c = coast.x_from(v_b) - coast.x_from(v_top)
    t_c = coast.t_from(v_b) - coast.t_from(v_top)
    x_cr = L - x_a - x_c - v_b ** 2 / (2 * b)
    if x_cr < -1e-6:
        return None
    return t_a + max(x_cr, 0.0) / v_top + t_c + v_b / b


def coast_plan(L, T, acc, coast, p=P):
    """Top speed and braking speed of the coasting strategy for a run of length L (m) and time T (s)."""
    b = p["b_service"]
    t_fast, v_pk = run_time(L, p["v_line"], acc, p)
    if t_fast >= T:
        return v_pk, v_pk, t_fast <= T + 1e-9

    def vb_min(v_top):
        # lowest braking speed, reached when coasting starts right after accelerating; the distance of coasting plus
        # braking grows as the braking speed falls, because coasting decelerates more slowly than braking
        def dist(v_b):
            return acc.x_of(v_top) + coast.x_from(v_b) - coast.x_from(v_top) + v_b ** 2 / (2 * b)
        lo, hi = 0.5, v_top
        if dist(lo) <= L:
            return lo
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if dist(mid) > L:
                lo = mid
            else:
                hi = mid
        return hi

    v_top = v_pk
    v_lo = vb_min(v_top)
    if run_time_coast(L, v_top, v_lo, acc, coast, p) >= T:
        lo, hi = v_lo, v_top          # bisection on the braking speed
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if run_time_coast(L, v_top, mid, acc, coast, p) > T:
                lo = mid
            else:
                hi = mid
        return v_top, hi, True
    lo, hi = 0.5, v_pk                # coasting from the line speed is still too fast: lower the top speed
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        tt = run_time_coast(L, mid, vb_min(mid), acc, coast, p)
        if tt is None or tt > T:
            lo = mid
        else:
            hi = mid
    return hi, vb_min(hi), True


def run_time(L, v_c, acc, p=P):
    """Run time of a stop-to-stop run of length L (m) with cruise speed v_c, and the peak speed actually reached."""
    b = p["b_service"]
    x_a, t_a = acc.x_of(v_c), acc.t_of(v_c)
    x_b = v_c ** 2 / (2 * b)
    if x_a + x_b <= L:
        return t_a + v_c / b + (L - x_a - x_b) / v_c, v_c
    # the train brakes before reaching v_c: peak speed where acceleration and braking distances fill the run
    lo, hi = 0.0, v_c
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if acc.x_of(mid) + mid ** 2 / (2 * b) > L:
            hi = mid
        else:
            lo = mid
    v_p = lo
    return acc.t_of(v_p) + v_p / b, v_p


def cruise_speed(L, T, acc, p=P):
    """Lowest cruise speed that meets the run time T (s). Returns (v_c, feasible)."""
    t_fast, _ = run_time(L, p["v_line"], acc, p)
    if t_fast > T:
        return p["v_line"], False
    lo, hi = 1.0, p["v_line"]
    if run_time(L, lo, acc, p)[0] <= T:
        return lo, True
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if run_time(L, mid, acc, p)[0] > T:
            lo = mid
        else:
            hi = mid
    return hi, True


def profile(L, T, acc, p=P, coast=None):
    """Time series of one run: arrays t, v, F_wheel, F_elec, phase (1 accelerate, 2 cruise, 3 brake, 5 coast), plus a
    summary. coast=None gives the 'cruise' strategy, a Coasting object the 'coast' strategy. The run ends at standstill
    at the next station; dwell is added by the caller."""
    b, dt = p["b_service"], p["dt"]
    if coast is None:
        v_c, feasible = cruise_speed(L, T, acc, p)
        T_run, v_top = run_time(L, v_c, acc, p)
        v_b, t_co = v_top, 0.0
    else:
        v_top, v_b, feasible = coast_plan(L, T, acc, coast, p)
        T_run = run_time_coast(L, v_top, v_b, acc, coast, p)
        v_c = v_top
        t_co = float(coast.t_from(v_b) - coast.t_from(v_top))
    t_a = float(acc.t_of(v_top))
    t_b = v_b / b
    t_c = max(0.0, T_run - t_a - t_co - t_b)
    n = int(math.ceil(T_run / dt))
    t = np.arange(n + 1) * dt
    t[-1] = min(t[-1], T_run)
    e1, e2, e3 = t_a, t_a + t_c, t_a + t_c + t_co
    v_coast = coast.v_after(v_top, np.maximum(t - e2, 0.0)) if coast is not None else np.full_like(t, v_top)
    v = np.select([t <= e1, t <= e2, t <= e3], [acc.v_of_t(np.minimum(t, t_a)), np.full_like(t, v_top), v_coast],
                  np.maximum(0.0, v_b - b * (t - e3)))
    phase = np.select([t <= e1, t <= e2, t <= e3], [1, 2, 5], 3)
    R = resistance(v, p)
    mass = p["xi"] * p["M"]
    F = np.select([phase == 1, phase == 2, phase == 5], [f_trac(v, p), R, np.zeros_like(t)], -(mass * b - R))
    # regenerative braking up to the traction limit, fading out at low speed; the rest is friction braking
    F_e = np.where(phase == 3, np.maximum(F, -f_trac(v, p) * np.minimum(1.0, v / p["v_fade"])), F)
    summ = dict(L_km=L / 1e3, T_s=T, T_model_s=T_run, v_c_kmh=3.6 * v_c, v_peak_kmh=3.6 * v_top,
                v_brake_kmh=3.6 * v_b, feasible=feasible, t_accel_s=t_a, t_cruise_s=t_c, t_coast_s=t_co, t_brake_s=t_b)
    return dict(t=t, v=v, F=F, F_e=F_e, phase=phase), summ


def electrical(v, F_e, p=P):
    """Operating point of one inverter from speed and electric wheel force: I_pk (A), m, cos phi (signed: negative in
    regenerative braking), f_e (Hz), P_e (W, negative in braking). p holds the nominal drive: a derated profile is
    converted with the nominal parameters, so that derated torque gives proportionally less current."""
    Pw_max, F_max, v_base = limits(p)
    Vpk_max = p["m_rated"] * p["Vdc"] / math.sqrt(3)
    w_base = v_base / p["r_w"] * p["G"]
    T_rated = Pw_max / p["n_motor"] / p["eta_g"] / w_base
    w_m = v / p["r_w"] * p["G"]
    motoring = F_e >= 0
    T_m = F_e * p["r_w"] / p["n_motor"] / p["G"] * np.where(motoring, 1 / p["eta_g"], p["eta_g"])
    P_mech = T_m * w_m
    P_e = np.where(motoring, P_mech / p["eta_m"], P_mech * p["eta_m"])
    f_e = p["pole_pairs"] * w_m / (2 * math.pi)
    k = np.minimum(1.0, w_m / w_base)
    m = p["m_rated"] * k
    with np.errstate(divide="ignore", invalid="ignore"):
        I_v = p["n_par"] * np.abs(P_e) / (1.5 * Vpk_max * k * p["cosphi"])
    I_0 = p["I_rated"] * np.abs(T_m) / T_rated * np.where(motoring, 1.0, p["eta_m"] ** 2)
    I_pk = np.where(k > 1e-3, I_v, I_0)
    off = np.abs(T_m) < 1e-9
    I_pk = np.where(off, 0.0, I_pk)
    m = np.where(off, 0.0, m)
    cphi = np.where(off, 0.0, p["cosphi"] * np.sign(T_m))
    return dict(I_pk=I_pk, m=m, cosphi=cphi, f_e=f_e, P_e=P_e)


if __name__ == "__main__":
    acc = Acceleration()
    Pw_max, F_max, v_base = limits()
    print(f"F_max {F_max / 1e3:.1f} kN, wheel power {Pw_max / 1e6:.2f} MW, base speed {3.6 * v_base:.0f} km/h; "
          f"0-{3.6 * P['v_line']:.0f} km/h in {acc.t[-1]:.0f} s over {acc.x[-1] / 1e3:.1f} km")
    coast = Coasting()
    print(f"top speed {3.6 * top_speed():.1f} km/h; derated by 20 %: line speed {3.6 * derated(0.2)['v_line']:.1f} km/h")
    for L, T in ((406e3, 82 * 60), (406e3, 75 * 60), (50e3, 15 * 60), (20e3, 12 * 60), (59e3, 4 * 60)):
        for name, c in (("cruise", None), ("coast", coast)):
            prof, s = profile(L, T, acc, coast=c)
            el = electrical(prof["v"], prof["F_e"])
            h = np.diff(prof["t"], append=prof["t"][-1])
            x = float(np.sum(prof["v"] * h)) / 1e3
            print(f"L {s['L_km']:.0f} km, T {T / 60:.0f} min, {name:6}: model {s['T_model_s'] / 60:.2f} min, "
                  f"x {x:.2f} km, peak {s['v_peak_kmh']:.0f} km/h, brake from {s['v_brake_kmh']:.0f}, "
                  f"feasible {s['feasible']}; accel {s['t_accel_s']:.0f} s, cruise {s['t_cruise_s']:.0f} s, "
                  f"coast {s['t_coast_s']:.0f} s, brake {s['t_brake_s']:.0f} s; I_pk max {el['I_pk'].max():.0f} A")
