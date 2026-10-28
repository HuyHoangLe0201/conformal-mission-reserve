"""P3 case study (G7): the information loop of Proposition P3.26 on the full system, with the load shift of thermal
balancing (Remark P3.27).

Per module (3.3 kV converter, 18/10/2026; the 1.7 kV runs, with one bound per chip of a dual module, are kept in
luu_1700V_20261018/): the certified devices are the outer and the inner single-switch module of the leg
(thu_nghiem_g5.MODULES), each with its own strength, Miner factor, true thermal scale and thermal bound; the chips of a
module share its strength and its thermal scale.

Thermal bound per module instead of the stated upper scale 1.10 of G5/G6. The modelled junction rise of the chips of
module i is scaled by an unknown factor s_i (true value s_i,true, drawn per module from the scenario range and rising
with the true damage of the module, by RHO per unit, as its thermal path degrades). A recursive weighted least-squares
estimate with forgetting LAM per interval gives
    F_i <- LAM F_i + I_i,   S_i <- LAM S_i + I_i y_i,   y_i ~ N(s_i,true, 1/I_i),   s_hat = S_i / F_i,
and the certificate uses the upper bound s_up,i = s_hat_i + Z / sqrt(F_i). The information of an interval comes from
  operation  a * F_OP * e_i / e_base,i, where e_i is the energy sum n (Delta Theta)^2 of the slow junction cycles of the
             chips of module i in the interval (what a temperature-sensitive electrical parameter would see) and
             e_base,i its mean under the base pattern; F_OP is set so that operation alone keeps the bound width at
             OMEGA_OP = 0.10, the stated bound of G5/G6;
  pulses     IOTA_P per test pulse in the dwell windows (DeltaTheta 30 K, 2 s, through both IGBTs of the half-leg, so that
             a pulse informs both modules); N_MAX pulses alone keep the bound width at OMEGA_PULSE = 0.05. The pulses
             are cycles of the lifetime model: their margined damage enters V and their true damage the truth, for the
             IGBT chips.
Controller at each decision (policy scores and recorded intervals, as in experiment C of G6):
  balancing  pattern 'base', 'in' (relieves the inner IGBTs) or 'out' (relieves the outer IGBTs) that minimizes the
             larger certificate of the two modules for the coming interval, i.e. moves loss away from the module with
             the highest certified damage; or always 'base';
  pulses     'none'; 'gated': N_MAX scaled by the certified reserve R (the smaller of the two modules; full above R_HI,
             none below R_LO intervals), the cautious rule that saves life when the reserve is low; 'gated_plan': the
             same with the thresholds R_HI_PLAN and R_LO_PLAN of the planned replacement (test only while the reserve
             covers the plan); the suffix '_min' keeps at least N_MIN pulses per interval (dwell-window calibration,
             item (iv)); 'always': N_MAX;
  replacement of the modules of the leg when no pattern passes for both modules (C <= 1), or when a bound exceeds
             S_CAP.
Coverage is recorded per decision and module. The configurations of one repetition share the world, bench, recorded
and field days and the estimator noise. Phase map of Proposition P3.26 (Figure F7): Phi(F) = LAM F + I(F) for the
information of each module at the start of life and at mid-life, with the bound s_up = 1 + Z / sqrt(F) on both.

Run: python -I thu_nghiem_g7.py <tag> [workers]   (tags in CONFIGS; resumable, one line per repetition in g7_<tag>.jsonl)
"""
import json
import math
import os
import platform
import site
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
if hasattr(platform, "_wmi_query"):
    # Python 3.12 queries WMI when scipy is imported; the WMI client crashed a run under load (G6, 16/10/2026)
    def _no_wmi(*_a):
        raise OSError("WMI queries disabled")
    platform._wmi_query = _no_wmi

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

import chung_chi as cc  # noqa: E402
import thu_nghiem_g5 as g5  # noqa: E402
import thu_nghiem_g6 as g6  # noqa: E402
import tuoi_tho as tt  # noqa: E402

BASE = tt.Margin(stat="delta")
PATTERNS = ("base", "in", "out")
S_GRID = np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.45, 1.6, 1.8, 2.0, 2.4, 3.0])
LOG_S = np.log(S_GRID)
Z = 1.645                       # one-sided 0.95 bound on the thermal scale
LAM = 0.8                       # forgetting per 30-day interval (the thermal path drifts with degradation)
OMEGA_OP = 0.10                 # bound width kept by operation alone at base load (stated bound of G5/G6)
OMEGA_PULSE = 0.05              # bound width kept by N_MAX pulses alone
N_MAX, N_MIN = 300, 60          # test pulses per interval: 10 a day, or 2 a day in the turnaround
R_HI, R_LO = 30.0, 10.0         # reserve (intervals) above which pulses run at full rate, below which they stop
R_HI_PLAN, R_LO_PLAN = 60.0, 30.0   # a more cautious rule: test only while the reserve covers the plan of 60 intervals
RHO = 0.2                       # true thermal scale rises by RHO per unit of true damage
S_CAP = 3.0                     # a bound wider than this makes the module uncertifiable


def rates(lam):
    """Information per interval of base operation and per pulse, for the calibration widths at forgetting lam."""
    return (Z / OMEGA_OP) ** 2 * (1 - lam), (Z / OMEGA_PULSE) ** 2 * (1 - lam) / N_MAX


F_OP, IOTA_P = rates(LAM)
F0 = (Z / OMEGA_OP) ** 2        # commissioning information: bound width 0.10
PULSE = dict(dTheta=30.0, theta_min=15.0, ton=2.0, T_amb=15.0)
T_MAX = 240

# Configurations of the lives (run on the same world in every repetition)
LIVES = {
    "stated": dict(stated=True, a=0.0, pulses="none", balance=False),
    "stated_bal": dict(stated=True, a=0.0, pulses="none", balance=True),
    "op": dict(stated=False, a=1.0, pulses="none", balance=False),
    "op_bal": dict(stated=False, a=1.0, pulses="none", balance=True),
    "pulse_gated": dict(stated=False, a=0.0, pulses="gated", balance=True),
    "pulse_gated_min": dict(stated=False, a=0.0, pulses="gated_min", balance=True),
    "pulse_gated_plan": dict(stated=False, a=0.0, pulses="gated_plan", balance=True),
    "pulse_gated_plan_min": dict(stated=False, a=0.0, pulses="gated_plan_min", balance=True),
    "pulse_always": dict(stated=False, a=0.0, pulses="always", balance=True),
}
PHASE = {   # configurations of the phase map
    "pulse_gated": dict(a=0.0, pulses="gated", balance=True),
    "pulse_gated_min": dict(a=0.0, pulses="gated_min", balance=True),
    "pulse_gated_plan": dict(a=0.0, pulses="gated_plan", balance=True),
    "pulse_gated_plan_min": dict(a=0.0, pulses="gated_plan_min", balance=True),
    "op": dict(a=1.0, pulses="none", balance=False),
    "op_bal": dict(a=1.0, pulses="none", balance=True),
}
F_GRID = np.geomspace(5.0, 5000.0, 31)

CONFIGS = {
    "L_evidence": dict(scen="evidence", n1=20, n2=50, alpha=0.1, reps=100, seed=711),
    "L_boundary": dict(scen="boundary", n1=20, n2=50, alpha=0.1, reps=100, seed=712),
    # forgetting sensitivity (rates recalibrated to the same bound widths); seed of L_evidence: paired with its first
    # 30 repetitions
    "S_lam06": dict(scen="evidence", n1=20, n2=50, alpha=0.1, reps=30, seed=711, lam=0.6),
    "S_lam09": dict(scen="evidence", n1=20, n2=50, alpha=0.1, reps=30, seed=711, lam=0.9),
    # confidence of the thermal bound (information rates kept), drift of the thermal path, minimum dwell pulses
    "S_z128": dict(scen="evidence", n1=20, n2=50, alpha=0.1, reps=30, seed=711, z=1.282),
    "S_z233": dict(scen="evidence", n1=20, n2=50, alpha=0.1, reps=30, seed=711, z=2.326),
    "S_rho0": dict(scen="boundary", n1=20, n2=50, alpha=0.1, reps=30, seed=712, rho=0.0),
    "S_rho04": dict(scen="boundary", n1=20, n2=50, alpha=0.1, reps=30, seed=712, rho=0.4),
    "S_nmin20": dict(scen="evidence", n1=20, n2=50, alpha=0.1, reps=30, seed=711, n_min=20),
    "S_nmin120": dict(scen="evidence", n1=20, n2=50, alpha=0.1, reps=30, seed=711, n_min=120),
}

_W = {}


def excitation(npz, chip_names, groups):
    """Energy sum n (Delta Theta)^2 of the slow junction cycles (kind 0) per vehicle day and module (over its chips),
    in the day order of g5.DayPool."""
    d = np.load(HERE / npz)
    cyc, days = d["cycles"], d["day"]
    names = [str(x) for x in d["names"]]
    lut = np.full(int(days.max()) + 1, -1)
    lut[days] = np.arange(len(days))
    out = np.zeros((len(days), len(chip_names)))
    slow = cyc[:, 6] == 0
    for j, c in enumerate(chip_names):
        s = cyc[slow & (cyc[:, 1] == names.index(c))]
        np.add.at(out[:, j], lut[s[:, 0].astype(int)], s[:, 5].astype(float) * s[:, 2].astype(float) ** 2)
    return np.stack([out[:, g].sum(1) for g in groups], 1)


def _init():
    try:
        h = tt.hernes()
        _W["b_full"] = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])["b"]
        _W["pools"] = {p: g5.DayPool(f"chu_ky_cruise_d00_{p}.npz", BASE) for p in PATTERNS}
        groups = _W["pools"]["base"].groups
        _W["exc"] = {p: excitation(f"chu_ky_cruise_d00_{p}.npz", g5.CHIPS, groups) for p in PATTERNS}
        _W["exc_base"] = _W["exc"]["base"].mean(0) * g5.INTERVAL_DAYS
        _W["igbt"] = np.array([c.startswith("T") for c in g5.CHIPS], float)     # chips that the test pulses heat
    except Exception as e:  # noqa: BLE001
        _W.clear()
        _W["error"] = f"worker initialization failed: {e!r}"


def model_chip(pool, idx, fitd, s, j):
    rw = pool.raw[j]
    cyc = {k: np.concatenate([rw[i][k] for i in idx]) for k in ("dTheta", "theta_min", "ton", "n", "T_amb")}
    V, la = tt.prep(cyc, BASE, s)
    return tt.damage_from_prep(*tt.compress(V, la), fitd, BASE)


def pulse_cycles(n):
    return {k: np.array([v], float) for k, v in dict(PULSE, n=float(n)).items()}


def interp(logv, s):
    """Interpolate log V linearly in log s on S_GRID (linear extrapolation outside); logv [..., len(S_GRID)]."""
    x = math.log(s)
    k = int(np.clip(np.searchsorted(LOG_S, x) - 1, 0, len(LOG_S) - 2))
    w = (x - LOG_S[k]) / (LOG_S[k + 1] - LOG_S[k])
    return np.exp(logv[..., k] * (1 - w) + logv[..., k + 1] * w)


def pulses(rule, R, n_min=N_MIN):
    if rule == "none":
        return 0
    if rule == "always":
        return N_MAX
    hi, lo = (R_HI_PLAN, R_LO_PLAN) if rule.startswith("gated_plan") else (R_HI, R_LO)
    g = int(round(N_MAX * min(max((R - lo) / (hi - lo), 0.0), 1.0)))
    return max(g, n_min) if rule.endswith("_min") else g


def v_at(VG, p, s_chip):
    """Margined damage of the recorded intervals under pattern p with the bound of each chip's module: [m, chips]."""
    return np.stack([interp(VG[p][:, j, :], s_chip[j]) for j in range(len(s_chip))], 1)


def pulse_v(VPG, s_chip, n):
    """Margined damage of n test pulses per chip (IGBT chips only), at the bound of each chip's module."""
    return n * _W["igbt"] * np.array([float(interp(VPG, s)) for s in s_chip])


def decide(cfg, VG, VPG, S_pol, D0, s_chip, n_plan, alpha, groups):
    """Pattern (balancing), the larger certificate of the modules and the smaller reserve on the policy data; None if no
    pattern passes for both modules."""
    best = None
    for p in (PATTERNS if cfg["balance"] else ("base",)):
        V = v_at(VG, p, s_chip) + pulse_v(VPG, s_chip, n_plan)[None, :]
        res = [g5.reserve("paired", S_pol, V[:, g], D0[g], alpha) for g in groups]
        C, R = max(x[0] for x in res), min(x[1] for x in res)
        if C <= 1 and (best is None or C < best[1]):
            best = (p, C, R)
    return best


def life(cfg, W, rng_est):
    pools, exc, exc_base = _W["pools"], _W["exc"], _W["exc_base"]
    pb = pools["base"]
    K, groups, mod_of, nc = pb.K, pb.groups, pb.mod_of, len(pb.chips)
    world, fitd, S_new, kap, s0 = W["world"], W["fitd"], W["S_new"], W["kap"], W["s0"]
    alpha = W["alpha"]
    F = np.full(K, F0)
    Ssum = F * (s0 + rng_est.standard_normal(K) / math.sqrt(F0))
    D0, Dt = np.zeros(nc), np.zeros(nc)
    n_prev = N_MAX if cfg["pulses"] in ("always", "gated", "gated_min") else 0
    rec = dict(t=[], pattern=[], n_p=[], bits=[], th_cov=[], omega=[], s_up=[], s_true=[], C=[], R=[])
    failed, collapse, pulse_model, pulse_true = False, False, 0.0, 0.0
    for t in range(T_MAX):
        if cfg["stated"]:
            s_up = np.full(K, g5.S_UP)
        else:
            s_up = Ssum / F + W["z"] / np.sqrt(F)
            if s_up.max() > S_CAP:
                collapse = True
                break
        s_true = s0 * (1 + W["rho"] * pb.mod(Dt))
        s_chip, st_chip = s_up[mod_of], s_true[mod_of]
        best = decide(cfg, W["VG1"], W["VPG"], W["S_pol"], D0, s_chip, n_prev, alpha, groups)
        if best is None:
            break
        p, Cp, R = best
        n_p = pulses(cfg["pulses"], R, W["n_min"])
        Vc = v_at(W["VG2"], p, s_chip) + pulse_v(W["VPG"], s_chip, n_p)[None, :]
        m2 = min(len(W["S_cer"]), len(Vc))
        Cc = np.array([cc.u6_paired(W["S_cer"][:m2], Vc[:m2][:, g], D0[g], alpha) for g in groups])
        q = W["days"][t]
        vp = pulse_v(W["VPG"], s_chip, n_p)
        Vt = np.array([model_chip(pools[p], q, fitd, s_chip[j], j) for j in range(nc)]) + vp
        tp = np.array([world.true_damage(pulse_cycles(n_p), S_new[mod_of[j]], kap[mod_of[j]], st_chip[j])
                       * _W["igbt"][j] if n_p else 0.0 for j in range(nc)])
        Dn = pools[p].true(q, world, S_new, kap, s_true) + tp
        cov = pb.mod(Dt + Dn) <= Cc
        rec["t"].append(t)
        rec["pattern"].append(p)
        rec["n_p"].append(n_p)
        rec["bits"].append(int(sum(1 << j for j in range(K) if cov[j])))
        rec["th_cov"].append([bool(s_up[j] >= s_true[j]) for j in range(K)])
        rec["omega"].append([g5.S_UP - 1.0] * K if cfg["stated"] else [float(x) for x in (W["z"] / np.sqrt(F))])
        rec["s_up"].append([float(x) for x in s_up])
        rec["s_true"].append([float(x) for x in s_true])
        rec["C"].append(float(Cp))
        rec["R"].append(float(R))
        e_op = np.array([exc[p][q, j].sum() / exc_base[j] for j in range(K)])
        info = cfg["a"] * W["f_op"] * e_op + W["iota"] * n_p
        y = s_true + rng_est.standard_normal(K) / np.sqrt(np.maximum(info, 1e-12))
        Ssum = W["lam"] * Ssum + info * y
        F = W["lam"] * F + info
        D0, Dt = D0 + Vt, Dt + Dn
        pulse_model += float(vp.max())
        pulse_true += float(tp.max())
        n_prev = n_p
        if Dt.max() >= 1:
            failed = True
            break
    n = len(rec["t"])
    th = np.array(rec["th_cov"]) if n else np.zeros((0, K))
    om = np.array(rec["omega"]) if n else np.zeros((0, K))
    cv = np.array([[(b >> j) & 1 for j in range(K)] for b in rec["bits"]], float) if n else np.zeros((0, K))
    pats = rec["pattern"]
    return dict(intervals=n, failed=failed, collapse=collapse, harvest=float(Dt.max()), D0=float(D0.max()),
                coverage=float(cv.mean()) if n else None, cov_all=float(cv.min(1).mean()) if n else None,
                cov_life=bool(cv.min()) if n else None, bits=rec["bits"],
                th_cov=[float(x) for x in th.mean(0)] if n else None,
                omega_mean=[float(x) for x in om.mean(0)] if n else None,
                omega_end=[float(x) for x in om[-1]] if n else None,
                omega_max=[float(x) for x in om.max(0)] if n else None,
                share={p: float(np.mean([x == p for x in pats])) if n else 0.0 for p in PATTERNS},
                pulses=int(np.sum(rec["n_p"])), pulse_model_share=pulse_model / max(float(D0.max()), 1e-12),
                pulse_true_share=pulse_true / max(float(Dt.max()), 1e-12),
                trace=dict(omega=[[round(v, 4) for v in o] for o in rec["omega"]], n_p=rec["n_p"], pattern=pats,
                           R=[round(v, 2) for v in rec["R"]]))


def phase(W, D0):
    """Phi(F) for both modules with the bound 1 + Z / sqrt(F) on both, at accumulated model damage D0: rows of
    [Phi outer, Phi inner, pulses, reserve]."""
    pb = _W["pools"]["base"]
    out = {}
    for name, cfg in PHASE.items():
        rows = []
        n_plan = N_MAX if cfg["pulses"] != "none" else 0
        for Fv in F_GRID:
            s_up = np.full(pb.K, 1.0 + W["z"] / math.sqrt(Fv))
            if s_up.max() > S_CAP:
                rows.append([W["lam"] * Fv] * pb.K + [0.0, -1.0])
                continue
            best = decide(dict(cfg, stated=False), W["VG1"], W["VPG"], W["S_pol"], D0, s_up[pb.mod_of], n_plan,
                          W["alpha"], pb.groups)
            if best is None:
                rows.append([W["lam"] * Fv] * pb.K + [0.0, -1.0])
                continue
            p, _, R = best
            n_p = pulses(cfg["pulses"], R, W["n_min"])
            e_op = _W["exc"][p].sum(0) / _W["exc"]["base"].sum(0)
            info = cfg["a"] * W["f_op"] * e_op + W["iota"] * n_p
            rows.append([float(W["lam"] * Fv + info[k]) for k in range(pb.K)] + [float(n_p), float(R)])
        out[name] = rows
    return out


def rep(args):
    cfg, r = args
    if "error" in _W:
        raise RuntimeError(_W["error"])
    t0 = time.time()
    rng = np.random.default_rng([cfg["seed"], r])
    sc = g6.scenario(cfg["scen"])
    pools = _W["pools"]
    pb = pools["base"]
    world = tt.World.draw(rng, _W["b_full"], sc)
    n1, n2, alpha = cfg["n1"], cfg["n2"], cfg["alpha"]
    bn, fi, si, fitd, X, S_all = g6.bench_scores(world, rng, 3 * n1)
    perm = rng.permutation(len(si))
    S_pol, S_cer = S_all[perm[:len(si) // 2]], S_all[perm[len(si) // 2:]]
    J1 = [pb.sample(rng) for _ in range(n2)]
    J2 = [pb.sample(rng) for _ in range(n2)]
    S_new = world.strengths(rng, pb.K)
    kap = rng.uniform(*sc.kappa, size=pb.K)
    s0 = rng.uniform(*sc.s_th, size=pb.K)            # true thermal scale per module at commissioning
    days = [pb.sample(rng) for _ in range(T_MAX)]
    m1, m2 = min(len(S_pol), n2), min(len(S_cer), n2)
    VG1, VG2 = {}, {}
    grid = lambda J: np.log(np.maximum(np.array([[g6.model_raw(pools[p], q, fitd, BASE, s=s) for s in S_GRID]
                                                  for q in J]), 1e-300)).transpose(0, 2, 1)
    for p in PATTERNS:
        VG1[p], VG2[p] = grid(J1[:m1]), grid(J2[:m2])
    VPG = np.log([tt.damage_from_prep(*tt.prep(pulse_cycles(1), BASE, s), fitd, BASE) for s in S_GRID])
    lam = cfg.get("lam", LAM)
    f_op, iota = rates(lam)
    W = dict(world=world, fitd=fitd, S_new=S_new, kap=kap, s0=s0, alpha=alpha, S_pol=S_pol[:m1], S_cer=S_cer[:m2],
             VG1=VG1, VG2=VG2, VPG=VPG, days=days, lam=lam, f_op=f_op, iota=iota, z=cfg.get("z", Z),
             rho=cfg.get("rho", RHO), n_min=cfg.get("n_min", N_MIN))
    t_pre = time.time() - t0
    lives = {}
    for name, lc in LIVES.items():
        lives[name] = life(lc, W, np.random.default_rng([cfg["seed"], r, 99]))
    D_mid = 15 * v_at(VG1, "base", np.full(len(pb.chips), g5.S_UP)).mean(0)     # about mid-life under the stated bound
    ph = {"start": phase(W, np.zeros(len(pb.chips))), "mid": phase(W, D_mid)}
    return dict(rep=r, s0=[float(x) for x in s0], S_new=[float(x) for x in S_new], kappa=[float(x) for x in kap],
                lives=lives, phase=ph, pulse_damage_model=float(interp(VPG, g5.S_UP)),
                seconds=[round(t_pre), round(time.time() - t0)])


def run(tag, workers):
    cfg = CONFIGS[tag]
    out = HERE / f"g7_{tag}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["rep"])
    todo = [r for r in range(cfg["reps"]) if r not in done]
    if not todo:
        print(f"{tag}: {len(done)} done", flush=True)
        return
    free = g6.wait_memory(2.0 + 1.0 * workers, tag)
    if free is not None:
        workers = max(1, min(workers, int((free - 2.0) / 1.0)))
    print(f"{tag}: {len(done)} done, {len(todo)} to run with {workers} workers, {time.strftime('%d/%m %H:%M')}",
          flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init) as ex, open(out, "a", encoding="utf-8", newline="\n") as f:
        futs = [ex.submit(rep, (cfg, r)) for r in todo]
        for k, fu in enumerate(as_completed(futs)):
            f.write(json.dumps(fu.result()) + "\n")
            f.flush()
            print(f"  {tag}: {k + 1}/{len(todo)}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv
    run(a[1], int(a[2]) if len(a) > 2 else 4)
