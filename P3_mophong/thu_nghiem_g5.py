"""P3 case study (G5): experiment driver. E1 measures coverage and tightness of the certificates without feedback.

Pieces:
  DayPool      cycles of one mission database (nhiem_vu.py) per vehicle day and chip: compressed cells of the margined
               model damage under the upper thermal scale S_UP (independent of the fit), and raw cycles for the truth.
  bench        synthetic bench lives on the design of Hernes et al. (2021): conditions drawn from the 34 tested modules
               within each run, lives from the truth of the world (log-linear on the reference set).
  e1           per repetition: a world (tuoi_tho.Scenario), a bench of 2 n1 modules (half fits the model, half are
               scores), n2 recorded intervals of INTERVAL_DAYS vehicle days drawn with replacement, a field device of
               age k_past intervals and its coming interval; certificates U1, U2, U3, U5, U6, U9 against the true
               end-of-interval damage and against the calibrated-part score U_new of the field device (its model score
               at a tested condition times its margined damage: Theorem P3.10 says P(U_new <= C_P) is exact), and the oracle
               quantile U10 of the true damage in that world (Monte Carlo over devices and intervals).

Run: python thu_nghiem_g5.py e1 <npz> <reps> <n1 list> <n2> <alpha> <k_past> [scenario] [stat] -> e1_<...>.json

3.3 kV converter (18/10/2026): the leg is built of single-switch modules (ton_hao.py), so the certified devices are the
outer module (T1 with its diode D1) and the inner module (T2 with D2), each with its own strength; the chips of one
module share its strength (DayPool.groups, DayPool.mod). The clamp-diode module D5 accrues about a third of the damage of
the critical module when cruising and about half when coasting (g5_ngan_sach.py), so it never binds and is not
certified. E1, E2 and E5 are the preliminary runs of G5 and treat all
chips of a pool as one device; since 18/10/2026 they are not part of chay_tatca.sh (thu_nghiem_g6.py replaces them, and
vi_du_so.py computes the real-data certificates of E5).
"""
import json
import math
import os
import site
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402

import chung_chi as cc  # noqa: E402
import tuoi_tho as tt  # noqa: E402

S_UP = 1.10            # upper thermal scale of the certificate (Assumption P3.3 with alpha_th = 0)
INTERVAL_DAYS = 30
TAMB_REF = "chu_ky_cruise_d00_base.npz"   # source of the daily ambient for every database (DayPool)
CHIPS = ("T1", "D1", "T2", "D2")              # chips carried by the certificates
MODULES = (("T1", "D1"), ("T2", "D2"))         # certified modules: outer and inner switch with their diodes
MODULE_NAMES = ("outer", "inner")


class DayPool:
    def __init__(self, npz, mg, chips=CHIPS, s_up=S_UP, modules=MODULES):
        d = np.load(HERE / npz)
        cyc = d["cycles"]
        names = [str(x) for x in d["names"]]
        self.days = d["day"]
        self.n = len(self.days)
        self.date = d["date"]
        pos = {k: i for i, k in enumerate(self.days)}
        # ambient of a day: that of the timetable, the same under every policy. nhiem_vu.py stores the median ambient over
        # the simulated timeline of the day, which a derated (longer) day can shift to another station's value (up to
        # 10 K on 228 of 4215 days at 50 % derating); the cycles are rises above ambient, so the label of the
        # underated cruising database applies to every database of the same days (18/10/2026)
        t_amb = d["T_amb"]
        ref = HERE / TAMB_REF
        if ref.exists() and ref.name != Path(npz).name:
            r_ = np.load(ref)
            if np.array_equal(r_["day"], self.days):
                t_amb = r_["T_amb"]
        tamb = np.zeros(int(cyc[:, 0].max()) + 1)
        tamb[self.days] = t_amb
        self.chips = [names.index(c) for c in chips]
        # modules: positions (in chips) of the chips of each module; a module's damage is that of its largest chip
        self.groups = [[list(chips).index(c) for c in mod] for mod in modules]
        self.K = len(self.groups)
        self.mod_of = np.full(len(chips), -1)
        for j, g in enumerate(self.groups):
            self.mod_of[g] = j
        self.cells, self.raw, self.flat = [], [], []
        order = np.lexsort((cyc[:, 1], cyc[:, 0]))
        cyc = cyc[order]
        for j, c in enumerate(self.chips):
            # every cycle of chip c in day order, as flat arrays with the offsets of each day (self.flat, for the truth
            # of a whole interval in one evaluation); the per-day dicts of self.raw are views into them
            s = cyc[cyc[:, 1] == c]
            if mg.short_rule == "exclude":
                s = s[s[:, 4] >= tt.TON_LO]              # outside the scope of both the certificate and the truth
            dday = s[:, 0].astype(np.int64)
            off = np.searchsorted(dday, np.r_[self.days, self.days[-1] + 1])
            fl = dict(dTheta=s[:, 2].astype(float), theta_min=s[:, 3].astype(float), ton=s[:, 4].astype(float),
                      n=s[:, 5].astype(float), T_amb=tamb[dday].astype(float))
            fl["off"] = off
            self.flat.append(fl)
            cl, rw = [], []
            for k in range(self.n):
                r = {key: fl[key][off[k]:off[k + 1]] for key in ("dTheta", "theta_min", "ton", "n", "T_amb")}
                rw.append(r)
                cl.append(tt.compress(*tt.prep(r, mg, s_up)))
            self.cells.append(cl)
            self.raw.append(rw)
        self.pos = pos

    def sample(self, rng, n_days=INTERVAL_DAYS, date_range=None):
        if date_range is None:
            return rng.integers(0, self.n, n_days)
        ok = np.flatnonzero((self.date >= date_range[0]) & (self.date <= date_range[1]))
        return rng.choice(ok, n_days)

    def model(self, idx, fitd, mg):
        """Margined model damage per chip of the interval made of days idx."""
        out = []
        for cl in self.cells:
            V = np.vstack([cl[i][0] for i in idx])
            la = np.concatenate([cl[i][1] for i in idx])
            out.append(tt.damage_from_prep(*tt.compress(V, la), fitd, mg))
        return np.array(out)

    def mod(self, x):
        """Largest chip of each module: x[..., chips] -> [..., modules]."""
        x = np.asarray(x, float)
        return np.stack([x[..., g].max(-1) for g in self.groups], -1)

    def per_chip(self, x):
        """A value per module (or one value for all) spread over the chips of each module."""
        x = np.broadcast_to(np.asarray(x, float), (self.K,))
        return x[np.maximum(self.mod_of, 0)]

    def true(self, idx, world, S, kappa, s_th, short_rule="exclude", shift=0.0):
        """True damage per chip; S, kappa and s_th are given per module (or as one value for all modules), since the
        modules of a leg are separate devices; shift moves the ambient temperature. short_rule follows the scope of the
        certificate (under 'exclude', cycles with a heating time below the shortest tested one are outside both the
        certificate and the truth)."""
        S, kappa, s_th = self.per_chip(S), self.per_chip(kappa), self.per_chip(s_th)
        idx = np.asarray(idx)
        out = np.empty(len(self.flat))
        for c, fl in enumerate(self.flat):
            # all cycles of the interval at once (a day drawn twice counts twice); the pool already holds only the
            # cycles in the scope of 'exclude'
            lo, hi = fl["off"][idx], fl["off"][idx + 1]
            ln = hi - lo
            sel = np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln) + np.repeat(lo, ln)
            cyc = {k: fl[k][sel] for k in ("dTheta", "theta_min", "ton", "n", "T_amb")}
            out[c] = world.true_damage(cyc, S[c], kappa[c], s_th[c], shift=shift, short_rule=short_rule)
        return out


def bench(world, rng, n_total):
    h = tt.hernes()
    runs = np.unique(h["run"])
    share = np.array([np.sum(h["run"] == r) for r in runs], float)
    cnt = np.maximum(2, np.round(n_total * share / share.sum()).astype(int))
    rows = np.concatenate([rng.choice(np.flatnonzero(h["run"] == r), c) for r, c in zip(runs, cnt)])
    dT, ton, I, run = h["dT"][rows], h["ton"][rows], h["I"][rows], h["run"][rows]
    return dict(dT=dT, ton=ton, I=I, run=run, Nf=world.bench(rng, dT, ton, I))


def e1(npz, reps, n1_list, n2, alpha, k_past, scenario="evidence", stat="ellipsoid", seed=20261016, n_oracle=200):
    rng = np.random.default_rng(seed)
    mg = tt.Margin(stat=stat)
    mg_none = tt.Margin(stat="none")
    t0 = time.time()
    pool = DayPool(npz, mg)
    print(f"pool {npz}: {pool.n} days, {len(pool.chips)} chips, built in {time.time() - t0:.0f} s", flush=True)
    h = tt.hernes()
    b_full = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])["b"]
    sc = {"evidence": tt.Scenario(), "adverse": tt.Scenario.adverse(), "boundary": tt.Scenario.boundary(),
          "adverse_strict": tt.Scenario.adverse_strict()}[scenario]
    res = {n1: [] for n1 in n1_list}
    for r in range(reps):
        world = tt.World.draw(rng, b_full, sc)
        # one oracle per world: (1 - alpha) quantile of the true end-of-interval damage of a fresh field device
        def field(rng_):
            S = world.strengths(rng_, 1)[0]
            kap, s_th = rng_.uniform(*sc.kappa), rng_.uniform(*sc.s_th)
            past = [pool.sample(rng_) for _ in range(k_past)]
            new = pool.sample(rng_)
            D_true = sum((pool.true(p, world, S, kap, s_th, mg.short_rule) for p in past),
                         np.zeros(len(pool.chips))) + pool.true(new, world, S, kap, s_th, mg.short_rule)
            return S, past, new, float(D_true.max())
        oracle = float(np.quantile([field(rng)[3] for _ in range(n_oracle)], 1 - alpha))
        # recorded intervals, shared by the bench sizes of this repetition
        rec = [pool.sample(rng) for _ in range(n2)]
        S_new, past, new, D_star = field(rng)
        for n1 in n1_list:
            bn = bench(world, rng, 2 * n1)
            fi, si = tt.split(bn["run"], "half", rng)
            fitd = tt.fit(bn["dT"][fi], bn["ton"][fi], bn["I"][fi], bn["Nf"][fi], bn["run"][fi])
            X = tt.features(bn["dT"], bn["ton"], bn["I"])
            S = np.log(bn["Nf"][si]) - X[si] @ fitd["b"]
            V = np.array([pool.model(q, fitd, mg) for q in rec])                      # [n2, chips]
            D0 = sum((pool.model(p, fitd, mg) for p in past), np.zeros(len(pool.chips)))
            V_new = pool.model(new, fitd, mg)
            # score of the field device at a tested condition drawn from the bench design (calibrated part)
            j = rng.integers(0, len(bn["dT"]))
            S_fld = S_new + X[j] @ (world.b - fitd["b"])
            U_new = math.exp(-S_fld) * float((D0 + V_new).max())
            # statistical part alone: margined damage with the true coefficients, shifted to the field device's
            # tested condition, against the margined damage with the fitted ones and the statistical part
            f_true = dict(b=world.b, cov=fitd["cov"], refs=fitd["refs"], dof=fitd["dof"])
            V_rel = pool.model(new, f_true, mg_none) * math.exp(X[j] @ (world.b - fitd["b"]))
            stat_ok = bool(np.all(V_rel <= V_new * (1 + 1e-9)))
            m = min(len(S), n2)
            cert = {
                "U1 box": cc.u1_box(S, V, D0),
                "U2 Monte Carlo": cc.u2_montecarlo(S, V, D0, alpha, rng),
                "U3 Bayes": cc.u3_bayes(S, V, D0, alpha, rng),
                "U5 union": cc.u5_union(S, V, D0, alpha),
                "U6 paired": cc.u6_paired(S[:m], V[:m], D0, alpha),
                "U9 PAC": cc.u9_pac(S[:m], V[:m], D0, alpha, alpha),
            }
            res[n1].append(dict(D_star=D_star, oracle=oracle, U_new=U_new, n_scores=len(S), cert=cert,
                                stat_ok=stat_ok))
        if (r + 1) % 10 == 0:
            print(f"  rep {r + 1}/{reps}, {time.time() - t0:.0f} s", flush=True)
    summ = {}
    for n1, rows in res.items():
        s = {}
        D = np.array([x["D_star"] for x in rows])
        O = np.array([x["oracle"] for x in rows])
        U = np.array([x["U_new"] for x in rows])
        for k in rows[0]["cert"]:
            C = np.array([x["cert"][k] for x in rows])
            fin = np.isfinite(C)
            s[k] = dict(coverage=float(np.mean(D <= C)), calibrated_coverage=float(np.mean(U <= C)),
                        finite=float(fin.mean()),
                        ratio_to_oracle_median=float(np.median(C[fin] / O[fin])) if fin.any() else None)
        s["oracle coverage"] = float(np.mean(D <= O))
        s["statistical part coverage"] = float(np.mean([x["stat_ok"] for x in rows]))
        summ[str(n1)] = s
    tag = f"e1_{Path(npz).stem}_n2{n2}_a{alpha}_k{k_past}_{scenario}_{stat}"
    (HERE / f"{tag}.json").write_text(json.dumps(dict(summary=summ, reps=reps, alpha=alpha, n2=n2, k_past=k_past,
                                                      scenario=scenario, seed=seed), indent=1), encoding="utf-8",
                                      newline="\n")
    print(json.dumps(summ, indent=1))


def e5(npz, reps, n2, alpha, stat="delta", scenario="evidence", seed=20261018, n_splits=200):
    """Split of the 34-module Hernes design between fitting and scores: 'half' (16 fit, 18 scores) against 'two' (two
    modules of each run fit, 24 scores). Part A: coverage and tightness on synthetic benches of that design. Part B: the
    certificates on the real bench data for the case-study mission (median over random splits), D0 = 0."""
    rng = np.random.default_rng(seed)
    mg = tt.Margin(stat=stat)
    pool = DayPool(npz, mg)
    h = tt.hernes()
    b_full = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])["b"]
    sc = {"evidence": tt.Scenario(), "adverse": tt.Scenario.adverse(), "boundary": tt.Scenario.boundary(),
          "adverse_strict": tt.Scenario.adverse_strict()}[scenario]
    X = tt.features(h["dT"], h["ton"], h["I"])
    out = {"A": {s: [] for s in ("half", "two")}, "B": {s: [] for s in ("half", "two")}}
    for r in range(reps):
        world = tt.World.draw(rng, b_full, sc)
        Nf = world.bench(rng, h["dT"], h["ton"], h["I"])
        rec = [pool.sample(rng) for _ in range(n2)]
        S_new = world.strengths(rng, 1)[0]
        kap, s_th = rng.uniform(*sc.kappa), rng.uniform(*sc.s_th)
        new = pool.sample(rng)
        D_star = float(pool.true(new, world, S_new, kap, s_th, mg.short_rule).max())
        for scheme in ("half", "two"):
            fi, si = tt.split(h["run"], scheme, rng)
            fitd = tt.fit(h["dT"][fi], h["ton"][fi], h["I"][fi], Nf[fi], h["run"][fi])
            S = np.log(Nf[si]) - X[si] @ fitd["b"]
            V = np.array([pool.model(q, fitd, mg) for q in rec])
            V_new = pool.model(new, fitd, mg)
            j = rng.integers(0, len(Nf))
            U_new = math.exp(-(S_new + X[j] @ (world.b - fitd["b"]))) * float(V_new.max())
            m = min(len(S), n2)
            Cp = cc.u6_paired(S[:m], V[:m], 0.0, alpha)
            Cu = cc.u5_union(S, V, 0.0, alpha)
            out["A"][scheme].append(dict(cov_p=D_star <= Cp, cal_p=U_new <= Cp, cov_u=D_star <= Cu,
                                         fin_u=math.isfinite(Cu), Cp=Cp, Cu=Cu, D_star=D_star))
        print(f"  A rep {r + 1}/{reps}", flush=True)
    # part B: the real bench data
    rec = [pool.sample(rng) for _ in range(n2)]
    for b in range(n_splits):
        for scheme in ("half", "two"):
            fi, si = tt.split(h["run"], scheme, rng)
            fitd = tt.fit(h["dT"][fi], h["ton"][fi], h["I"][fi], h["Nf"][fi], h["run"][fi])
            S = np.log(h["Nf"][si]) - X[si] @ fitd["b"]
            V = np.array([pool.model(q, fitd, mg) for q in rec])
            m = min(len(S), n2)
            out["B"][scheme].append(dict(Cp=cc.u6_paired(S[:m], V[:m], 0.0, alpha), Cu=cc.u5_union(S, V, 0.0, alpha),
                                         Vmed=float(np.median(V.max(1))), n_scores=len(S)))
        if (b + 1) % 50 == 0:
            print(f"  B split {b + 1}/{n_splits}", flush=True)
    summ = {}
    for scheme in ("half", "two"):
        A, B = out["A"][scheme], out["B"][scheme]
        fin = [x for x in A if math.isfinite(x["Cp"])]
        summ[scheme] = dict(
            coverage_paired=float(np.mean([x["cov_p"] for x in A])),
            calibrated_coverage_paired=float(np.mean([x["cal_p"] for x in A])),
            coverage_union=float(np.mean([x["cov_u"] for x in A])), union_finite=float(np.mean([x["fin_u"] for x in A])),
            ratio_Cp_to_true_median=float(np.median([x["Cp"] / x["D_star"] for x in fin])) if fin else None,
            real_data_Cp_median=float(np.median([x["Cp"] for x in B])),
            real_data_Cp_q10_q90=[float(np.quantile([x["Cp"] for x in B], q)) for q in (0.1, 0.9)],
            real_data_Cu_finite=float(np.mean([math.isfinite(x["Cu"]) for x in B])),
            real_data_V_interval_median=float(np.median([x["Vmed"] for x in B])),
            n_scores=B[0]["n_scores"])
    tag = f"e5_{Path(npz).stem}_n2{n2}_a{alpha}_{scenario}_{stat}"
    (HERE / f"{tag}.json").write_text(json.dumps(dict(summary=summ, reps=reps, n_splits=n_splits, seed=seed),
                                                 indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(summ, indent=1))


LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def e3(n_int=200, alpha_S=0.05, stat="delta", strategy="cruise", pattern="base", seed=20261019,
       gammas=(0.0, 0.6, 1.2, 2.08, 2.5), anchors=(0, 5, 10, 20)):
    """Collapse by caution on the case-study mission (Figure F6).
    (a) Loading summary X = damage-weighted mean ln dT of an interval's cycles, under each derating level; test weight
        p_{n+1}(X_t) of the weighted certificate (Proposition P3.20) with the density ratio of target to bench X
        (Gaussian kernels), against alpha_S (Proposition P3.22), without and with n_a anchor tests run at the loading
        summary of that level (Corollary P3.23 needs n_a w_a >= ((1 - alpha_S) / alpha_S) max w, about 19 anchors
        when the target law is narrow).
    (b) Margined damage of the interval (median) against the derating level for several swing margins gamma, the fitted
        exponent being beta = 2.08 (Proposition P3.24): of the critical chip of the leg in 'b', and per module (outer,
        inner; median of the largest chip of each module) in 'b_mod'."""
    rng = np.random.default_rng(seed)
    h = tt.hernes()
    fitd = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])
    out = {"levels": [], "a": {}, "b": {}, "b_mod": {}}
    days = None
    for d in LEVELS:
        f = HERE / f"chu_ky_{strategy}_d{int(round(100 * d)):02d}_{pattern}.npz"
        if not f.exists():
            continue
        out["levels"].append(d)
        res_b, res_m = {}, {}
        for g in gammas:
            mg = tt.Margin(stat=stat, gamma=g)
            pool = DayPool(f.name, mg)
            if days is None:
                days = [pool.sample(rng) for _ in range(n_int)]
            Vch = np.array([pool.model(q, fitd, mg) for q in days])
            V = Vch.max(1)
            res_b[str(g)] = float(np.median(V))
            res_m[str(g)] = [float(v) for v in np.median(pool.mod(Vch), 0)]
            if g == 1.2:
                # loading summary of every interval from the damage-weighted swings of the critical chip
                X = []
                for q in days:
                    k = int(np.argmax(pool.model(q, fitd, mg)))
                    Vc = np.vstack([pool.cells[k][i][0] for i in q])
                    la = np.concatenate([pool.cells[k][i][1] for i in q])
                    _, parts = tt.damage_from_prep(*tt.compress(Vc, la), fitd, tt.Margin(stat="none", gamma=g),
                                                   return_parts=True)
                    Vcc, _ = tt.compress(Vc, la)
                    X.append(float(np.sum(parts * Vcc[:, 1]) / parts.sum()))
                X = np.array(X)
                out["a"][str(d)] = {"X_median_K": float(np.exp(np.median(X)))}
                for n_a in anchors:
                    x_b = np.r_[np.log(h["dT"]), np.full(n_a, np.median(X))]
                    w = cc.kde_ratio(x_b, X)
                    wb = w(x_b)
                    wt = w(np.median(X))[0]
                    p = wt / (wb.sum() + wt)
                    out["a"][str(d)][f"p_test_anchors_{n_a}"] = float(p)
                    out["a"][str(d)][f"collapse_anchors_{n_a}"] = bool(p > alpha_S)
        out["b"][str(d)] = res_b
        out["b_mod"][str(d)] = res_m
        print(f"derating {d}: X {out['a'][str(d)]['X_median_K']:.1f} K; " + ", ".join(
            f"p_test(n_a={n}) {out['a'][str(d)][f'p_test_anchors_{n}']:.3f}" for n in anchors) + "; V(gamma): "
            + ", ".join(f"{k} {v:.4f}" for k, v in res_b.items()), flush=True)
    (HERE / f"e3_{strategy}_{pattern}_{stat}.json").write_text(json.dumps(out, indent=1), encoding="utf-8",
                                                               newline="\n")


def reserve(meth, S, V, D0, alpha, level=None):
    """Certificate of the coming interval and certified reserve R (intervals) for a method on data (S, V).
    union: R_c of Definition P3.6 (largest chip); paired-type: with k_i = min_chip (e^{S_i} - D0) / V_i the largest k with
    U_i(k) = e^{-S_i} max (D0 + k V_i) <= 1, R is the (m - K + 1)-th smallest k_i, K = ceil((1 - a)(m + 1))."""
    a = alpha if level is None else min(max(level, 1e-6), 1 - 1e-6)
    V = np.asarray(V, float)
    if meth == "union":
        # reserve of the union certificate of a module: the largest k for which the quantile of the largest chip of
        # D0 + k V over the recorded intervals stays below exp(q_S), so that C <= 1 exactly when R >= 1 (until
        # 18/10/2026 per-chip quantiles, which overstate the reserve when the critical chip changes between intervals)
        C = cc.u5_union(S, V, D0, alpha)
        qS = cc.kth(S, math.floor(alpha / 2 * (len(S) + 1) + 1e-12))
        Kv = math.ceil((1 - alpha / 2) * (len(V) + 1) - 1e-12)
        if not math.isfinite(qS) or Kv > len(V):
            return C, 0.0
        top = math.exp(qS)
        D0v = np.broadcast_to(np.asarray(D0, float), V.shape[1:])
        q = lambda k: float(np.partition((D0v[None, :] + k * V).max(1), Kv - 1)[Kv - 1])
        if q(0.0) >= top:
            return C, 0.0
        lo, hi = 0.0, 1.0
        while q(hi) < top and hi < 1e9:
            lo, hi = hi, 2 * hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if q(mid) < top else (lo, mid)
        return C, float(lo)
    m = min(len(S), len(V))
    U = cc.paired_scores(S[:m], V[:m], D0)
    K = math.ceil((1 - a) * (m + 1) - 1e-12)
    C = cc.kth(U, K)
    if K > m:
        return C, 0.0
    k = np.min((np.exp(S[:m])[:, None] - D0[None, :]) / V[:m], axis=1)
    return C, float(max(0.0, np.sort(k)[m - K]))


def e2(reps, n1, n2, alpha, scenario="evidence", stat="delta", strategy="cruise", pattern="base", t_max=240,
       seed=20261017, policy="reserve", t_plan=60):
    """Certificate-triggered derating over the life of a field module. At every decision instant the policy computes, for
    every derating level on policy data, the certificate of the coming interval and the certified reserve R (in
    intervals). 'pass': the smallest level whose certificate passes (C <= 1). 'reserve': the smallest level that passes
    with R >= t_plan - t (the module should last to its planned replacement), else the passing level with the largest R.
    The module is replaced when no level passes. The coming interval is then certified on separate data under the chosen
    level and run. Methods:
      union   policy on all bench scores and recorded split J1; certificate on the same scores and split J2
              (Proposition P3.11 with Proposition P3.17: only the recorded inputs are split);
      paired  policy on a bench subset kept apart and J1; certificate pairs the other scores with J2
              (Remark P3.18: the price of pairing is the bench subset spent on the policy);
      split0  as paired, but policy and certificate use the recorded intervals under the baseline policy at every
              derating level (no re-simulation, U4): the policy cannot see what derating buys;
      reuse   paired policy and certificate on the same scores and J2 (data reuse, no guarantee);
      aci     paired certificate at a level alpha_t updated after every interval from the miscoverage indicator given
              by the truth (Gibbs and Candes 2021); the level persists across the modules of the fleet, as it would
              online. No deployment observes this indicator before failure (Remark P3.16)."""
    rng = np.random.default_rng(seed)
    mg = tt.Margin(stat=stat)
    t0 = time.time()
    pools = {}
    for d in LEVELS:
        f = HERE / f"chu_ky_{strategy}_d{int(round(100 * d)):02d}_{pattern}.npz"
        if f.exists():
            pools[d] = DayPool(f.name, mg)
    levels = sorted(pools)
    print(f"pools for derating levels {levels} built in {time.time() - t0:.0f} s", flush=True)
    late = json.loads((HERE / "giam_tai_kha_thi.json").read_text(encoding="utf-8"))
    late = {d: 1 - late[f"{d:.1f}"]["on_time_share"] for d in levels}
    h = tt.hernes()
    b_full = tt.fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])["b"]
    sc = {"evidence": tt.Scenario(), "adverse": tt.Scenario.adverse(), "boundary": tt.Scenario.boundary(),
          "adverse_strict": tt.Scenario.adverse_strict()}[scenario]
    out = {m: [] for m in ("union", "paired", "split0", "reuse", "aci")}
    aci = cc.ACI(alpha, eta=0.005)
    for r in range(reps):
        world = tt.World.draw(rng, b_full, sc)
        bn = bench(world, rng, 3 * n1)                     # half fits the model; the rest splits into two score sets
        fi, si = tt.split(bn["run"], "half", rng)
        fitd = tt.fit(bn["dT"][fi], bn["ton"][fi], bn["I"][fi], bn["Nf"][fi], bn["run"][fi])
        X = tt.features(bn["dT"], bn["ton"], bn["I"])
        S_all = np.log(bn["Nf"][si]) - X[si] @ fitd["b"]
        perm = rng.permutation(len(si))
        S_pol, S_cer = S_all[perm[:len(si) // 2]], S_all[perm[len(si) // 2:]]
        J1 = [pools[0.0].sample(rng) for _ in range(n2)]
        J2 = [pools[0.0].sample(rng) for _ in range(n2)]
        V1 = {d: np.array([pools[d].model(q, fitd, mg) for q in J1]) for d in levels}
        V2 = {d: np.array([pools[d].model(q, fitd, mg) for q in J2]) for d in levels}
        S_new = world.strengths(rng, 1)[0]
        kap, s_th = rng.uniform(*sc.kappa), rng.uniform(*sc.s_th)
        life = [pools[0.0].sample(rng) for _ in range(t_max)]   # the same mission days for every method
        for meth in out:
            D0 = np.zeros(len(pools[0.0].chips))
            Dt = np.zeros_like(D0)
            hist = []
            failed = False
            for t in range(t_max):
                cand = []
                for d in levels:
                    if meth == "union":
                        Cp, R = reserve("union", S_all, V1[d], D0, alpha)
                    elif meth == "reuse":
                        Cp, R = reserve("paired", S_cer, V2[d], D0, alpha)
                    elif meth == "aci":
                        Cp, R = reserve("paired", S_pol, V1[d], D0, alpha, level=aci.a)
                    elif meth == "split0":
                        # without re-simulation the recorded intervals carry the baseline damage at every level, so the
                        # policy cannot see what derating buys (Proposition P3.17)
                        Cp, R = reserve("paired", S_pol, V1[0.0], D0, alpha)
                    else:
                        Cp, R = reserve("paired", S_pol, V1[d], D0, alpha)
                    if Cp <= 1:
                        cand.append((d, R))
                if not cand:
                    break                                          # replacement: no level is certified
                if policy == "pass":
                    d_sel = cand[0][0]
                else:
                    ok = [d for d, R in cand if R >= max(t_plan - t, 1)]
                    d_sel = ok[0] if ok else max(cand, key=lambda x: x[1])[0]
                if meth == "union":
                    Cc = cc.u5_union(S_all, V2[d_sel], D0, alpha)
                elif meth == "aci":
                    m = min(len(S_cer), n2)
                    Cc = aci.certificate(S_cer[:m], V2[d_sel][:m], D0)
                else:
                    m = min(len(S_cer), n2)
                    Vc = V2[0.0] if meth == "split0" else V2[d_sel]
                    Cc = cc.u6_paired(S_cer[:m], Vc[:m], D0, alpha)
                Vt = pools[d_sel].model(life[t], fitd, mg)
                Dn = pools[d_sel].true(life[t], world, S_new, kap, s_th, mg.short_rule)
                hist.append((d_sel, bool((Dt + Dn).max() <= Cc)))
                if meth == "aci":
                    aci.update(not hist[-1][1])
                D0, Dt = D0 + Vt, Dt + Dn
                if Dt.max() >= 1:
                    failed = True
                    break
            ds = np.array([x[0] for x in hist]) if hist else np.zeros(0)
            out[meth].append(dict(intervals=len(hist), harvest=float(Dt.max()), failed=failed,
                                  aci_level=float(aci.a) if meth == "aci" else math.nan,
                                  coverage=float(np.mean([x[1] for x in hist])) if hist else math.nan,
                                  derated_share=float(np.mean(ds > 0)) if hist else math.nan,
                                  mean_level=float(ds.mean()) if hist else math.nan,
                                  late_share=float(np.mean([late[d] for d in ds])) if hist else math.nan))
        print(f"  rep {r + 1}/{reps}, {time.time() - t0:.0f} s: " + "; ".join(
            f"{m} {v[-1]['intervals']} int., harvest {v[-1]['harvest']:.3f}" for m, v in out.items()), flush=True)
    summ = {m: {k: float(np.nanmean([x[k] for x in v])) for k in v[0]} for m, v in out.items()}
    tag = f"e2_{strategy}_{pattern}_n1{n1}_n2{n2}_a{alpha}_{scenario}_{stat}_{policy}{t_plan if policy == 'reserve' else ''}"
    (HERE / f"{tag}.json").write_text(json.dumps(dict(summary=summ, runs=out, reps=reps, levels=levels, seed=seed,
                                                      policy=policy, t_plan=t_plan),
                                                 indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "e2":
        e2(int(a[2]), int(a[3]), int(a[4]), float(a[5]), a[6] if len(a) > 6 else "evidence",
           a[7] if len(a) > 7 else "delta")
    if a[1] == "e3":
        e3(int(a[2]) if len(a) > 2 else 200, strategy=a[3] if len(a) > 3 else "cruise")
    if a[1] == "e5":
        e5(a[2], int(a[3]), int(a[4]), float(a[5]), a[6] if len(a) > 6 else "delta", a[7] if len(a) > 7 else "evidence")
    if a[1] == "e1":
        e1(a[2], int(a[3]), [int(x) for x in a[4].split(",")], int(a[5]), float(a[6]), int(a[7]),
           a[8] if len(a) > 8 else "evidence", a[9] if len(a) > 9 else "ellipsoid")
