"""P3 case study (G5): lifetime model, stated margin (Table T5) and the data-generating truth.

Model (G3): ln N = c + b1 ln dT + b3 ln t_on + b4 ln I, fitted by least squares on the Hernes et al. (2021) design
(../P3_kiemtoan/G2_hernes2021_sohoa.csv, 34 modules, five runs) or on synthetic bench data drawn on that design.
A field cycle x = (dT, T_min, t_on) is evaluated at x_c: t_on clipped to [TON_LO, TON_HI], current clipped to
[I_LO, I_HI] (the field current, at most 554 A in the 3.3 kV converter of the case study, is always taken at 890 A);
dT is not clipped.

Stated margin eps(x) of Assumption P3.2 (Table T5), in natural-log units of life:
  swing      gamma * ln(dT_T / dT) below dT_T = 26 K, gamma = 1.2;
  heating    t_on above 4.5 s: gamma_t * ln(t_on / 4.5), gamma_t = 0.5, with a rule above TON_CAP = 60 s
             ('uncapped': same slope; 'cap': no further margin; 'slope': slope gamma_h; 'anchor': the stated rule
             replaced by anchor tests at heating times t_a[0] < t_a[1] < ... with upper confidence bounds delta_a on
             the effect there; life falls with the heating time, so for 4.5 s < t_on <= t_a[-1] the margin is the
             smallest bound of the anchors at or above t_on, and above t_a[-1] it is delta_a[-1] + gamma_t
             ln(t_on / t_a[-1]); only monotonicity is assumed below the last anchor);
  short      t_on below 0.15 s ('clip': no credit, the G3 rule; 'credit': life rises with slope gamma_s below
             0.15 s; 'exclude': such cycles are outside the certificate and the truth);
  T_min      (E_a / k_B)(1 / 333.15 - 1 / T_min) above 60 C, E_a = 0.159 eV; no credit below 60 C;
  Miner      eps_M = -ln kappa_M, kappa_M = 0.85;
  statistics of the fit, relative to the tested condition at which a score is a residual (the error of the
             intercept cancels in U = exp(-S) V): 'pointwise' z * se((v(x_c) - v_ref)' b); 'ellipsoid' the largest
             damage over the confidence ellipsoid of the coefficients (one error common to all cycles, Scheffe radius on
             the rank of the differences, level 1 - alpha_stat); both maximized over the tested conditions; 'none'.

Truth (protocol of the outline, Section 10): on the reference set the log-life is linear with coefficients b_true;
off it, the true swing exponent below dT_T, the true heating-time slopes, the true short-pulse slope, the true E_a,
the true Miner factor and the true thermal scale are drawn per world from evidence ranges (Scenario). The strength
law is the minimum of two Gumbel laws (two failure mechanisms), not a Weibull, so the parametric certificates are
misspecified in a controlled way.
"""
import csv
import math
import site
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

HERE = Path(__file__).parent
sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

K_B = 8.617333e-5
TON_LO, TON_HI, TON_CAP = 0.15, 4.5, 60.0
I_LO, I_HI, I_FIELD = 890.0, 1250.0, 890.0
TMIN_T = 60.0
HERNES = HERE.parent / "P3_kiemtoan" / "G2_hernes2021_sohoa.csv"


def hernes():
    with open(HERNES, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return dict(run=np.array([int(r["run"]) for r in rows]), I=np.array([float(r["I_load_A"]) for r in rows]),
                ton=np.array([float(r["t_on_s"]) for r in rows]), dT=np.array([float(r["dT_C"]) for r in rows]),
                Nf=np.array([float(r["Nf"]) for r in rows]))


def features(dT, ton, I):
    return np.column_stack([np.ones(np.size(dT)), np.log(dT), np.log(ton), np.log(I)])


def fit(dT, ton, I, Nf, run=None):
    """Least squares; refs are the feature vectors of the tested conditions (run means), against which the statistical
    part of the margin is taken (the scores are residuals at those conditions)."""
    X, y = features(dT, ton, I), np.log(Nf)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    dof = len(y) - X.shape[1]
    s2 = float(res @ res / dof)
    run = np.zeros(len(y), int) if run is None else np.asarray(run)
    refs = np.array([X[run == r].mean(0) for r in np.unique(run)])
    return dict(b=b, cov=s2 * np.linalg.inv(X.T @ X), s2=s2, dof=dof, refs=refs)


def loo_scores(dT, ton, I, Nf):
    """Leave-one-out (PRESS) residuals: the scores of jackknife+ with the model fitted on all modules."""
    X, y = features(dT, ton, I), np.log(Nf)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    return (y - X @ b) / (1 - np.diag(H))


def split(run, scheme, rng):
    """Indices (fit, score) per run: 'half' floor(n_r / 2) modules of each run fit the model, 'two' two of each run."""
    fi, si = [], []
    for r in np.unique(run):
        idx = rng.permutation(np.flatnonzero(run == r))
        k = idx.size // 2 if scheme == "half" else 2
        fi += list(idx[:k])
        si += list(idx[k:])
    return np.array(fi), np.array(si)


@dataclass
class Margin:
    dT_T: float = 26.0
    gamma: float = 1.2
    gamma_t: float = 0.5
    ton_rule: str = "uncapped"     # 'uncapped' | 'cap' | 'slope'
    gamma_h: float = 0.27
    short_rule: str = "exclude"    # 'clip' | 'credit' | 'exclude' (G5 base: exclude, a stated scope item)
    gamma_s: float = 0.59
    Ea: float = 0.159
    kappa_M: float = 0.85
    stat: str = "ellipsoid"        # 'ellipsoid' | 'delta' | 'pointwise' | 'none'
    alpha_stat: float = 0.05
    t_a: tuple = (300.0,)          # heating times of the anchor tests, ascending (ton_rule 'anchor')
    delta_a: tuple = (math.inf,)   # upper bounds on the heating-time effect at t_a (natural-log units of life)

    def eps_M(self):
        return -math.log(self.kappa_M)


def clip_x(dT, ton):
    return dT, np.clip(ton, TON_LO, TON_HI), np.full(np.size(dT), I_FIELD)


def eps_stated(dT, Tmin, ton, mg):
    """Stated margin per cycle without the statistical part and without the Miner factor (natural-log units)."""
    e = mg.gamma * np.log(np.maximum(mg.dT_T / dT, 1.0))
    t_mid = np.clip(ton, TON_HI, TON_CAP)
    if mg.ton_rule == "anchor":
        ta, da = np.asarray(mg.t_a, float), np.asarray(mg.delta_a, float)
        dm = np.minimum.accumulate(da[::-1])[::-1]          # smallest bound at or above each anchor time
        j = np.searchsorted(ta, ton, side="left")           # first anchor time >= t_on
        et = np.where(j < ta.size, dm[np.minimum(j, ta.size - 1)],
                      da[-1] + mg.gamma_t * np.log(np.maximum(ton / ta[-1], 1.0)))
        e = e + np.where(ton > TON_HI, et, 0.0)
    else:
        e = e + mg.gamma_t * np.log(t_mid / TON_HI)
    if mg.ton_rule == "uncapped":
        e = e + mg.gamma_t * np.log(np.maximum(ton / TON_CAP, 1.0))
    elif mg.ton_rule == "slope":
        e = e + mg.gamma_h * np.log(np.maximum(ton / TON_CAP, 1.0))
    if mg.short_rule == "credit":
        e = e - mg.gamma_s * np.log(np.maximum(TON_LO / ton, 1.0))
    TK = Tmin + 273.15
    e = e + mg.Ea / K_B * np.maximum(1.0 / (TMIN_T + 273.15) - 1.0 / TK, 0.0)
    return e


def prep(cyc, mg, s=1.0, shift=0.0):
    """Feature rows V (k x 4) and log weights loga = ln n + eps_stated + eps_M of a set of cycles, cyc = dict(dTheta,
    theta_min, ton, n, T_amb) (arrays), under the upper thermal scale s and an ambient shift. Independent of the fit."""
    dT = np.maximum(s * cyc["dTheta"], 1e-3)
    Tmin = cyc["T_amb"] + shift + s * cyc["theta_min"]
    ton, n = cyc["ton"], cyc["n"]
    keep = (n > 0) if mg.short_rule != "exclude" else (n > 0) & (ton >= TON_LO)
    dT, Tmin, ton, n = dT[keep], Tmin[keep], ton[keep], n[keep]
    return features(*clip_x(dT, ton)), np.log(n) + eps_stated(dT, Tmin, ton, mg) + mg.eps_M()


def compress(V, loga, step=(0.01, 0.05)):
    """Aggregate cycles into cells of (ln dT, ln t_on clipped): weight sum and weight-averaged features. The damage of a
    cell under coefficients b is then W exp(-v_cell'b), with an error below step * |b| / 2 per cycle."""
    if V.shape[0] == 0:
        return V, loga
    key = np.floor(V[:, 1] / step[0]).astype(np.int64) * 100_000 + np.floor(V[:, 2] / step[1]).astype(np.int64)
    u, inv = np.unique(key, return_inverse=True)
    m = loga.max()
    w = np.exp(loga - m)
    W = np.bincount(inv, weights=w)
    Vc = np.stack([np.bincount(inv, weights=w * V[:, j]) / W for j in range(V.shape[1])], 1)
    return Vc, np.log(W) + m


_MC_CACHE = {}


def _mc_draws(p, dof, n=2000, seed=20261019):
    """Standard multivariate Student t draws [n, p] with dof degrees of freedom, fixed (statistical part 'mc')."""
    key = (p, int(dof), n)
    if key not in _MC_CACHE:
        rng = np.random.default_rng(seed)
        _MC_CACHE[key] = rng.standard_normal((n, p)) * np.sqrt(dof / rng.chisquare(dof, n))[:, None]
    return _MC_CACHE[key]


def damage_from_prep(V, loga, fitd, mg, return_parts=False):
    """Margined model damage sum_k exp(loga_k - v_k'b), with the statistical part of mg.stat."""
    b, cov = fitd["b"], fitd["cov"]
    if V.shape[0] == 0:
        return 0.0 if not return_parts else (0.0, np.zeros(0))
    d0 = np.exp(loga - V @ b)
    if mg.stat == "none":
        return d0.sum() if not return_parts else (d0.sum(), d0)
    best, best_d = -np.inf, d0
    if mg.stat == "delta":
        # one-sided bound on log V along its gradient: exact for a linear functional of the coefficient error, first
        # order for the convex log-sum; t quantile with the residual degrees of freedom
        t = stats.t.ppf(1 - mg.alpha_stat, fitd["dof"])
        for vr in fitd["refs"]:
            Dv = V - vr
            g = -(Dv.T @ d0) / d0.sum()
            d = d0 * math.exp(t * math.sqrt(max(float(g @ cov @ g), 0.0)))
            if d.sum() > best:
                best, best_d = d.sum(), d
        return best if not return_parts else (best, best_d)
    if mg.stat == "mc":
        # Monte Carlo over the coefficient error (19/10/2026, referee request): the 1 - alpha_stat quantile of the
        # interval damage over draws db = L t (multivariate Student t with the residual degrees of freedom, common to
        # all cycles), relative to each tested condition; keeps the curvature of the convex log V that the delta bound
        # linearizes, without the Scheffe radius of the ellipsoid. Fixed draws, so the bound is deterministic.
        Z = _mc_draws(V.shape[1], fitd["dof"])
        L = np.linalg.cholesky(cov)
        dB = Z @ L.T                                       # [draws, p]
        for vr in fitd["refs"]:
            Dv = V - vr
            logs = np.log(np.exp(-(Dv @ dB.T)).T @ d0)     # [draws]
            k = int(np.ceil((1 - mg.alpha_stat) * len(logs))) - 1
            i = int(np.argsort(logs)[k])
            d = d0 * np.exp(-(Dv @ dB[i]))
            if d.sum() > best:
                best, best_d = d.sum(), d
        return best if not return_parts else (best, best_d)
    if mg.stat == "pointwise":
        z = stats.norm.ppf(1 - mg.alpha_stat)
        for vr in fitd["refs"]:
            Dv = V - vr
            d = d0 * np.exp(z * np.sqrt(np.einsum("ij,jk,ik->i", Dv, cov, Dv)))
            if d.sum() > best:
                best, best_d = d.sum(), d
        return best if not return_parts else (best, best_d)
    # ellipsoid: the coefficient error db is common to all cycles; the scores are residuals at a tested condition
    # v_ref, so a cycle carries (v_k - v_ref)'db. Maximize log sum_k d0_k exp(-(v_k - v_ref)'db) over
    # db' cov^-1 db <= r^2 (Scheffe radius on the rank of the differences), and over the tested conditions.
    L = np.linalg.cholesky(cov)
    for vr in fitd["refs"]:
        Dv = V - vr
        p = max(1, min(V.shape[1], int(np.linalg.matrix_rank(Dv[np.argsort(-d0)[:200]]))))
        r = math.sqrt(p * stats.f.ppf(1 - mg.alpha_stat, p, fitd["dof"]))
        u = np.zeros(V.shape[1])
        for _ in range(40):
            w = d0 * np.exp(-(Dv @ (L @ u)))
            g = -(L.T @ (Dv.T @ w)) / w.sum()            # gradient of log V in whitened coordinates
            u_new = r * g / max(np.linalg.norm(g), 1e-300)
            if np.linalg.norm(u_new - u) < 1e-10:
                break
            u = u_new
        d = d0 * np.exp(-(Dv @ (L @ u)))
        if d.sum() > best:
            best, best_d = d.sum(), d
    return best if not return_parts else (best, best_d)


def model_damage(cyc, fitd, mg, s=1.0, shift=0.0, return_parts=False):
    """Margined model damage of a set of cycles (see prep and damage_from_prep)."""
    V, loga = prep(cyc, mg, s, shift)
    return damage_from_prep(V, loga, fitd, mg, return_parts)


@dataclass
class Scenario:
    """Ranges of the truth off the reference set, drawn per world. 'evidence' (default): every stated margin of the base
    rule set holds, so only the statistical part and the calibrated part can fail; ranges from the sources of G3 and
    G10. 'adverse': every stated margin can be exceeded."""
    beta_low_shift: tuple = (-1.2, 3.0)   # true exponent below dT_T minus |b1|; covered by gamma = 1.2 if >= -1.2
    gamma_t: tuple = (0.14, 0.46)         # true heating-time slope 4.5-60 s (Choi 2017 to CIPS08), covered by 0.5
    gamma_h: tuple = (0.0, 0.46)          # true slope above 60 s, covered by the uncapped 0.5
    gamma_s: tuple = (0.59, 1.2)          # true short-pulse slope below 0.15 s; 0.59-0.74 measured over 0.07-2 s
    #                                       (Junghaenel 2017), steeper below 30 K for wire bonds (Lutz 2020)
    Ea: tuple = (0.069, 0.159)            # eV, covered by 0.159
    kappa: tuple = (0.85, 1.08)           # true Miner factor, covered by 0.85
    s_th: tuple = (0.92, 1.08)            # true thermal scale against the nominal model, covered by the upper 1.10
    tmin_credit: bool = True              # the truth credits T_min below 60 C, the stated margin does not
    strength: tuple = (0.08, 0.20, 0.40)  # Gumbel scales of the two mechanisms and the location offset of the second:
    #                                       sd of S 0.138 (Hernes 0.127), 14 % second mechanism, lower 1 % tail 0.07
    #                                       below a Weibull fitted by moments
    # Structural departures from the family of the stated margin (added 19/10/2026 after the third review; zero by
    # default, and then not drawn, so that every earlier configuration keeps its random stream):
    c_int: tuple = (0.0, 0.0)             # swing-heating-time interaction: below 26 K the heating-time effect above
    #                                       4.5 s steepens by c_int log(26/dT), which no slope of the margin can follow
    c_lutz: tuple = (0.0, 0.0)            # wire-bond lives rising faster than a power law below 26 K (Lutz 2020):
    #                                       life multiplied by exp(c_lutz log(26/dT)^2)
    # second batch (19/10/2026, editor re-review):
    nc_sd: tuple = (0.0, 0.0)             # strength not common: the field module's log-strength at field conditions
    #                                       differs from its reference strength by N(0, nc_sd^2) (applied in rep_A)
    c_th: tuple = (0.0, 0.0)              # thermal error not a pure scale: the true swing grows with the heating time,
    #                                       s_th dTheta (1 + c_th ln(t_on / 4.5 s)) above 4.5 s (slow branches degrade)
    c_seq: tuple = (0.0, 0.0)             # Miner factor depending on the load mix: kappa (1 - c_seq f_small), f_small the
    #                                       share of the true damage of an interval carried by cycles below 26 K
    # fourth round (20/10/2026): a weak subpopulation (manufacturing defects such as voided die attach), a share weak_p
    # of the devices of the lot, bench and field alike, with log-strength lower by weak_d; a strength law that the
    # Weibull (Gumbel) family of the parametric certificates does not contain
    weak_p: tuple = (0.0, 0.0)
    weak_d: tuple = (0.0, 0.0)

    def structural(self, kind):
        """A copy with a structural departure: 'int' (interaction, c_int 0.10-0.30), 'lutz' (c_lutz 1-3), 'nc'
        (non-common strength, nc_sd 0.05-0.15), 'th' (thermal error growing with heating time, c_th 0.02-0.05),
        'seq' (Miner factor depending on the share of small cycles, c_seq 0.1-0.3) or 'weak' (a weak subpopulation of
        2-10 % of the devices with lives 1.6-2.7 times shorter, weak_d 0.5-1.0)."""
        return replace(self, **{"int": dict(c_int=(0.10, 0.30)), "lutz": dict(c_lutz=(1.0, 3.0)),
                                "nc": dict(nc_sd=(0.05, 0.15)), "th": dict(c_th=(0.02, 0.05)),
                                "seq": dict(c_seq=(0.1, 0.3)),
                                "weak": dict(weak_p=(0.02, 0.10), weak_d=(0.5, 1.0))}[kind])

    @staticmethod
    def boundary():
        """Truth at the edge of every stated margin: only the statistical part of the fit and the calibrated part can
        make the certificate fail, so coverage is tested without the slack of the margins."""
        return Scenario(beta_low_shift=(-1.2, -1.2), gamma_t=(0.5, 0.5), gamma_h=(0.5, 0.5), gamma_s=(0.59, 0.59),
                        Ea=(0.159, 0.159), kappa=(0.85, 0.85), s_th=(1.10, 1.10), tmin_credit=False)

    @staticmethod
    def adverse_strict():
        """The adverse ranges without the credit of T_min below 60 C in the truth, which the stated margin does not take
        and which otherwise hides margin violations up to a factor of about 1.7."""
        sc = Scenario.adverse()
        sc.tmin_credit = False
        return sc

    @staticmethod
    def adverse():
        return Scenario(beta_low_shift=(-2.0, 1.0), gamma_t=(0.3, 0.6), gamma_h=(0.2, 0.6), gamma_s=(0.3, 0.74),
                        Ea=(0.1, 0.2), kappa=(0.80, 1.0), s_th=(0.95, 1.15))

    @staticmethod
    def beyond():
        """Added 18/10/2026 after the second review: every acting component uniform on a range centred beyond its
        stated value, so that each exceeds it with probability about 2/3 (shift of the swing exponent below 26 K -2.0
        to -0.8 against the stated -1.2; both heating-time slopes 0.40-0.70 against 0.5; activation energy 0.13-0.22 eV
        against 0.159; Miner factor 0.75-0.90 against 0.85; thermal scale 1.05-1.20 against 1.10), without the credit
        of T_min below 60 C."""
        return Scenario(beta_low_shift=(-2.0, -0.8), gamma_t=(0.40, 0.70), gamma_h=(0.40, 0.70), gamma_s=(0.3, 0.74),
                        Ea=(0.13, 0.22), kappa=(0.75, 0.90), s_th=(1.05, 1.20), tmin_credit=False)


@dataclass
class World:
    b: np.ndarray
    beta_low: float
    gamma_t: float
    gamma_h: float
    gamma_s: float
    Ea: float
    strength: tuple
    sc: Scenario = field(default_factory=Scenario)
    tmin_credit: bool = True       # Arrhenius credit below 60 C in the truth (none in the stated margin)
    c_int: float = 0.0             # structural departures (Scenario.structural), zero unless drawn
    c_lutz: float = 0.0
    nc_sd: float = 0.0
    c_th: float = 0.0
    c_seq: float = 0.0
    weak_p: float = 0.0            # weak subpopulation (20/10/2026), zero unless drawn
    weak_d: float = 0.0

    @staticmethod
    def draw(rng, b_true, sc=Scenario()):
        u = lambda ab: rng.uniform(*ab)
        w = World(b=np.array(b_true, float), beta_low=abs(b_true[1]) + u(sc.beta_low_shift), gamma_t=u(sc.gamma_t),
                  gamma_h=u(sc.gamma_h), gamma_s=u(sc.gamma_s), Ea=u(sc.Ea), strength=sc.strength, sc=sc,
                  tmin_credit=sc.tmin_credit)
        # drawn only when present, after the other draws, so that the default scenarios keep their random stream
        if sc.c_int != (0.0, 0.0):
            w.c_int = u(sc.c_int)
        if sc.c_lutz != (0.0, 0.0):
            w.c_lutz = u(sc.c_lutz)
        if sc.nc_sd != (0.0, 0.0):
            w.nc_sd = u(sc.nc_sd)
        if sc.c_th != (0.0, 0.0):
            w.c_th = u(sc.c_th)
        if sc.c_seq != (0.0, 0.0):
            w.c_seq = u(sc.c_seq)
        if sc.weak_p != (0.0, 0.0):
            w.weak_p, w.weak_d = u(sc.weak_p), u(sc.weak_d)
        return w

    def strengths(self, rng, n):
        """Log-strength S of n devices: minimum of two Gumbel (minimum) laws; the same law on the bench and in the field
        (Assumption P3.1). Note (17/10/2026): the term s*ln(ln 2) lowers each mechanism by 0.3665 s instead of centring
        it, so the median of the first mechanism is -0.059 and the location of the second lies 0.356, not 0.40, above
        that of the first. Only these differences matter (the fit absorbs the common location); the moments quoted in
        the manuscript (sd 0.138, 14 % second mechanism) are those of this law as coded."""
        s1, s2, off = self.strength
        g1 = -rng.gumbel(0.0, s1, n) + s1 * math.log(math.log(2))
        g2 = -rng.gumbel(0.0, s2, n) + s2 * math.log(math.log(2)) + off
        S = np.minimum(g1, g2)
        if self.weak_p:
            # weak subpopulation (20/10/2026); drawn only in that truth, so other streams are unchanged
            S = S - self.weak_d * (rng.random(n) < self.weak_p)
        return S

    def bench(self, rng, dT, ton, I):
        """Synthetic bench lives on reference conditions (truth exactly log-linear there)."""
        S = self.strengths(rng, len(dT))
        return np.exp(features(dT, ton, I) @ self.b + S)

    def log_life(self, dT, Tmin, ton, S):
        """True log-life of the cycles of a device with log-strength S."""
        dT = np.maximum(dT, 1e-3)
        base = features(*clip_x(dT, ton)) @ self.b
        b1 = abs(self.b[1])
        dev = -(b1 - self.beta_low) * np.log(np.maximum(26.0 / dT, 1.0))          # life change below 26 K
        dev -= self.gamma_t * np.log(np.clip(ton, TON_HI, TON_CAP) / TON_HI)
        dev -= self.gamma_h * np.log(np.maximum(ton / TON_CAP, 1.0))
        dev += self.gamma_s * np.log(np.maximum(TON_LO / ton, 1.0))
        TK = Tmin + 273.15
        arr = self.Ea / K_B * (1.0 / TK - 1.0 / (TMIN_T + 273.15))
        dev += arr if self.tmin_credit else np.minimum(arr, 0.0)                  # Arrhenius (credit below 60 C)
        if self.c_int or self.c_lutz:
            low = np.log(np.maximum(26.0 / dT, 1.0))
            dev -= self.c_int * low * np.log(np.maximum(ton, TON_HI) / TON_HI)      # interaction (19/10/2026)
            dev += self.c_lutz * low ** 2                                           # Lutz-type steepening
        return S + base + dev

    def true_damage(self, cyc, S, kappa, s_th, shift=0.0, short_rule="clip"):
        dT = np.maximum(s_th * cyc["dTheta"], 1e-3)
        Tmin = cyc["T_amb"] + shift + s_th * cyc["theta_min"]
        keep = np.ones(dT.size, bool) if short_rule != "exclude" else cyc["ton"] >= TON_LO
        if not (self.c_th or self.c_seq):
            return float(np.sum(cyc["n"][keep] / (kappa * np.exp(self.log_life(dT[keep], Tmin[keep], cyc["ton"][keep],
                                                                                 S)))))
        # structural departures of 19/10/2026: a thermal error growing with the heating time, and a Miner factor
        # lowered by the share of damage carried by small cycles
        ton = cyc["ton"][keep]
        dTk = dT[keep] * (1.0 + self.c_th * np.log(np.maximum(ton / TON_HI, 1.0)))
        d = cyc["n"][keep] / np.exp(self.log_life(dTk, Tmin[keep], ton, S))
        tot = float(d.sum())
        f_small = float(d[dTk < 26.0].sum() / tot) if tot > 0 else 0.0
        return tot / (kappa * (1.0 - self.c_seq * f_small))


if __name__ == "__main__":
    h = hernes()
    f = fit(h["dT"], h["ton"], h["I"], h["Nf"], h["run"])
    print("Hernes full fit: b =", np.round(f["b"], 3), "se", np.round(np.sqrt(np.diag(f["cov"])), 3),
          "resid sd", round(math.sqrt(f["s2"]), 4))
    rng = np.random.default_rng(1)
    w = World.draw(rng, f["b"])
    S = w.strengths(rng, 200000)
    print(f"strength law: sd {S.std():.4f} (Hernes leave-one-out 0.127), 1 % {np.quantile(S, 0.01):+.3f}, "
          f"10 % {np.quantile(S, 0.1):+.3f}, share of second mechanism {np.mean(S < -rng.gumbel(0, 0.12, S.size)):.2f}")
    # one cycle: 40 K, 200 s, T_min 45 C against the model with the default margin
    cyc = dict(dTheta=np.array([40.0]), theta_min=np.array([25.0]), ton=np.array([200.0]), n=np.array([1.0]),
               T_amb=np.array([20.0]))
    for st in ("none", "pointwise", "ellipsoid"):
        mg = Margin(stat=st)
        print(f"margined damage of one 40 K, 200 s cycle, stat {st:9s}: {model_damage(cyc, f, mg):.3e}")
    print(f"true damage, median device, world draw: {w.true_damage(cyc, 0.0, 1.0, 1.0):.3e}")
