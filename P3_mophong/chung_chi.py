"""P3 case study (G5): certificates of the end-of-interval damage, U1-U9 (U10, the oracle, is computed from the truth
by the experiment driver).

Inputs common to all: bench log-strengths S (scores, residuals of the fitted model at tested conditions), margined
model damages V of recorded intervals under the deployed policy (re-simulated), the margined damage D0 accumulated by
the field device, and the level alpha. For a module, V and D0 are per chip and the certificate takes the largest chip:
U = exp(-S) max_k (D0_k + V_k).

  U1 box            exp(-min S) (D0 + max V): the envelope of the data, no quantile.
  U2 Monte Carlo    Weibull lives (Gumbel log-strength) fitted to S by maximum likelihood; S drawn from the fit, V
                    from the recorded intervals; the (1 - alpha) quantile of U (mission-profile practice, Reigosa 2016).
  U3 Bayes          Gumbel log-strength with prior 1 / sigma on a grid; posterior predictive S; V from the recorded
                    intervals; the (1 - alpha) predictive quantile of U.
  U4 split          paired certificate with V computed under the baseline policy, ignoring feedback.
  U5 union          Definition P3.6 with alpha_S = alpha_V = alpha / 2, V re-simulated under the deployed policy.
  U6 paired         Definition P3.6, m pairs, V re-simulated.
  U7 weighted       paired scores with weights w(X) on the loading summary (Proposition P3.20); infinite when the test
                    weight exceeds alpha (Proposition P3.22).
  U8 adaptive       paired scores at a level alpha_t updated by adaptive conformal inference (Gibbs and Candes 2021)
                    from observed miscoverage; in the simulation the indicator is given by the truth, which no field
                    deployment observes.
  U9 PAC            training-conditional paired certificate (Proposition P3.14) with confidence delta.
"""
import math
import site
import sys

sys.path.append(site.getusersitepackages())
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402


def kth(x, k):
    """k-th smallest of x (1-based); +inf if k exceeds the sample, -inf if k < 1."""
    x = np.sort(np.asarray(x, float))
    if k > x.size:
        return math.inf
    if k < 1:
        return -math.inf
    return float(x[k - 1])


def paired_scores(S, V, D0):
    """U_i = exp(-S_i) max_k (D0_k + V_ik); V is [m] or [m, chips], D0 scalar or [chips]."""
    V = np.atleast_2d(np.asarray(V, float).T).T if np.ndim(V) == 1 else np.asarray(V, float)
    return np.exp(-np.asarray(S, float)) * (np.asarray(D0, float) + V).max(1)


def u6_paired(S, V, D0, alpha):
    U = paired_scores(S, V, D0)
    return kth(U, math.ceil((1 - alpha) * (U.size + 1) - 1e-12))


def u5_union(S, V, D0, alpha):
    a = alpha / 2
    qS = kth(S, math.floor(a * (len(S) + 1) + 1e-12))
    Vm = np.asarray(V, float)
    Vm = Vm if Vm.ndim == 1 else None
    if Vm is None:
        # module: the union bound on the largest chip of D0 + V
        tot = (np.asarray(D0, float) + np.asarray(V, float)).max(1)
        qT = kth(tot, math.ceil((1 - a) * (tot.size + 1) - 1e-12))
        return math.exp(-qS) * qT if math.isfinite(qS) and math.isfinite(qT) else math.inf
    qV = kth(Vm, math.ceil((1 - a) * (Vm.size + 1) - 1e-12))
    return math.exp(-qS) * (float(np.max(D0)) + qV) if math.isfinite(qS) and math.isfinite(qV) else math.inf


def u5_union_split(S, V, D0, a_S, a_V):
    """Union certificate with the levels allocated in advance as alpha_S + alpha_V (u5_union: alpha/2 each); added
    19/10/2026 to compare the paired certificate with a union certificate whose mission level is small."""
    qS = kth(S, math.floor(a_S * (len(S) + 1) + 1e-12))
    tot = (np.asarray(D0, float) + np.asarray(V, float))
    tot = tot.max(1) if tot.ndim == 2 else tot
    qT = kth(tot, math.ceil((1 - a_V) * (tot.size + 1) - 1e-12))
    return math.exp(-qS) * qT if math.isfinite(qS) and math.isfinite(qT) else math.inf


def u9_pac(S, V, D0, alpha, delta):
    U = np.sort(paired_scores(S, V, D0))
    m = U.size
    for k in range(1, m + 1):
        if stats.binom.sf(k - 1, m, 1 - alpha) <= delta:      # P(Bin(m, 1 - alpha) >= k) <= delta
            return float(U[k - 1])
    return math.inf


def u1_box(S, V, D0):
    tot = (np.asarray(D0, float) + np.atleast_2d(np.asarray(V, float).T).T).max(1) if np.ndim(V) == 1 else \
        (np.asarray(D0, float) + np.asarray(V, float)).max(1)
    return math.exp(-float(np.min(S))) * float(tot.max())


def _tot(V, D0):
    V = np.asarray(V, float)
    return (np.asarray(D0, float) + (V[:, None] if V.ndim == 1 else V)).max(1)


def u2_montecarlo(S, V, D0, alpha, rng, nmc=20000):
    loc, scale = stats.gumbel_l.fit(np.asarray(S, float))
    Sd = stats.gumbel_l.rvs(loc, scale, size=nmc, random_state=rng)
    T = _tot(V, D0)
    return float(np.quantile(np.exp(-Sd) * rng.choice(T, nmc), 1 - alpha))


def u3_bayes(S, V, D0, alpha, rng, nmc=20000, ngrid=60):
    S = np.asarray(S, float)
    mu0, sd = S.mean(), S.std(ddof=1)
    mus = np.linspace(mu0 - 4 * sd, mu0 + 4 * sd, ngrid)
    sig = np.linspace(0.2 * sd, 3 * sd, ngrid)
    M, G = np.meshgrid(mus, sig, indexing="ij")
    ll = stats.gumbel_l.logpdf(S[None, None, :], M[..., None], G[..., None]).sum(-1) - np.log(G)
    w = np.exp(ll - ll.max()).ravel()
    w /= w.sum()
    idx = rng.choice(w.size, nmc, p=w)
    Sd = stats.gumbel_l.rvs(M.ravel()[idx], G.ravel()[idx], random_state=rng)
    T = _tot(V, D0)
    return float(np.quantile(np.exp(-Sd) * rng.choice(T, nmc), 1 - alpha))


def weighted_quantile(scores, w_cal, w_test, alpha):
    """(1 - alpha) quantile of sum_i p_i delta_{score_i} + p_{n+1} delta_{+inf} (Tibshirani et al. 2019)."""
    o = np.argsort(scores)
    s, w = np.asarray(scores, float)[o], np.asarray(w_cal, float)[o]
    tot = w.sum() + w_test
    cum = np.cumsum(w) / tot
    k = np.searchsorted(cum, 1 - alpha - 1e-12)
    return float(s[k]) if k < s.size else math.inf


def u7_weighted_paired(S, V, D0, alpha, w_cal, w_test):
    """Weighted paired certificate; also returns the test weight p_{n+1}, whose excess over alpha makes it infinite."""
    U = paired_scores(S, V, D0)
    p_test = w_test / (np.sum(w_cal) + w_test)
    return weighted_quantile(U, w_cal, w_test, alpha), float(p_test)


class ACI:
    """Adaptive conformal inference on the paired scores: alpha_{t+1} = alpha_t + eta (alpha - err_t)."""

    def __init__(self, alpha, eta=0.01):
        self.alpha, self.eta, self.a = alpha, eta, alpha

    def certificate(self, S, V, D0):
        U = paired_scores(S, V, D0)
        a = min(max(self.a, 1e-6), 1 - 1e-6)
        return kth(U, math.ceil((1 - a) * (U.size + 1) - 1e-12))

    def update(self, miss):
        self.a += self.eta * (self.alpha - float(miss))


def kde_ratio(x_cal, x_tgt_sample, bw_cal=None, bw_tgt=None):
    """Density ratio w(x) = p_target(x) / p_bench(x) from Gaussian kernel estimates; returns a function."""
    x_cal, x_t = np.asarray(x_cal, float), np.asarray(x_tgt_sample, float)
    sil = lambda x: 1.06 * max(np.std(x), 1e-3) * x.size ** (-0.2)
    hc, ht = bw_cal or sil(x_cal), bw_tgt or sil(x_t)

    def w(x):
        x = np.atleast_1d(np.asarray(x, float))
        pb = stats.norm.pdf((x[:, None] - x_cal[None, :]) / hc).mean(1) / hc
        pt = stats.norm.pdf((x[:, None] - x_t[None, :]) / ht).mean(1) / ht
        return pt / np.maximum(pb, 1e-300)
    return w
