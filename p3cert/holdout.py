"""Hold-out check of the calibrated part on your own bench data (as in Section 7.1 and Table S31 of the manuscript).

The bench modules are split into disjoint fit, score and test sets: the model is fitted on the fit set, the conformal
lower bound of the log-strength is taken from the score set, and the share of test modules whose log-strength is not
below the bound is the coverage. The mean over random splits should be at least 1 - alpha when the modules are
exchangeable; with a whole condition withheld (run_held_out) it measures the extrapolation error of the model.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .conformal import conformal_lower


def _scores(X_fit, y_fit, X, y):
    beta, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    return y - X @ beta


def _ok(X) -> bool:
    return np.linalg.matrix_rank(X) >= X.shape[1] and np.linalg.cond(X) <= 1e6


def holdout_coverage(X: np.ndarray, y: np.ndarray, sizes: tuple[int, int, int], alphas: Sequence[float] = (0.1, 0.2),
                     reps: int = 4000, seed: int = 0, test_idx: np.ndarray | None = None,
                     pool_idx: np.ndarray | None = None) -> dict:
    """Mean coverage over random disjoint splits.

    X, y       design matrix of the tested conditions and log-lives of the bench modules
    sizes      (n_fit, n_score, n_test); the test set is drawn at random unless test_idx is given
    test_idx   fixed test modules (e.g. one whole condition); then the fit and score sets are random halves of the rest
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    cov = {a: [] for a in alphas}
    for _ in range(reps):
        if test_idx is None:
            p = rng.permutation(n)
            fit, score, test = p[:sizes[0]], p[sizes[0]:sizes[0] + sizes[1]], p[sizes[0] + sizes[1]:sum(sizes)]
        else:
            rest = np.setdiff1d(np.arange(n) if pool_idx is None else pool_idx, test_idx)
            p = rng.permutation(rest)
            fit, score, test = p[:len(p) // 2], p[len(p) // 2:], np.asarray(test_idx)
        if not _ok(X[fit]):
            continue
        s_score = _scores(X[fit], y[fit], X[score], y[score])
        s_test = _scores(X[fit], y[fit], X[test], y[test])
        for a in alphas:
            cov[a].append(float(np.mean(s_test >= conformal_lower(s_score, a))))
    return {a: {"mean_coverage": float(np.mean(c)), "p05": float(np.percentile(c, 5)), "p95": float(np.percentile(c, 95)),
                "splits": len(c)} for a, c in cov.items()}


def run_held_out(X: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas: Sequence[float] = (0.1, 0.2),
                 reps: int = 2000, seed: int = 0) -> dict:
    """Coverage with one whole group (condition, run, campaign) withheld, for every group."""
    return {g: holdout_coverage(X, y, (0, 0, 0), alphas, reps, seed, test_idx=np.where(groups == g)[0])
            for g in np.unique(groups)}
