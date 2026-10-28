"""Lifetime model, log-strengths and margined damage (Sections 2.1 and 2.3).

The default model is log-linear in the swing dT, the heating time t_on and the current I (as fitted to Hernes et al.,
2021): log N_f = b0 + b1 log dT + b2 log t_on + b3 log I. Any other feature map can be passed to fit_lifetime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def loglinear_features(dT, t_on, I) -> np.ndarray:
    """Design matrix [1, log dT, log t_on, log I]."""
    dT, t_on, I = (np.asarray(v, dtype=float) for v in (dT, t_on, I))
    return np.column_stack([np.ones(dT.size), np.log(dT), np.log(t_on), np.log(I)])


@dataclass(frozen=True)
class LifetimeModel:
    coef: np.ndarray
    cov: np.ndarray                # covariance of the coefficients (least squares)
    dof: int
    features: Callable[..., np.ndarray] = loglinear_features

    def log_life(self, *x) -> np.ndarray:
        """Model log-life log N_mod at the conditions x (the arguments of the feature map)."""
        return self.features(*x) @ self.coef


def fit_lifetime(logN: Sequence[float], *x, features: Callable[..., np.ndarray] = loglinear_features) -> LifetimeModel:
    """Least-squares fit of log N_f on the features of the tested conditions x."""
    X = features(*x)
    y = np.asarray(logN, dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    dof = max(len(y) - X.shape[1], 1)
    cov = float(res @ res / dof) * np.linalg.inv(X.T @ X)
    return LifetimeModel(coef=coef, cov=cov, dof=dof, features=features)


def log_strength(model: LifetimeModel, N_obs: Sequence[float], *x) -> np.ndarray:
    """S_i = log N_obs,i - log N_mod(x_i): the score that the conformal ranks act on.

    Scores must come from modules that were not used to fit the model (split the bench first; see holdout.py).
    """
    return np.log(np.asarray(N_obs, dtype=float)) - model.log_life(*x)


def margined_cycle_life(N_mod: np.ndarray, eps_ex: np.ndarray | float, eps_M: float = 0.0) -> np.ndarray:
    """Cycle life reduced by the stated margins: N_mod exp(-eps_ex(x) - eps_M) (eps_M = -log kappa_M)."""
    return np.asarray(N_mod, dtype=float) * np.exp(-np.asarray(eps_ex, dtype=float) - eps_M)


def miner_damage(counts: Sequence[float], lives: Sequence[float]) -> float:
    """Miner sum of rainflow-counted cycles: sum of count / cycle life at the cycle's condition."""
    return float(np.sum(np.asarray(counts, dtype=float) / np.asarray(lives, dtype=float)))
