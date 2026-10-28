"""Paired and union certificates of the end-of-interval damage (Section 2.4 of the manuscript, Eqs. 1 and 2).

Inputs
  S      log-strengths of bench modules, S_i = log N_obs,i - log N_mod(x_i), at the reference conditions
  V      margined model damage V(Z; policy) of recorded intervals, re-simulated under the policy being certified
  D0     margined model damage accumulated by the field device up to the decision instant
The lifetime model, the margins and the re-simulation are the caller's: see lifetime.py and policy.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .conformal import conformal_lower, conformal_upper, lower_rank


@dataclass(frozen=True)
class Certificate:
    value: float        # C: bound on the damage at the end of the coming interval (inf if no finite bound)
    reserve: float      # R_c: certified reserve, in units of the coming interval (0 if none)
    n: int              # number of pairs (paired) or of bench scores (union)

    @property
    def passes(self) -> bool:
        """The certificate passes when the damage bound does not exceed the budget 1 (equivalently R_c >= 1)."""
        return self.value <= 1.0


def paired_certificate(S: Sequence[float], V: Sequence[float], D0: float = 0.0, alpha: float = 0.10) -> Certificate:
    """Paired certificate: U_i = exp(-S_i)(D0 + V_i), C_P the ceil((1 - alpha)(m + 1))-th smallest U_i.

    S[i] and V[i] must be paired by a rule that does not look at the data (e.g. the first m of each in a preset order).
    The reserve is the floor(alpha (m + 1))-th smallest k_i = (exp(S_i) - D0) / V_i, clipped at 0.
    """
    S, V = np.asarray(S, dtype=float), np.asarray(V, dtype=float)
    if S.shape != V.shape or S.ndim != 1:
        raise ValueError("S and V must be one-dimensional and of the same length (one pair per index)")
    if np.any(V <= 0):
        raise ValueError("margined interval damages V must be positive")
    U = np.exp(-S) * (D0 + V)
    k = np.sort((np.exp(S) - D0) / V)
    r = lower_rank(S.size, alpha)
    reserve = 0.0 if r < 1 else max(0.0, float(k[r - 1]))
    return Certificate(value=conformal_upper(U, alpha), reserve=reserve, n=int(S.size))


def union_certificate(S: Sequence[float], V: Sequence[float], D0: float = 0.0, alpha_S: float = 0.05,
                      alpha_V: float = 0.05) -> Certificate:
    """Union certificate: C_U = exp(-q_S)(D0 + q_V), the strength and the mission bounded separately.

    q_S is the conformal lower bound of the bench scores at alpha_S, q_V the conformal upper bound of the recorded
    damages at alpha_V; the miscoverage of the certificate is at most alpha_S + alpha_V (union bound).
    """
    q_S = conformal_lower(S, alpha_S)
    q_V = conformal_upper(V, alpha_V)
    if q_S == -math.inf or q_V == math.inf:
        return Certificate(value=math.inf, reserve=0.0, n=len(S))
    w = math.exp(-q_S)
    return Certificate(value=w * (D0 + q_V), reserve=max(0.0, (1.0 - w * D0) / (w * q_V)), n=len(S))
