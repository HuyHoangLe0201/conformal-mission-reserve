"""Conformal ranks used by the certificates (finite-sample, distribution-free).

Given n exchangeable scores, the upper bound of level alpha is the ceil((1 - alpha)(n + 1))-th smallest score
(+inf if that rank exceeds n) and the lower bound the floor(alpha (n + 1))-th smallest (-inf if the rank is 0). A new
score exchangeable with them is not above the upper (not below the lower) bound with probability at least 1 - alpha,
and at most 1 - alpha + 1 / (n + 1) when ties have probability zero.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _clean(x: float) -> float:
    """Round away float noise so that, e.g., 0.9 * (9 + 1) is 9 and not 9.000000000000002."""
    return round(x, 10)


def upper_rank(n: int, alpha: float) -> int:
    """Rank (1-based) of the conformal upper bound from n scores."""
    return math.ceil(_clean((1.0 - alpha) * (n + 1)))


def lower_rank(n: int, alpha: float) -> int:
    """Rank (1-based) of the conformal lower bound from n scores; 0 means the bound is -inf."""
    return math.floor(_clean(alpha * (n + 1)))


def conformal_upper(scores: Sequence[float], alpha: float) -> float:
    """Conformal upper bound of level alpha: +inf if the rank exceeds the number of scores."""
    s = np.sort(np.asarray(scores, dtype=float))
    k = upper_rank(s.size, alpha)
    return math.inf if k > s.size else float(s[k - 1])


def conformal_lower(scores: Sequence[float], alpha: float) -> float:
    """Conformal lower bound of level alpha: -inf if the rank is 0."""
    s = np.sort(np.asarray(scores, dtype=float))
    k = lower_rank(s.size, alpha)
    return -math.inf if k < 1 else float(s[k - 1])


def coverage_interval(n: int, alpha: float) -> tuple[float, float]:
    """Guaranteed coverage (lower bound) and its upper limit when ties have probability zero."""
    k = upper_rank(n, alpha)
    return (1.0 - alpha, (k / (n + 1)) if k <= n else 1.0)


def min_scores(alpha: float) -> int:
    """Smallest number of scores for which the upper bound is finite: ceil((1 - alpha)(n + 1)) <= n."""
    return math.ceil(_clean(1.0 / alpha - 1.0))
