"""Derating or replacement decision from the certificates (Algorithm 1 of the manuscript).

The caller supplies, for each derating level, the paired certificate computed with the recorded inputs re-simulated under
that level (policy data) -- see Section 4.1 for why the re-simulation, on data kept apart from the final calibration,
keeps the coverage when the controller acts on the certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .certificate import Certificate


@dataclass(frozen=True)
class Decision:
    action: str                  # "replace" or "run"
    level: Optional[float]       # derating level applied (None when the module is replaced)
    certificate: Optional[Certificate]
    reaches_plan: Optional[bool]


def choose_level(levels: Sequence[float], certify: Callable[[float], Certificate], t_plan: float, t: float) -> Decision:
    """Reserve rule: apply the passing level with the smallest derating whose reserve reaches the planned replacement,
    R_c >= t_plan - t, else the passing level with the largest reserve; replace the module if no level passes.

    levels   derating levels in increasing order (0 = none)
    certify  level -> paired certificate of the coming interval under that level
    t_plan   planned replacement age in intervals; t the intervals served so far
    """
    ordered = sorted(levels)
    results = [(lv, certify(lv)) for lv in ordered]
    passing = [(lv, c) for lv, c in results if c.passes]
    if not passing:
        return Decision("replace", None, None, None)
    need = t_plan - t
    for lv, c in passing:
        if c.reserve >= need:
            return Decision("run", lv, c, True)
    lv, c = max(passing, key=lambda lc: lc[1].reserve)
    return Decision("run", lv, c, False)
