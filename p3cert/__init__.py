"""p3cert: conformal damage certificates for the fatigue damage of power modules.

Companion library of the manuscript "Conformal damage certificates for power module maintenance". It implements the certificates and the decision rule; the case-study simulation is in P3_mophong/.
"""
from .certificate import Certificate, paired_certificate, union_certificate
from .conformal import conformal_lower, conformal_upper, coverage_interval, lower_rank, min_scores, upper_rank
from .holdout import holdout_coverage, run_held_out
from .lifetime import (LifetimeModel, fit_lifetime, log_strength, loglinear_features, margined_cycle_life,
                       miner_damage)
from .policy import Decision, choose_level

__version__ = "0.1.0"
__all__ = ["Certificate", "Decision", "LifetimeModel", "choose_level", "conformal_lower", "conformal_upper",
           "coverage_interval", "fit_lifetime", "holdout_coverage", "log_strength", "loglinear_features",
           "lower_rank", "margined_cycle_life", "min_scores", "miner_damage", "paired_certificate", "run_held_out",
           "union_certificate", "upper_rank"]
