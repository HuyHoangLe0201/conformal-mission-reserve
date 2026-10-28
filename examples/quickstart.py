"""Quick start: from bench lives and recorded intervals to a replacement decision.

A synthetic bench and synthetic recorded intervals stand in for your data. Replace them with
  - the cycles to failure of bench modules of the field lot at the tested conditions,
  - the margined model damage of your recorded maintenance intervals, re-simulated under each derating level
    (rainflow-count the junction temperature, apply the lifetime model with the stated margins).
Run: python examples/quickstart.py    (from the repository root, after `pip install -e .`)
"""
import numpy as np

from p3cert import choose_level, fit_lifetime, log_strength, paired_certificate

rng = np.random.default_rng(0)

# 1. Bench: 36 modules at five tested conditions; the first half fits the lifetime model, the second gives the scores
dT = np.repeat([30.0, 40.0, 50.0, 60.0, 70.0], 8)[:36]
t_on = np.repeat([0.5, 1.0, 2.0, 4.0, 4.5], 8)[:36]
I = np.repeat([900.0, 1000.0, 1100.0, 1200.0, 1250.0], 8)[:36]
logN = 22 - 2.0 * np.log(dT) - 0.2 * np.log(t_on) - 0.3 * np.log(I) + rng.normal(0, 0.15, 36)
order = rng.permutation(36)
fit, score = order[:18], order[18:]
model = fit_lifetime(logN[fit], dT[fit], t_on[fit], I[fit])
S = log_strength(model, np.exp(logN[score]), dT[score], t_on[score], I[score])      # 18 log-strengths

# 2. Recorded intervals: margined damage of 18 intervals under each derating level (here a toy dependence on the level)
base = rng.uniform(0.020, 0.026, 18)


def certify(level):
    V = base * (1.0 - 0.8 * level)             # your re-simulation under this derating level goes here
    return paired_certificate(S, V, D0=0.0, alpha=0.10)


# 3. Decision: smallest derating whose certified reserve reaches the planned replacement at 60 intervals
decision = choose_level([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], certify, t_plan=60, t=0)
c = decision.certificate
print(f"action: {decision.action}, derating level: {decision.level}")
if c is not None:
    print(f"damage bound of the coming interval: {c.value:.4f} (budget 1); certified reserve: {c.reserve:.1f} intervals")
