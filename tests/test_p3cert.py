"""Tests of the p3cert library: ranks, certificates, decision rule, and agreement with the numbers of the manuscript.

Run from the repository root:  python -m unittest discover -s tests -v
"""
import csv
import json
import math
import unittest
from pathlib import Path

import numpy as np

from p3cert import (choose_level, conformal_lower, conformal_upper, coverage_interval, fit_lifetime, holdout_coverage,
                    loglinear_features, log_strength, min_scores, paired_certificate, run_held_out, union_certificate)
from p3cert.conformal import lower_rank, upper_rank

ROOT = Path(__file__).resolve().parent.parent


class TestRanks(unittest.TestCase):
    def test_ranks_and_float_noise(self):
        self.assertEqual(upper_rank(9, 0.1), 9)          # 0.9 * 10 is 9.000000000000002 in floats
        self.assertEqual(upper_rank(18, 0.1), 18)        # ceil(0.9 * 19) = ceil(17.1)
        self.assertEqual(lower_rank(19, 0.1), 2)
        self.assertEqual(lower_rank(8, 0.1), 0)

    def test_infinite_when_too_few_scores(self):
        self.assertEqual(conformal_upper([1.0] * 8, 0.1), math.inf)
        self.assertEqual(conformal_lower([1.0] * 8, 0.1), -math.inf)
        self.assertEqual(min_scores(0.1), 9)
        self.assertTrue(np.isfinite(conformal_upper(np.arange(9.0), 0.1)))

    def test_coverage_of_a_new_exchangeable_score(self):
        rng = np.random.default_rng(1)
        n, alpha, reps = 19, 0.1, 20000
        hit = 0
        for _ in range(reps):
            x = rng.normal(size=n + 1)
            hit += x[-1] <= conformal_upper(x[:-1], alpha)
        lo, hi = coverage_interval(n, alpha)
        self.assertGreaterEqual(hit / reps, lo - 0.01)
        self.assertLessEqual(hit / reps, hi + 0.01)


class TestCertificates(unittest.TestCase):
    def test_paired_matches_definition(self):
        S = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.05, 0.05, 0.15, -0.15])
        V = np.linspace(0.01, 0.02, 10)
        c = paired_certificate(S, V, D0=0.005, alpha=0.2)
        U = np.sort(np.exp(-S) * (0.005 + V))
        self.assertAlmostEqual(c.value, U[math.ceil(0.8 * 11) - 1])
        self.assertTrue(c.passes and c.reserve >= 1.0)

    def test_pass_equals_reserve_at_least_one_when_no_prior_damage(self):
        rng = np.random.default_rng(2)
        S, V = rng.normal(0, 0.2, 30), rng.uniform(0.02, 0.06, 30)
        c = paired_certificate(S, V, 0.0, 0.1)
        self.assertEqual(c.passes, c.reserve >= 1.0 - 1e-12)

    def test_union_infinite_without_enough_scores(self):
        c = union_certificate(np.zeros(5), np.ones(5) * 0.01, 0.0, 0.05, 0.05)
        self.assertEqual(c.value, math.inf)
        self.assertEqual(c.reserve, 0.0)

    def test_union_matches_definition(self):
        rng = np.random.default_rng(3)
        S, V = rng.normal(0, 0.2, 60), rng.uniform(0.01, 0.03, 60)
        c = union_certificate(S, V, 0.01, 0.05, 0.05)
        qS, qV = conformal_lower(S, 0.05), conformal_upper(V, 0.05)
        self.assertAlmostEqual(c.value, math.exp(-qS) * (0.01 + qV))


class TestDecision(unittest.TestCase):
    def test_reserve_rule(self):
        def certify(level):                      # more derating, more reserve
            return paired_certificate(np.zeros(20), np.full(20, 0.04 * (1 - level)), 0.0, 0.1)
        d = choose_level([0.0, 0.1, 0.2, 0.3], certify, t_plan=30, t=0)
        self.assertEqual(d.action, "run")
        self.assertTrue(d.reaches_plan)
        self.assertTrue(certify(d.level).reserve >= 30)
        self.assertTrue(d.level == 0.0 or certify(d.level - 0.1).reserve < 30)

    def test_replace_when_no_level_passes(self):
        d = choose_level([0.0, 0.5], lambda lv: paired_certificate(np.zeros(20), np.full(20, 5.0), 0.0, 0.1), 10, 0)
        self.assertEqual(d.action, "replace")


class TestManuscript(unittest.TestCase):
    def test_worked_example_of_section_7_1(self):
        path = ROOT / "P3_mophong" / "vi_du_so.json"
        if not path.exists():
            self.skipTest("vi_du_so.json not present")
        d = json.loads(path.read_text(encoding="utf-8"))
        U = np.array(d["paired_scores_sorted"])
        from p3cert.conformal import conformal_upper as cu
        self.assertAlmostEqual(cu(U, d["alpha"]), d["certificate"], places=12)
        self.assertAlmostEqual(1.0 / d["certificate"], d["reserve_intervals"], places=9)
        self.assertAlmostEqual(math.exp(-d["S_of_that_pair"]) * d["V_of_that_pair"], d["certificate"], places=12)

    def test_holdout_on_hernes_is_near_nominal(self):
        path = ROOT / "P3_kiemtoan" / "G2_hernes2021_sohoa.csv"
        if not path.exists():
            self.skipTest("digitized data not present")
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        X = loglinear_features([float(r["dT_C"]) for r in rows], [float(r["t_on_s"]) for r in rows],
                               [float(r["I_load_A"]) for r in rows])
        y = np.log([float(r["Nf"]) for r in rows])
        res = holdout_coverage(X, y, (12, 12, 10), alphas=(0.1,), reps=1500, seed=5)
        self.assertGreater(res[0.1]["mean_coverage"], 0.88)       # manuscript Table S31: 0.923 at alpha = 0.10
        runs = np.array([int(r["run"]) for r in rows])
        out = run_held_out(X, y, runs, alphas=(0.1,), reps=300, seed=5)
        self.assertEqual(sorted(out), [1, 2, 3, 4, 5])

    def test_fit_and_score_roundtrip(self):
        rng = np.random.default_rng(4)
        dT, ton, I = rng.uniform(20, 70, 40), rng.uniform(0.2, 5, 40), rng.uniform(800, 1250, 40)
        y = 20 - 2 * np.log(dT) - 0.1 * np.log(ton) - 0.3 * np.log(I) + rng.normal(0, 0.1, 40)
        m = fit_lifetime(y, dT, ton, I)
        S = log_strength(m, np.exp(y), dT, ton, I)
        self.assertLess(np.abs(S).max(), 0.5)


if __name__ == "__main__":
    unittest.main()
