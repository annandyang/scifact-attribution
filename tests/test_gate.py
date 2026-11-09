"""Conformal calibration and the risk-coverage curve.

The finite-sample level is checked against a hand-computed value, because the whole
guarantee rests on that formula: an off-by-one in the ceiling produces a threshold that
looks plausible and quietly under-covers.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.gate.conformal import (apply_to_predictions, conformal_quantile_level,
                                         conformal_threshold, coverage_and_error, decide,
                                         nonconformity)
from scifact_attr.gate.risk_coverage import (aurc, coverage_at_risk, risk_coverage_curve,
                                              summarise_gate)


class TestConformal(unittest.TestCase):
    def test_quantile_level_hand_computed(self):
        # n = 300, alpha = 0.10 -> ceil(301 * 0.90)/300 = ceil(270.9)/300 = 271/300
        self.assertAlmostEqual(conformal_quantile_level(300, 0.10), 271 / 300, places=9)

    def test_quantile_level_small_sample(self):
        self.assertAlmostEqual(conformal_quantile_level(9, 0.10), 9 / 9, places=9)

    def test_impossible_level_is_capped(self):
        # n = 5, alpha = 0.01 -> ceil(6 * 0.99) = 6 -> 6/5 = 1.2, capped at 1.0
        self.assertEqual(conformal_quantile_level(5, 0.01), 1.0)

    def test_threshold_is_monotone_in_alpha(self):
        scores = np.linspace(0, 1, 301)[1:]
        thresholds = [conformal_threshold(scores, a) for a in (0.05, 0.10, 0.20, 0.30)]
        self.assertTrue(all(a >= b for a, b in zip(thresholds, thresholds[1:])))

    def test_threshold_covers_all_labels_when_rank_exceeds_sample_size(self):
        scores = [0.1, 0.5, 0.7, 0.8, 0.9]
        self.assertEqual(conformal_threshold(scores, 0.01), 1.0)

    def test_threshold_is_exact_order_statistic(self):
        self.assertEqual(conformal_threshold([0.1, 0.2, 0.3, 0.4], 0.6), 0.2)

    def test_calibration_uses_gold_and_gate_requires_agreeing_singleton(self):
        from scifact_attr.gate.conformal import gold_score, gate_prediction

        row = {"gold_label": "REFUTE", "pred_label": "SUPPORT", "status": "ok",
               "p_support": 0.8, "p_refute": 0.1, "p_nei": 0.1}
        self.assertAlmostEqual(gold_score(row), 0.9)
        self.assertEqual(gate_prediction(row, 0.5), ("SUPPORT", ["SUPPORT"]))
        self.assertEqual(gate_prediction(row, 0.95), ("ABSTAIN", ["SUPPORT", "REFUTE", "NEI"]))
        self.assertEqual(gate_prediction(dict(row, pred_label="REFUTE"), 0.5),
                         ("ABSTAIN", ["SUPPORT"]))
        self.assertEqual(gate_prediction(row, 0.1), ("ABSTAIN", []))

    def test_partition_is_stable_and_has_no_overlap(self):
        from scifact_attr.gate.conformal import partition_claim_ids

        calibration, evaluation = partition_claim_ids(["3", "2", "1", "4"])
        self.assertEqual((calibration, evaluation), partition_claim_ids(["1", "2", "3", "4"]))
        self.assertEqual(len(calibration), 2)
        self.assertFalse(calibration & evaluation)

    def test_threshold_within_range(self):
        rng = np.random.default_rng(0)
        scores = rng.random(300)
        threshold = conformal_threshold(scores, 0.10)
        self.assertGreaterEqual(threshold, scores.min())
        self.assertLessEqual(threshold, scores.max())

    def test_nonconformity_is_one_minus_confidence(self):
        self.assertAlmostEqual(nonconformity(0.8), 0.2, places=9)

    def test_decide_direction(self):
        self.assertEqual(decide([0.1, 0.9], threshold=0.5), ["answer", "ABSTAIN"])

    def test_apply_to_predictions_substitutes(self):
        out = apply_to_predictions(["SUPPORT", "REFUTE"], [0.9, 0.2], threshold=0.5)
        self.assertEqual(out, ["SUPPORT", "ABSTAIN"])

    def test_coverage_and_error_hand_computed(self):
        stats = coverage_and_error(["SUPPORT", "NEI", "REFUTE"], ["SUPPORT", "NEI", "SUPPORT"],
                                   [True, True, False])
        self.assertAlmostEqual(stats["coverage"], 2 / 3)
        self.assertEqual(stats["selective_error"], 0.0)

    def test_guarantee_holds_on_simulated_set_predictor(self):
        """The set-valued guarantee, checked once with a wide margin.

        True labels are drawn from the model's own distribution, which is the situation the
        guarantee describes. With 300 calibration points the empirical error should sit near
        alpha; the assertion allows a generous band because a single realisation is noisy.
        """
        rng = np.random.default_rng(7)
        probs = rng.dirichlet([2, 2, 2], size=1300)
        labels = np.array([rng.choice(3, p=p) for p in probs])

        cal_probs, cal_labels = probs[:300], labels[:300]
        test_probs, test_labels = probs[300:], labels[300:]
        alpha = 0.10

        cal_scores = 1.0 - cal_probs[np.arange(300), cal_labels]
        threshold = conformal_threshold(cal_scores, alpha)

        covered = test_probs[np.arange(len(test_labels)), test_labels] >= (1.0 - threshold)
        empirical_error = 1.0 - covered.mean()
        self.assertLess(empirical_error, alpha + 0.05,
                        f"empirical error {empirical_error:.3f} far above nominal {alpha}")


class TestRiskCoverage(unittest.TestCase):
    def test_curve_shape_and_monotone_coverage(self):
        coverage, risk = risk_coverage_curve([0.9, 0.8, 0.7, 0.6], [True, True, False, False])
        self.assertEqual(len(coverage), 4)
        self.assertTrue(np.all(np.diff(coverage) > 0))
        self.assertAlmostEqual(risk[0], 0.0)      # the most confident claim is correct

    def test_risk_end_to_end_is_error_rate(self):
        coverage, risk = risk_coverage_curve([0.9, 0.8, 0.7, 0.6], [True, True, False, False])
        self.assertAlmostEqual(risk[-1], 0.5)     # half wrong overall

    def test_perfect_ranking_has_zero_aurc(self):
        coverage, risk = risk_coverage_curve([0.9, 0.5], [True, True])
        self.assertAlmostEqual(aurc(coverage, risk), 0.0)

    def test_coverage_at_target_risk(self):
        coverage = np.array([0.25, 0.5, 0.75, 1.0])
        risk = np.array([0.0, 0.1, 0.3, 0.6])
        self.assertAlmostEqual(coverage_at_risk(risk, coverage, 0.1), 0.5)

    def test_summary_keys(self):
        summary = summarise_gate([0.9, 0.5, 0.4], [True, False, False])
        self.assertIn("aurc", summary)
        self.assertIn("coverage_at_target_risk", summary)


if __name__ == "__main__":
    unittest.main()
