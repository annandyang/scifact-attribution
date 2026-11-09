"""Metric layer: accuracy with abstention, per-class F1, ECE, bootstrap, McNemar."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact_attr.eval.bootstrap import bootstrap_ci, mcnemar, paired_delta_ci
from scifact_attr.eval.metrics import (accuracy, expected_calibration_error, macro_f1,
                                       per_class_f1, selective_accuracy, summarise)


class TestLabelMetrics(unittest.TestCase):
    def test_accuracy_counts_abstain_as_wrong(self):
        self.assertAlmostEqual(accuracy(["SUPPORT", "NEI"], ["SUPPORT", "ABSTAIN"]), 0.5)

    def test_perfect_predictions(self):
        gold = ["SUPPORT", "REFUTE", "NEI"]
        self.assertAlmostEqual(macro_f1(gold, gold), 1.0)
        self.assertEqual(set(per_class_f1(gold, gold).values()), {1.0})

    def test_absent_class_scores_zero(self):
        gold = ["SUPPORT", "SUPPORT"]
        pred = ["SUPPORT", "SUPPORT"]
        self.assertEqual(per_class_f1(gold, pred)["NEI"], 0.0)

    def test_abstentions_and_invalid_labels_count_as_false_negatives(self):
        self.assertEqual(per_class_f1(["NEI"] * 3, ["NEI", "ABSTAIN", "INVALID"])["NEI"], 0.5)

    def test_macro_f1_hand_computed(self):
        # SUPPORT: tp=1, fp=1, fn=0 -> F1 = 2/3
        # REFUTE, NEI: no true positives at all -> F1 = 0
        gold = ["SUPPORT", "NEI"]
        pred = ["SUPPORT", "SUPPORT"]
        self.assertAlmostEqual(macro_f1(gold, pred), (2 / 3) / 3, places=6)

    def test_selective_accuracy_and_coverage(self):
        gold = ["SUPPORT", "NEI", "REFUTE", "SUPPORT"]
        pred = ["SUPPORT", "ABSTAIN", "ABSTAIN", "REFUTE"]
        sel_acc, coverage = selective_accuracy(gold, pred)
        self.assertAlmostEqual(sel_acc, 0.5)
        self.assertAlmostEqual(coverage, 0.5)

    def test_all_abstained(self):
        sel_acc, coverage = selective_accuracy(["NEI"], ["ABSTAIN"])
        self.assertNotEqual(sel_acc, sel_acc)   # NaN
        self.assertEqual(coverage, 0.0)

    def test_invalid_outputs_are_not_answers(self):
        row = summarise(["SUPPORT", "NEI"], ["SUPPORT", "INVALID"])
        self.assertEqual(row["coverage"], 0.5)
        self.assertEqual(row["selective_accuracy"], 1.0)

    def test_ece_zero_when_calibrated(self):
        conf = [0.9] * 10
        correct = [True] * 9 + [False]
        self.assertAlmostEqual(expected_calibration_error(conf, correct), 0.0, places=6)

    def test_ece_detects_overconfidence(self):
        conf = [0.95] * 10
        correct = [True] * 5 + [False] * 5
        ece = expected_calibration_error(conf, correct)
        self.assertAlmostEqual(ece, 0.45, places=6)

    def test_ece_empty(self):
        self.assertNotEqual(expected_calibration_error([], []), expected_calibration_error([], []))

    def test_summary_keys(self):
        row = summarise(["SUPPORT", "NEI"], ["SUPPORT", "ABSTAIN"], confidences=[0.9, 0.4])
        for key in ("accuracy", "macro_f1", "nei_f1", "coverage", "selective_accuracy", "ece"):
            self.assertIn(key, row)


class TestBootstrap(unittest.TestCase):
    def test_interval_brackets_mean(self):
        values = [1.0] * 30 + [0.0] * 70
        low, high = bootstrap_ci(values, n_resamples=2000, seed=1)
        self.assertLess(low, 0.30)
        self.assertGreater(high, 0.30)

    def test_constant_input_collapses(self):
        low, high = bootstrap_ci([1.0] * 20, n_resamples=500)
        self.assertAlmostEqual(low, 1.0)
        self.assertAlmostEqual(high, 1.0)

    def test_paired_delta_is_difference(self):
        gold = ["SUPPORT"] * 100
        pred_a = ["SUPPORT"] * 40 + ["NEI"] * 60
        pred_b = ["SUPPORT"] * 60 + ["NEI"] * 40
        result = paired_delta_ci(pred_a, pred_b, gold, n_resamples=1000, seed=3)
        self.assertAlmostEqual(result["delta"], 0.20, places=6)
        self.assertLess(result["ci_low"], result["delta"])
        self.assertGreater(result["ci_high"], result["delta"])


class TestMcNemar(unittest.TestCase):
    def test_identical_systems_are_not_different(self):
        gold = ["SUPPORT"] * 10
        result = mcnemar(gold, gold, gold)
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["n_discordant"], 0.0)

    def test_counts_discordant_pairs(self):
        gold = ["SUPPORT"] * 10
        a = ["SUPPORT"] * 10
        b = ["SUPPORT"] * 9 + ["NEI"]
        result = mcnemar(a, b, gold)
        self.assertEqual(result["b"], 1.0)    # a right, b wrong
        self.assertEqual(result["c"], 0.0)    # b right, a wrong
        self.assertEqual(result["n_discordant"], 1.0)


if __name__ == "__main__":
    unittest.main()
