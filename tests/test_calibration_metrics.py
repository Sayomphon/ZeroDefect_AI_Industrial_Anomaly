from __future__ import annotations

import unittest

from zerodefect_ai.calibration import (
    calibrate_business_cost,
    calibrate_f1,
    calibrate_normal_quantile,
)
from zerodefect_ai.metrics import average_precision, binary_metrics, roc_auc


class CalibrationAndMetricsTest(unittest.TestCase):
    def test_perfect_ranking_metrics(self) -> None:
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        self.assertEqual(roc_auc(scores, labels), 1.0)
        self.assertEqual(average_precision(scores, labels), 1.0)
        metrics = binary_metrics(scores, labels, threshold=0.5)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["false_positives"], 0)

    def test_normal_quantile_returns_observed_threshold(self) -> None:
        result = calibrate_normal_quantile([0.1, 0.2, 0.3, 0.4], 0.75)
        self.assertEqual(result.threshold, 0.4)
        self.assertEqual(result.false_positives, 1)

    def test_f1_calibration(self) -> None:
        result = calibrate_f1([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
        self.assertEqual(result.threshold, 0.8)
        self.assertEqual(result.objective_value, 1.0)

    def test_high_escape_cost_prefers_recall(self) -> None:
        scores = [0.1, 0.4, 0.3, 0.5]
        labels = [0, 0, 1, 1]
        result = calibrate_business_cost(
            scores,
            labels,
            false_reject_cost=1.0,
            defect_escape_cost=100.0,
        )
        self.assertEqual(result.false_negatives, 0)


if __name__ == "__main__":
    unittest.main()
