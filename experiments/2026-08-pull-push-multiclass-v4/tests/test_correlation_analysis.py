import math
import sys
import unittest
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from analyze_cka_correlations import (  # noqa: E402
    average_ranks,
    class_geometry,
    distance_percentile,
    pearson,
    prototype_distances,
    spearman,
    stratified_rank_correlation,
)

import torch  # noqa: E402


class CorrelationAnalysisTest(unittest.TestCase):
    def test_average_ranks_handle_ties(self):
        self.assertEqual(average_ranks([3.0, 1.0, 1.0, 2.0]), [4.0, 1.5, 1.5, 3.0])

    def test_correlations_have_expected_direction(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(spearman([1, 3, 2], [6, 2, 4]), -1.0)

    def test_prototype_distance_uses_normalized_class_centers(self):
        features = torch.tensor([[1.0, 0.0], [2.0, 0.0]] * 5 + [[0.0, 1.0], [0.0, 2.0]] * 5)
        distance = prototype_distances(features, 2)
        self.assertEqual(tuple(distance.shape), (10, 10))
        self.assertAlmostEqual(float(distance[0, 5]), 1.0)
        self.assertAlmostEqual(float(distance[0, 1]), 0.0)

    def test_class_geometry_and_distance_percentile_are_finite(self):
        features = torch.eye(20, dtype=torch.float32).repeat(24, 1)
        centers, rows = class_geometry(features, 48)
        self.assertEqual(tuple(centers.shape), (10, 20))
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(math.isfinite(row["effective_rank"]) for row in rows))
        distance = 1.0 - centers @ centers.T
        percentile = distance_percentile(distance, 0, 1)
        self.assertTrue(0.0 <= percentile <= 1.0)

    def test_stratified_correlation_removes_pair_level_offset(self):
        rows = []
        for pair, offset in (("A", 0), ("B", 100)):
            for value in range(4):
                rows.append(
                    {
                        "pair_id": pair,
                        "metric": value + offset,
                        "outcome": 3 - value + offset,
                    }
                )
        value, p_value = stratified_rank_correlation(
            rows,
            "metric",
            "outcome",
            permutations=100,
            seed=42,
        )
        self.assertAlmostEqual(value, -1.0)
        self.assertTrue(math.isfinite(p_value))

    def test_stratified_constant_metric_has_no_p_value(self):
        rows = []
        for pair in ("A", "B"):
            for value in range(4):
                rows.append(
                    {
                        "pair_id": pair,
                        "metric": 1.0,
                        "outcome": float(value),
                    }
                )
        value, p_value = stratified_rank_correlation(
            rows,
            "metric",
            "outcome",
            permutations=100,
            seed=42,
        )
        self.assertTrue(math.isnan(value))
        self.assertTrue(math.isnan(p_value))


if __name__ == "__main__":
    unittest.main()
