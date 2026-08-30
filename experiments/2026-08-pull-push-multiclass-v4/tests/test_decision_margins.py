import math
import sys
import unittest
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from analyze_decision_margins import gap_closure, margin_values  # noqa: E402

import torch  # noqa: E402


class DecisionMarginTest(unittest.TestCase):
    def test_margin_values_use_human_labels(self):
        logits = torch.tensor([[1.0, 3.0, 2.0]])
        values = margin_values(logits, source_label=1, target_label=3)
        self.assertAlmostEqual(float(values["source_target_margin"][0]), 1.0)
        self.assertAlmostEqual(float(values["robust_margin"][0]), -1.0)
        self.assertEqual(float(values["target_rank"][0]), 2.0)

    def test_gap_closure_is_normalized_by_negative_clean_gap(self):
        self.assertAlmostEqual(gap_closure(-4.0, 2.0), 0.5)
        self.assertTrue(math.isnan(gap_closure(1.0, 2.0)))


if __name__ == "__main__":
    unittest.main()
