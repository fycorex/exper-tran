import sys
import unittest
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from common import DEFAULT_CONFIG, load_experiment, pair_specs, transitions  # noqa: E402


class PrimaryConfigTest(unittest.TestCase):
    def setUp(self):
        self.raw = load_experiment(DEFAULT_CONFIG)

    def test_balanced_ten_transition_cycle(self):
        items = transitions(self.raw)
        self.assertEqual(len(items), 10)
        self.assertEqual(sorted(item.source for item in items), list(range(1, 11)))
        self.assertEqual(sorted(item.target for item in items), list(range(1, 11)))
        self.assertEqual(
            len({frozenset((item.source, item.target)) for item in items}),
            10,
        )

    def test_primary_pairs_are_small_to_large(self):
        self.assertEqual(set(pair_specs(self.raw)), {"P14", "P16", "P19"})

    def test_small_step_schedule_preserves_nominal_path_length(self):
        arms = {arm["name"]: arm for arm in self.raw["arms"]}
        for prefix in ("pull_push", "multiclass"):
            standard = arms[f"{prefix}_standard"]
            small = arms[f"{prefix}_small_steps"]
            self.assertAlmostEqual(
                standard["steps"] * standard["step_size"],
                small["steps"] * small["step_size"],
            )
            self.assertLess(small["step_size"], standard["step_size"])
            self.assertGreater(small["steps"], standard["steps"])

    def test_loss_modes_are_a_controlled_pair(self):
        arms = {arm["name"]: arm for arm in self.raw["arms"]}
        self.assertEqual(arms["pull_push_small_steps"]["semantic_mode"], "prototype")
        self.assertEqual(
            arms["multiclass_small_steps"]["semantic_mode"],
            "multiclass_prototype",
        )


if __name__ == "__main__":
    unittest.main()
