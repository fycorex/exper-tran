import sys
import unittest
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from common import (  # noqa: E402
    DEFAULT_CONFIG,
    class_names,
    class_specs,
    classification_prompt,
    load_experiment,
    pair_specs,
    transitions,
)


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

    def test_catalog_is_semantically_diverse(self):
        items = class_specs(self.raw)
        self.assertEqual(
            tuple(item["wnid"] for item in items),
            (
                "n01443537",
                "n02279972",
                "n07753275",
                "n02676566",
                "n03642806",
                "n07920052",
                "n09472597",
                "n04099969",
                "n04254680",
                "n04146614",
            ),
        )
        self.assertGreaterEqual(len({item["domain"] for item in items}), 8)
        self.assertIn("volcano", class_names(self.raw))
        self.assertIn("monarch butterfly", class_names(self.raw))

    def test_prompt_uses_diverse_catalog_and_zero_based_codes(self):
        prompt = classification_prompt(self.raw)
        for code, name in enumerate(class_names(self.raw)):
            self.assertIn(f"{code} {name}", prompt)
        self.assertIn("Return only one integer from 0 to 9", prompt)
        self.assertNotIn("pickup truck", prompt)

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
