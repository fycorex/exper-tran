import json
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from common import load_experiment, pair_specs  # noqa: E402
from scale50_helpers import batch_slices, conditional_hits  # noqa: E402
from screen_transitions import predictions_complete  # noqa: E402

from primary_ml_cka.config.loader import load_config  # noqa: E402


class Scale50Test(unittest.TestCase):
    def setUp(self):
        self.raw = load_experiment(EXPERIMENT_ROOT / "config" / "scale50.yaml")

    def test_scale_and_fixed_family_settings(self):
        self.assertEqual(self.raw["attack_count"], 50)
        self.assertGreaterEqual(self.raw["candidate_count"], 50)
        self.assertTrue(self.raw["common_across_pairs"])
        specs = pair_specs(self.raw)
        self.assertEqual(specs["P14"]["selected_rho"], 0.5)
        self.assertEqual(specs["P14"]["source_logit_weight"], 0.75)
        self.assertEqual(specs["P16"]["selected_rho"], 1.0)
        self.assertEqual(specs["P19"]["selected_rho"], 0.5)

    def test_exact_partial_batch_plan(self):
        self.assertEqual(
            batch_slices(50),
            ((0, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 48), (48, 50)),
        )

    def test_conditional_transfer_counts_only_proxy_hits(self):
        self.assertEqual(
            conditional_hits(
                ((True, False, True), (True,)),
                ((True, True, False), (True,)),
            ),
            (2, 3),
        )

    def test_resume_requires_exact_candidate_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps({"image_id": "a", "parsed_label": 1}) + "\n",
                encoding="utf-8",
            )
            self.assertTrue(predictions_complete(path, {"a"}))
            self.assertFalse(predictions_complete(path, {"a", "b"}))

    def test_p16_reserve_search_weakens_push_independently(self):
        raw = load_config(EXPERIMENT_ROOT / "config" / "p16_reserve_search.yaml")
        arms = {arm["name"]: arm for arm in raw["arms"]}
        self.assertEqual(len(arms), 5)
        self.assertEqual(arms["p16_reserve_push025"]["source_logit_weight"], 0.25)
        self.assertEqual(arms["p16_reserve_push050"]["source_logit_weight"], 0.5)
        self.assertTrue(all(arm["target_logit_weight"] == 1.0 for arm in arms.values()))

    def test_extended_search_keeps_loss_families_explicit(self):
        raw = load_config(EXPERIMENT_ROOT / "config" / "extended_reserve_search.yaml")
        arms = {arm["name"]: arm for arm in raw["arms"]}
        self.assertEqual(len(arms), 24)
        self.assertEqual(arms["p16_ext_push010"]["semantic_mode"], "prototype")
        self.assertEqual(arms["p16_ext_push010"]["source_logit_weight"], 0.1)
        self.assertEqual(
            arms["multi_ext_rho050"]["semantic_mode"],
            "multiclass_prototype",
        )
        self.assertTrue(all(arm["steps"] == 50 for arm in arms.values()))

    def test_optimization_diagnostics_align_checkpoints_with_budget(self):
        raw = load_config(EXPERIMENT_ROOT / "config" / "optimization_diagnostic.yaml")
        arms = {arm["name"]: arm for arm in raw["arms"]}
        self.assertEqual(len(arms), 6)
        for arm in arms.values():
            self.assertEqual(arm["checkpoint_steps"][-1], arm["steps"])
            self.assertLess(arm["gradient_trace_steps"][-1], arm["steps"])
            self.assertTrue(
                all(0 < step <= arm["steps"] for step in arm["checkpoint_steps"])
            )

    def test_layer_gate_seed_search_changes_one_axis_at_a_time(self):
        raw = load_config(EXPERIMENT_ROOT / "config" / "layer_gate_seed.yaml")
        arms = {arm["name"]: arm for arm in raw["arms"]}
        self.assertEqual(len(arms), 9)
        self.assertEqual(arms["p16_layer11_standard"]["representation_layer"], 11)
        self.assertEqual(arms["p16_layer23_standard"]["representation_layer"], 23)
        self.assertTrue(arms["p16_layer17_gate_small"]["early_stop_proxy_gate"])
        self.assertEqual(arms["p19_layer15_small_seed43"]["seed"], 43)
        self.assertEqual(arms["p19_layer15_small_seed44"]["seed"], 44)

    def test_tuned_scale50_is_isolated_and_uses_frozen_reserve_settings(self):
        raw = load_experiment(EXPERIMENT_ROOT / "config" / "scale50_tuned.yaml")
        specs = pair_specs(raw)
        self.assertEqual(raw["state_namespace"], "states_scale50_tuned")
        self.assertEqual(raw["objective_tag"], "tuned_pull_push")
        self.assertEqual(raw["summary_filename"], "scale50_tuned_results.csv")
        self.assertEqual(specs["P14"]["semantic_temperature"], 0.2)
        self.assertEqual(specs["P14"]["source_logit_weight"], 0.5)
        self.assertEqual(specs["P16"]["source_logit_weight"], 0.25)
        self.assertEqual(specs["P19"]["source_logit_weight"], 0.5)
        self.assertEqual(
            tuple(specs[pair]["representation_layer"] for pair in specs),
            (17, 15, 17),
        )


if __name__ == "__main__":
    unittest.main()
