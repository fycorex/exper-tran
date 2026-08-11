from primary_ml_cka.experiment.scaled import _migrate_attack_result_payload


def test_legacy_attack_result_fields_are_migrated_without_fabricating_mask() -> None:
    migrated = _migrate_attack_result_payload(
        {
            "lambda_cka": 3.0,
            "source_image_ids": ["a", "b"],
            "target_reference_ids": ["r1", "r2"],
        }
    )

    assert migrated["effective_lambda_cka"] == 3.0
    assert migrated["gradient_ratio"] is None
    assert migrated["cka_source_weight"] == 1.0
    assert migrated["semantic_target_weight"] == 0.0
    assert migrated["proxy_target_hit_mask"] is None
    assert migrated["proxy_tap_path"] == "legacy-unrecorded"
    assert migrated["source_image_ids"] == ("a", "b")
    assert migrated["target_reference_ids"] == ("r1", "r2")


def test_current_attack_result_mask_is_preserved_as_tuple() -> None:
    migrated = _migrate_attack_result_payload(
        {
            "lambda_cka": 1.0,
            "source_image_ids": ["a", "b"],
            "target_reference_ids": ["r1", "r2"],
            "proxy_target_hit_mask": [True, False],
        }
    )

    assert migrated["proxy_target_hit_mask"] == (True, False)
