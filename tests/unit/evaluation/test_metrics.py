from primary_ml_cka.evaluation.attack_metrics import attack_rates


def test_clean_conditioned_hit_counts_and_denominator() -> None:
    rates = attack_rates(
        (4, 4, 7, None),
        (9, 6, 9, 9),
        source_human_label=4,
        target_human_label=9,
    )
    assert rates.clean_valid_count == 2
    assert rates.targeted_hit_count == 1
    assert rates.untargeted_hit_count == 2
    assert rates.tasr_percent == 50
    assert rates.asr_percent == 100
