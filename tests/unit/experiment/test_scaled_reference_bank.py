from primary_ml_cka.experiment.scaled import _reference_batch_index


def test_controlled_scale_uses_the_first_reference_bank_for_every_source_batch() -> None:
    assert _reference_batch_index(True) == 0


def test_legacy_scale_preserves_source_index_driven_reference_selection() -> None:
    assert _reference_batch_index(False) is None
