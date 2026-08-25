import pytest

from primary_ml_cka.models.representations import RepresentationSpec, resolve_vision_layer


def test_resolves_final_and_explicit_vision_layers() -> None:
    assert resolve_vision_layer(-1, 24) == 23
    assert resolve_vision_layer(12, 24) == 12


@pytest.mark.parametrize("layer", [-2, 24])
def test_rejects_out_of_range_vision_layer(layer: int) -> None:
    with pytest.raises(ValueError):
        resolve_vision_layer(layer, 24)


def test_representation_component_and_pooling_are_validated() -> None:
    RepresentationSpec("vision_encoder", -1, "mean").validate()
    with pytest.raises(ValueError):
        RepresentationSpec("llm_backbone", -1, "mean").validate()
    with pytest.raises(ValueError):
        RepresentationSpec("vision_encoder", -1, "max").validate()
