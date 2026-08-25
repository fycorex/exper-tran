"""Explicit representation-selection contract for proxy image embeddings."""

from dataclasses import dataclass

REPRESENTATION_TYPES = ("legacy_projected", "vision_encoder")
POOLING_MODES = ("none", "mean")


@dataclass(frozen=True, slots=True)
class RepresentationSpec:
    representation_type: str = "legacy_projected"
    layer: int = -1
    pooling: str = "mean"

    def validate(self) -> None:
        if self.representation_type not in REPRESENTATION_TYPES:
            raise ValueError(f"Unknown representation_type: {self.representation_type}")
        if not isinstance(self.layer, int):
            raise TypeError("layer must be an integer")
        if self.pooling not in POOLING_MODES:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")
        if self.representation_type == "legacy_projected" and self.layer != -1:
            raise ValueError("legacy_projected only supports layer=-1")


def resolve_vision_layer(requested_layer: int, total_layers: int) -> int:
    if total_layers < 1:
        raise ValueError("Vision encoder must contain at least one layer")
    resolved = total_layers - 1 if requested_layer == -1 else requested_layer
    if not 0 <= resolved < total_layers:
        raise ValueError(f"Vision layer {requested_layer} resolves outside [0,{total_layers - 1}]")
    return resolved
