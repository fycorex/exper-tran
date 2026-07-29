"""Black-box target generation only; no target representation or gradient access."""

from .api import BlackBoxTargetAPI, DecodingConfig

__all__ = ["BlackBoxTargetAPI", "DecodingConfig"]
