"""Layer configuration modules.

This package contains configuration models for all layer types in spectrans.

Modules
-------
mixing : Configuration models for mixing layers
attention : Configuration models for attention layers
operators : Configuration models for operator layers
"""

from .mixing import (
    FourierMixingConfig,
    GlobalFilterMixingConfig,
    WaveletMixing2DConfig,
    WaveletMixingConfig,
)

__all__ = [
    "FourierMixingConfig",
    "GlobalFilterMixingConfig",
    "WaveletMixing2DConfig",
    "WaveletMixingConfig",
]
