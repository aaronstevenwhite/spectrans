"""Layer configuration modules.

This package contains configuration models for all layer types in spectrans.

Modules
-------
mixing : Configuration models for mixing layers
attention : Configuration models for attention layers
operators : Configuration models for operator layers
"""

from .attention import (
    DCTAttentionConfig,
    HadamardAttentionConfig,
    LSTAttentionConfig,
    MixedTransformAttentionConfig,
    SpectralAttentionConfig,
    SpectralKernelAttentionConfig,
)
from .mixing import (
    AFNOMixingConfig,
    FourierMixingConfig,
    GlobalFilterMixingConfig,
    WaveletMixing2DConfig,
    WaveletMixingConfig,
)

# ruff: noqa: RUF022
__all__ = [
    # Mixing configurations
    "AFNOMixingConfig",
    "FourierMixingConfig",
    "GlobalFilterMixingConfig",
    "WaveletMixing2DConfig",
    "WaveletMixingConfig",
    # Attention configurations
    "DCTAttentionConfig",
    "HadamardAttentionConfig",
    "LSTAttentionConfig",
    "MixedTransformAttentionConfig",
    "SpectralAttentionConfig",
    "SpectralKernelAttentionConfig",
]
