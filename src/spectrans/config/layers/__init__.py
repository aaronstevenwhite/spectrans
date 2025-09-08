r"""Layer configuration schemas for spectral transformers.

This module provides Pydantic configuration models for all layer types in spectrans,
enabling type-safe configuration of mixing layers, attention mechanisms, and neural
operators. Each configuration class validates parameters and provides sensible defaults
for production use.

Modules
-------
attention
    Configuration models for attention layers.
mixing
    Configuration models for mixing layers.
operators
    Configuration models for operator layers.

Classes
-------
AFNOMixingConfig
    Configuration for Adaptive Fourier Neural Operator mixing.
DCTAttentionConfig
    Configuration for DCT-based attention.
FourierMixingConfig
    Configuration for Fourier mixing layers.
GlobalFilterMixingConfig
    Configuration for global filter networks.
HadamardAttentionConfig
    Configuration for Hadamard attention.
LSTAttentionConfig
    Configuration for Linear Spectral Transform attention.
MixedTransformAttentionConfig
    Configuration for mixed transform attention.
SpectralAttentionConfig
    Configuration for spectral attention with RFF.
SpectralKernelAttentionConfig
    Configuration for kernel-based spectral attention.
WaveletMixing2DConfig
    Configuration for 2D wavelet mixing.
WaveletMixingConfig
    Configuration for 1D wavelet mixing.

Examples
--------
Configuring a Fourier mixing layer:

>>> from spectrans.config.layers import FourierMixingConfig
>>> 
>>> config = FourierMixingConfig(
...     hidden_dim=768,
...     dropout=0.1,
...     use_real_fft=True
... )
>>> print(config.model_dump())

Configuring spectral attention:

>>> from spectrans.config.layers import SpectralAttentionConfig
>>> 
>>> config = SpectralAttentionConfig(
...     hidden_dim=512,
...     num_heads=8,
...     num_features=256,
...     kernel_type="gaussian"
... )

Validation example:

>>> from spectrans.config.layers import GlobalFilterMixingConfig
>>> 
>>> # This will raise a validation error
>>> try:
...     config = GlobalFilterMixingConfig(
...         hidden_dim=-1,  # Invalid dimension
...         sequence_length=512
...     )
>>> except ValueError as e:
...     print(f"Validation error: {e}")

Notes
-----
**Configuration Validation:**

All configuration classes perform:
- Range validation (e.g., dimensions > 0)
- Type coercion where appropriate
- Default value assignment
- Cross-field validation where needed

**Common Parameters:**

- `hidden_dim`: Model hidden dimension
- `dropout`: Dropout probability (0.0-1.0)
- `bias`: Whether to include bias terms
- `activation`: Activation function type

**Layer-Specific Parameters:**

- Mixing layers: sequence_length, normalization
- Attention layers: num_heads, num_features, kernel_type
- Operator layers: modes, grid_size, lifting_dim

See Also
--------
[`spectrans.layers`][] : Actual layer implementations.
[`spectrans.config`][] : Parent configuration module.
[`spectrans.config.models`][] : Model configuration schemas.
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

# Public API - alphabetically sorted
__all__ = [
    "AFNOMixingConfig",
    "DCTAttentionConfig",
    "FourierMixingConfig",
    "GlobalFilterMixingConfig",
    "HadamardAttentionConfig",
    "LSTAttentionConfig",
    "MixedTransformAttentionConfig",
    "SpectralAttentionConfig",
    "SpectralKernelAttentionConfig",
    "WaveletMixing2DConfig",
    "WaveletMixingConfig",
]
