"""Base configuration models for spectrans components.

This module provides the foundational Pydantic models that define common
configuration patterns used across spectrans components. These base classes
are extended by specific component configuration models in submodules.

Classes
-------
BaseLayerConfig
    Base configuration for all neural network layers.
UnitaryLayerConfig
    Configuration for layers that preserve energy/unitarity.
FilterLayerConfig
    Configuration for layers using learnable spectral filters.
AttentionLayerConfig
    Configuration for attention-based layers.

Notes
-----
All configuration models use Pydantic v2 BaseModel for validation and type safety.
The base classes here provide common parameter patterns that are inherited by
specific component configurations.

Examples
--------
>>> from spectrans.config.core import BaseLayerConfig
>>> class MyLayerConfig(BaseLayerConfig):
...     special_param: int = 42
>>> config = MyLayerConfig(hidden_dim=768)
>>> print(config.hidden_dim)
768
"""

from pydantic import BaseModel, Field

from ..core.types import FFTNorm


class BaseLayerConfig(BaseModel):
    """Base configuration for all neural network layers.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension size, must be positive.
    dropout : float
        Dropout probability, must be between 0.0 and 1.0, defaults to 0.0.
    """

    hidden_dim: int = Field(gt=0, description="Hidden dimension size")
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout probability"
    )


class UnitaryLayerConfig(BaseLayerConfig):
    """Configuration for layers that preserve energy/unitarity.

    Parameters
    ----------
    norm_eps : float
        Epsilon for numerical stability in normalization, defaults to 1e-5.
    energy_tolerance : float
        Tolerance for energy preservation checks, defaults to 1e-4.
    fft_norm : FFTNorm
        FFT normalization mode, defaults to "ortho".
    """

    norm_eps: float = Field(default=1e-5, gt=0, description="Normalization epsilon")
    energy_tolerance: float = Field(default=1e-4, gt=0, description="Energy tolerance")
    fft_norm: FFTNorm = Field(default="ortho", description="FFT normalization")


class FilterLayerConfig(BaseLayerConfig):
    """Configuration for layers using learnable spectral filters.

    Parameters
    ----------
    sequence_length : int
        Input sequence length, must be positive.
    learnable_filters : bool
        Whether filters are learnable, defaults to True.
    fft_norm : FFTNorm
        FFT normalization mode, defaults to "ortho".
    filter_init_std : float
        Standard deviation for filter initialization, defaults to 0.02.
    """

    sequence_length: int = Field(gt=0, description="Input sequence length")
    learnable_filters: bool = Field(default=True, description="Learnable filters")
    fft_norm: FFTNorm = Field(default="ortho", description="FFT normalization")
    filter_init_std: float = Field(default=0.02, gt=0, description="Filter init std")


class AttentionLayerConfig(BaseLayerConfig):
    """Configuration for attention-based layers.

    Parameters
    ----------
    num_heads : int
        Number of attention heads, must be positive, defaults to 8.
    head_dim : int | None
        Dimension per head, defaults to None (auto-computed).
    """

    num_heads: int = Field(default=8, gt=0, description="Number of attention heads")
    head_dim: int | None = Field(default=None, description="Dimension per head")
