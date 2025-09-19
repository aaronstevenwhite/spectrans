"""Configuration models for attention layer components.

This module provides Pydantic models for validating and typing configuration
data used to construct attention layers in spectrans.

Classes
-------
SpectralAttentionConfig
    Configuration for Spectral Attention layer with Random Fourier Features.
LSTAttentionConfig
    Configuration for Linear Spectral Transform Attention layer.
DCTAttentionConfig
    Configuration for DCT-based attention layer.
HadamardAttentionConfig
    Configuration for Hadamard-based attention layer.
MixedTransformAttentionConfig
    Configuration for mixed transform attention layer.

Notes
-----
All configuration models use Pydantic v2 BaseModel for validation and type safety.
Attention layer configurations inherit from AttentionLayerConfig in the parent
core module.

Examples
--------
>>> from spectrans.config.layers.attention import SpectralAttentionConfig
>>> config = SpectralAttentionConfig(
...     hidden_dim=768,
...     num_heads=8,
...     num_features=256
... )
>>> print(config.kernel_type)
'softmax'
"""

from pydantic import Field

from ...core.types import KernelType, SpectralKernelType, TransformLSTType
from ..core import AttentionLayerConfig


class SpectralAttentionConfig(AttentionLayerConfig):
    """Configuration for Spectral Attention with Random Fourier Features.

    Attributes
    ----------
    num_features : int | None
        Number of random Fourier features, defaults to None (uses head_dim).
    kernel_type : KernelType
        Type of kernel ('gaussian' or 'softmax'), defaults to 'softmax'.
    use_orthogonal : bool
        Whether to use orthogonal random features, defaults to True.
    feature_redraw : bool
        Whether to redraw features during training, defaults to False.
    use_bias : bool
        Whether to use bias in projections, defaults to True.
    """

    num_features: int | None = Field(
        default=None, ge=1, description="Number of random Fourier features"
    )
    kernel_type: KernelType = Field(
        default="softmax", description="Kernel type for RFF approximation"
    )
    use_orthogonal: bool = Field(default=True, description="Use orthogonal random features")
    feature_redraw: bool = Field(default=False, description="Redraw features during training")
    use_bias: bool = Field(default=True, description="Use bias in projections")


class LSTAttentionConfig(AttentionLayerConfig):
    """Configuration for Linear Spectral Transform Attention.

    Attributes
    ----------
    transform_type : TransformLSTType
        Type of spectral transform ('dct', 'dst', 'hadamard', 'mixed'), defaults to 'dct'.
    learnable_scale : bool
        Whether to use learnable diagonal scaling, defaults to True.
    normalize : bool
        Whether to normalize transform output, defaults to True.
    use_bias : bool
        Whether to use bias in projections, defaults to True.
    """

    transform_type: TransformLSTType = Field(
        default="dct", description="Type of spectral transform"
    )
    learnable_scale: bool = Field(default=True, description="Learnable diagonal scaling")
    normalize: bool = Field(default=True, description="Normalize transform output")
    use_bias: bool = Field(default=True, description="Use bias in projections")


class DCTAttentionConfig(AttentionLayerConfig):
    """Configuration for DCT-based attention layer.

    Attributes
    ----------
    dct_type : int
        Type of DCT transform (typically 2), defaults to 2.
    learnable_scale : bool
        Whether to use learnable diagonal scaling, defaults to True.
    """

    dct_type: int = Field(default=2, ge=1, le=4, description="DCT transform type")
    learnable_scale: bool = Field(default=True, description="Learnable diagonal scaling")


class HadamardAttentionConfig(AttentionLayerConfig):
    """Configuration for Hadamard-based attention layer.

    Attributes
    ----------
    scale_by_sqrt : bool
        Whether to scale by sqrt(n), defaults to True.
    learnable_scale : bool
        Whether to use learnable diagonal scaling, defaults to True.
    """

    scale_by_sqrt: bool = Field(default=True, description="Scale by sqrt(n)")
    learnable_scale: bool = Field(default=True, description="Learnable diagonal scaling")


class MixedTransformAttentionConfig(AttentionLayerConfig):
    """Configuration for mixed transform attention layer.

    Attributes
    ----------
    use_fft : bool
        Whether to use FFT transforms, defaults to True.
    use_dct : bool
        Whether to use DCT transforms, defaults to True.
    use_hadamard : bool
        Whether to use Hadamard transforms, defaults to True.
    """

    use_fft: bool = Field(default=True, description="Use FFT transforms")
    use_dct: bool = Field(default=True, description="Use DCT transforms")
    use_hadamard: bool = Field(default=True, description="Use Hadamard transforms")


class SpectralKernelAttentionConfig(AttentionLayerConfig):
    """Configuration for spectral kernel attention.

    Attributes
    ----------
    kernel_type : SpectralKernelType
        Type of spectral kernel ('gaussian', 'polynomial', 'spectral'), defaults to 'gaussian'.
    rank : int | None
        Rank for low-rank approximation, defaults to None (uses min(64, head_dim)).
    num_features : int | None
        Number of features for approximation, defaults to None.
    """

    kernel_type: SpectralKernelType = Field(
        default="gaussian", description="Type of spectral kernel"
    )
    rank: int | None = Field(default=None, ge=1, description="Rank for low-rank approximation")
    num_features: int | None = Field(
        default=None, ge=1, description="Number of approximation features"
    )
