"""Full model configuration schemas for spectrans.

This module provides complete configuration models for entire spectrans models,
including all their components and parameters. These are the top-level
configurations that would be loaded from YAML files.

Classes
-------
ModelConfig
    Base configuration for all spectrans models.
FNetModelConfig
    Configuration for FNet transformer models.
GFNetModelConfig
    Configuration for Global Filter Network models.
AFNOModelConfig
    Configuration for Adaptive Fourier Neural Operator models.
LSTModelConfig
    Configuration for Linear Spectral Transform models.
SpectralAttentionModelConfig
    Configuration for Spectral Attention transformer models.

Notes
-----
Full model configurations compose together layer, block, and other component
configurations to define complete model architectures. These are what would
typically be loaded from YAML configuration files.

Examples
--------
>>> from spectrans.config.models import FNetModelConfig
>>> config = FNetModelConfig(
...     hidden_dim=768,
...     num_layers=12,
...     sequence_length=512
... )
>>> print(config.model_type)
'fnet'
"""

from pydantic import BaseModel, Field

from spectrans.core.types import (
    KernelType,
    OutputHeadType,
    PositionalEncodingType,
    TransformLSTType,
    WaveletType,
)


class ModelConfig(BaseModel):
    """Base configuration for all spectrans models.

    Attributes
    ----------
    model_type : str
        Type identifier for the model.
    hidden_dim : int
        Hidden dimension size, must be positive.
    num_layers : int
        Number of transformer layers, must be positive.
    sequence_length : int
        Maximum input sequence length, must be positive.
    dropout : float
        Global dropout probability, defaults to 0.0.
    vocab_size : int | None
        Vocabulary size for token embeddings, optional.
    num_classes : int | None
        Number of output classes for classification, optional.
    ffn_hidden_dim : int | None
        Hidden dimension for feedforward network, optional.
    use_positional_encoding : bool
        Whether to use positional encoding, defaults to True.
    positional_encoding_type : PositionalEncodingType
        Type of positional encoding ('sinusoidal', 'learned', 'rotary', 'alibi', 'none'), defaults to 'sinusoidal'.
    norm_eps : float
        Layer normalization epsilon, defaults to 1e-12.
    output_type : OutputHeadType
        Type of output head ('classification', 'regression', 'sequence', 'lm', 'none'), defaults to 'classification'.
    gradient_checkpointing : bool
        Whether to use gradient checkpointing, defaults to False.
    """

    model_type: str = Field(description="Model type identifier")
    hidden_dim: int = Field(gt=0, description="Hidden dimension size")
    num_layers: int = Field(gt=0, description="Number of layers")
    sequence_length: int = Field(gt=0, description="Sequence length")
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Global dropout")
    vocab_size: int | None = Field(default=None, ge=1, description="Vocabulary size")
    num_classes: int | None = Field(default=None, ge=1, description="Number of output classes")
    ffn_hidden_dim: int | None = Field(default=None, ge=1, description="FFN hidden dimension")
    use_positional_encoding: bool = Field(default=True, description="Use positional encoding")
    positional_encoding_type: PositionalEncodingType = Field(default="sinusoidal", description="Positional encoding type")
    norm_eps: float = Field(default=1e-12, gt=0, description="Layer norm epsilon")
    output_type: OutputHeadType = Field(default="classification", description="Output head type")
    gradient_checkpointing: bool = Field(default=False, description="Use gradient checkpointing")


class FNetModelConfig(ModelConfig):
    """Configuration for FNet transformer models.

    FNet models use Fourier mixing layers instead of attention.

    Attributes
    ----------
    use_real_fft : bool
        Whether to use real FFT for efficiency, defaults to True.
    """

    model_type: str = Field(default="fnet", description="Model type identifier")
    use_real_fft: bool = Field(default=True, description="Use real FFT for efficiency")


class GFNetModelConfig(ModelConfig):
    """Configuration for Global Filter Network models.

    GFNet models use learnable global filters in the frequency domain.

    Attributes
    ----------
    filter_activation : str
        Activation function for filters ('sigmoid' or 'tanh'), defaults to 'sigmoid'.
    """

    model_type: str = Field(default="gfnet", description="Model type identifier")
    filter_activation: str = Field(default="sigmoid", description="Filter activation function")


class AFNOModelConfig(ModelConfig):
    """Configuration for Adaptive Fourier Neural Operator models.

    AFNO models use adaptive Fourier mode truncation for efficient token mixing.

    Attributes
    ----------
    n_modes : int | None
        Number of Fourier modes to retain in sequence dimension, optional.
    modes_seq : int | None
        Number of Fourier modes in sequence dimension (alias for n_modes), optional.
    modes_hidden : int | None
        Number of Fourier modes in hidden dimension, optional.
    compression_ratio : float
        Compression ratio for modes_hidden when using n_modes, defaults to 0.5.
    mlp_ratio : float
        MLP expansion ratio in frequency domain, defaults to 2.0.
    """

    model_type: str = Field(default="afno", description="Model type identifier")
    n_modes: int | None = Field(default=None, ge=1, description="Number of Fourier modes")
    modes_seq: int | None = Field(default=None, ge=1, description="Modes in sequence dimension")
    modes_hidden: int | None = Field(default=None, ge=1, description="Modes in hidden dimension")
    compression_ratio: float = Field(default=0.5, gt=0.0, le=1.0, description="Mode compression ratio")
    mlp_ratio: float = Field(default=2.0, gt=0.0, description="MLP expansion ratio")


class LSTModelConfig(ModelConfig):
    """Configuration for Linear Spectral Transform models.

    LST models use linear spectral transforms (DCT/DST/Hadamard) for sequence mixing,
    achieving O(n log n) complexity through fast transform algorithms.

    Attributes
    ----------
    transform_type : TransformLSTType
        Type of spectral transform to use, defaults to "dct".
    use_conv_bias : bool
        Whether to use bias in spectral convolution, defaults to True.
    """

    model_type: str = Field(default="lst", description="Model type identifier")
    transform_type: TransformLSTType = Field(
        default="dct",
        description="Type of spectral transform"
    )
    use_conv_bias: bool = Field(default=True, description="Use bias in spectral convolution")


class SpectralAttentionModelConfig(ModelConfig):
    """Configuration for Spectral Attention transformer models.

    Spectral attention models use Random Fourier Features (RFF) to approximate
    attention with linear complexity O(n) instead of quadratic O(n²).

    Attributes
    ----------
    num_features : int | None
        Number of random Fourier features, defaults to None (uses hidden_dim).
    kernel_type : KernelType
        Type of kernel ('gaussian', 'softmax'), defaults to 'gaussian'.
    use_orthogonal : bool
        Whether to use orthogonal random features, defaults to True.
    num_heads : int
        Number of attention heads, defaults to 8.
    """

    model_type: str = Field(default="spectral_attention", description="Model type identifier")
    num_features: int | None = Field(default=None, ge=1, description="Number of RFF features")
    kernel_type: KernelType = Field(
        default="gaussian",
        description="Kernel type for RFF approximation"
    )
    use_orthogonal: bool = Field(default=True, description="Use orthogonal random features")
    num_heads: int = Field(default=8, ge=1, description="Number of attention heads")


class FNOTransformerConfig(ModelConfig):
    """Configuration for Fourier Neural Operator transformer models.

    FNO models use spectral convolutions in the Fourier domain to learn
    mappings between function spaces with O(n log n) complexity.

    Attributes
    ----------
    modes : int
        Number of Fourier modes to retain (frequency truncation), defaults to 32.
    mlp_ratio : float
        Expansion ratio for the MLP in FNO blocks, defaults to 2.0.
    use_2d : bool
        Whether to use 2D spectral convolutions for spatial data, defaults to False.
    spatial_dim : int | None
        Spatial dimension when using 2D convolutions (sequence = spatial_dim²), optional.
    """

    model_type: str = Field(default="fno_transformer", description="Model type identifier")
    modes: int = Field(default=32, ge=1, description="Number of Fourier modes")
    mlp_ratio: float = Field(default=2.0, gt=0.0, description="MLP expansion ratio")
    use_2d: bool = Field(default=False, description="Use 2D spectral convolutions")
    spatial_dim: int | None = Field(default=None, ge=1, description="Spatial dimension for 2D")


class WaveletTransformerConfig(ModelConfig):
    """Configuration for Wavelet Transformer models.

    Wavelet transformers use discrete wavelet transforms (DWT) for sequence mixing,
    providing multi-resolution analysis with O(n) complexity.

    Attributes
    ----------
    wavelet : WaveletType
        Type of wavelet to use ('db4', 'sym6', 'coif3', etc.), defaults to 'db4'.
    levels : int
        Number of decomposition levels (typically 1-5), defaults to 3.
    mixing_mode : str
        How to mix wavelet coefficients ('pointwise', 'channel', 'level'), defaults to 'pointwise'.
    """

    model_type: str = Field(default="wavelet_transformer", description="Model type identifier")
    wavelet: WaveletType = Field(
        default="db4",
        description="Wavelet type (e.g., 'db4', 'sym6', 'coif3')"
    )
    levels: int = Field(default=3, ge=1, le=10, description="Number of decomposition levels")
    mixing_mode: str = Field(
        default="pointwise",
        pattern="^(pointwise|channel|level)$",
        description="Wavelet coefficient mixing strategy"
    )


class HybridModelConfig(ModelConfig):
    """Configuration for Hybrid Spectral-Spatial Transformer models.

    Hybrid models alternate between different mixing strategies (e.g., spectral
    and spatial) across layers, combining strengths of multiple approaches.

    Attributes
    ----------
    spectral_type : str
        Type of spectral mixing ('fourier', 'wavelet', 'afno', 'gfnet'), defaults to 'fourier'.
    spatial_type : str
        Type of spatial mixing ('attention', 'spectral_attention', 'lst'), defaults to 'attention'.
    alternation_pattern : str
        How to alternate layers ('even_spectral', 'alternate', 'custom'), defaults to 'even_spectral'.
    num_heads : int
        Number of attention heads for spatial layers, defaults to 8.
    spectral_config : dict | None
        Additional configuration for spectral layers, optional.
    spatial_config : dict | None
        Additional configuration for spatial layers, optional.
    """

    model_type: str = Field(default="hybrid", description="Model type identifier")
    spectral_type: str = Field(
        default="fourier",
        pattern="^(fourier|wavelet|afno|gfnet)$",
        description="Type of spectral mixing"
    )
    spatial_type: str = Field(
        default="attention",
        pattern="^(attention|spectral_attention|lst)$",
        description="Type of spatial mixing"
    )
    alternation_pattern: str = Field(
        default="even_spectral",
        pattern="^(even_spectral|alternate|custom)$",
        description="Layer alternation pattern"
    )
    num_heads: int = Field(default=8, ge=1, description="Number of attention heads")
    spectral_config: dict | None = Field(default=None, description="Spectral layer config")
    spatial_config: dict | None = Field(default=None, description="Spatial layer config")
