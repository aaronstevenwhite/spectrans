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


class ModelConfig(BaseModel):
    """Base configuration for all spectrans models.

    Parameters
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
    positional_encoding_type : str
        Type of positional encoding ('sinusoidal' or 'learned'), defaults to 'sinusoidal'.
    norm_eps : float
        Layer normalization epsilon, defaults to 1e-12.
    output_type : str
        Type of output head ('classification', 'regression', 'sequence', 'none'), defaults to 'classification'.
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
    positional_encoding_type: str = Field(default="sinusoidal", description="Positional encoding type")
    norm_eps: float = Field(default=1e-12, gt=0, description="Layer norm epsilon")
    output_type: str = Field(default="classification", description="Output head type")
    gradient_checkpointing: bool = Field(default=False, description="Use gradient checkpointing")


class FNetModelConfig(ModelConfig):
    """Configuration for FNet transformer models.

    FNet models use Fourier mixing layers instead of attention.

    Parameters
    ----------
    use_real_fft : bool
        Whether to use real FFT for efficiency, defaults to True.
    """

    model_type: str = Field(default="fnet", description="Model type identifier")
    use_real_fft: bool = Field(default=True, description="Use real FFT for efficiency")


class GFNetModelConfig(ModelConfig):
    """Configuration for Global Filter Network models.

    GFNet models use learnable global filters in the frequency domain.

    Parameters
    ----------
    filter_activation : str
        Activation function for filters ('sigmoid' or 'tanh'), defaults to 'sigmoid'.
    """

    model_type: str = Field(default="gfnet", description="Model type identifier")
    filter_activation: str = Field(default="sigmoid", description="Filter activation function")


class AFNOModelConfig(ModelConfig):
    """Configuration for Adaptive Fourier Neural Operator models.

    AFNO models use adaptive Fourier mode truncation for efficient token mixing.

    Parameters
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
