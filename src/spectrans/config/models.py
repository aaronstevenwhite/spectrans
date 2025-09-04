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
    """

    model_type: str = Field(description="Model type identifier")
    hidden_dim: int = Field(gt=0, description="Hidden dimension size")
    num_layers: int = Field(gt=0, description="Number of layers")
    sequence_length: int = Field(gt=0, description="Sequence length")
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Global dropout")


class FNetModelConfig(ModelConfig):
    """Configuration for FNet transformer models.

    FNet models use Fourier mixing layers instead of attention.
    """

    model_type: str = Field(default="fnet", description="Model type identifier")


class GFNetModelConfig(ModelConfig):
    """Configuration for Global Filter Network models.

    GFNet models use learnable global filters in the frequency domain.
    """

    model_type: str = Field(default="gfnet", description="Model type identifier")
