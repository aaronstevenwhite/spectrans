r"""Configuration system for spectral transformer models.

This module provides configuration management for spectrans
models and components using Pydantic for type safety and validation. The
configuration system supports YAML-based configuration files, programmatic
configuration, and dynamic model building.

Modules
-------
builder
    YAML configuration loading and model building.
core
    Base configuration classes.
layers
    Layer-specific configuration schemas.
models
    Model configuration schemas.

Classes
-------
AFNOMixingConfig
    Configuration for AFNO mixing layers.
AFNOModelConfig
    Configuration for AFNO models.
AttentionLayerConfig
    Base configuration for attention layers.
BaseLayerConfig
    Base configuration for all layers.
ConfigBuilder
    Builder for creating models from configuration.
ConfigurationError
    Exception for configuration errors.
DCTAttentionConfig
    Configuration for DCT attention layers.
FilterLayerConfig
    Configuration for filter-based layers.
FNetModelConfig
    Configuration for FNet models.
FNOTransformerConfig
    Configuration for FNO transformer models.
FourierMixingConfig
    Configuration for Fourier mixing layers.
GFNetModelConfig
    Configuration for GFNet models.
GlobalFilterMixingConfig
    Configuration for global filter mixing.
HadamardAttentionConfig
    Configuration for Hadamard attention.
HybridModelConfig
    Configuration for hybrid models.
LSTAttentionConfig
    Configuration for LST attention.
LSTModelConfig
    Configuration for LST models.
MixedTransformAttentionConfig
    Configuration for mixed transform attention.
ModelConfig
    Base configuration for all models.
SpectralAttentionConfig
    Configuration for spectral attention.
SpectralAttentionModelConfig
    Configuration for spectral attention models.
SpectralKernelAttentionConfig
    Configuration for spectral kernel attention.
UnitaryLayerConfig
    Configuration for unitary layers.
WaveletMixing2DConfig
    Configuration for 2D wavelet mixing.
WaveletMixingConfig
    Configuration for wavelet mixing.
WaveletTransformerConfig
    Configuration for wavelet transformers.

Functions
---------
build_component_from_config
    Build a component from configuration dictionary.
build_model_from_config
    Build a model from configuration dictionary.
load_yaml_config
    Load configuration from YAML file.

Examples
--------
Loading and building from YAML:

>>> from spectrans.config import ConfigBuilder
>>>
>>> builder = ConfigBuilder()
>>> model = builder.build_model("configs/fnet.yaml")
>>> print(model.num_parameters())

Programmatic configuration:

>>> from spectrans.config import FNetModelConfig, build_model_from_config
>>>
>>> config = FNetModelConfig(
...     hidden_dim=768,
...     num_layers=12,
...     vocab_size=50000,
...     max_seq_len=512
... )
>>> model = build_model_from_config(config.model_dump())

Custom layer configuration:

>>> from spectrans.config import GlobalFilterMixingConfig
>>>
>>> layer_config = GlobalFilterMixingConfig(
...     hidden_dim=512,
...     sequence_length=1024,
...     activation="sigmoid",
...     filter_regularization=0.01
... )
>>> layer = build_component_from_config(layer_config.model_dump())

Notes
-----
**Configuration System Design:**

The configuration system uses Pydantic for:
- Type validation and coercion
- Default value management
- Nested configuration structures
- JSON/YAML serialization

Configuration hierarchy:
1. Base classes (BaseLayerConfig, ModelConfig)
2. Specialized layer configs (mixing, attention, etc.)
3. Model configurations
4. Builder system for instantiation

All configurations support:
- Validation of parameter ranges
- Type checking at configuration time
- Serialization to/from YAML and JSON
- Programmatic and file-based configuration

See Also
--------
[`spectrans.config.builder`][] : Configuration building utilities.
[`spectrans.config.models`][] : Model configuration schemas.
[`spectrans.config.layers`][] : Layer configuration schemas.
"""

from .builder import (
    ConfigBuilder,
    ConfigurationError,
    build_component_from_config,
    build_model_from_config,
    load_yaml_config,
)
from .core import AttentionLayerConfig, BaseLayerConfig, FilterLayerConfig, UnitaryLayerConfig
from .layers import (
    AFNOMixingConfig,
    DCTAttentionConfig,
    FourierMixingConfig,
    GlobalFilterMixingConfig,
    HadamardAttentionConfig,
    LSTAttentionConfig,
    MixedTransformAttentionConfig,
    SpectralAttentionConfig,
    SpectralKernelAttentionConfig,
    WaveletMixing2DConfig,
    WaveletMixingConfig,
)
from .models import (
    AFNOModelConfig,
    FNetModelConfig,
    FNOTransformerConfig,
    GFNetModelConfig,
    HybridModelConfig,
    LSTModelConfig,
    ModelConfig,
    SpectralAttentionModelConfig,
    WaveletTransformerConfig,
)

# Public API - alphabetically sorted
__all__ = [
    "AFNOMixingConfig",
    "AFNOModelConfig",
    "AttentionLayerConfig",
    "BaseLayerConfig",
    "ConfigBuilder",
    "ConfigurationError",
    "DCTAttentionConfig",
    "FNOTransformerConfig",
    "FNetModelConfig",
    "FilterLayerConfig",
    "FourierMixingConfig",
    "GFNetModelConfig",
    "GlobalFilterMixingConfig",
    "HadamardAttentionConfig",
    "HybridModelConfig",
    "LSTAttentionConfig",
    "LSTModelConfig",
    "MixedTransformAttentionConfig",
    "ModelConfig",
    "SpectralAttentionConfig",
    "SpectralAttentionModelConfig",
    "SpectralKernelAttentionConfig",
    "UnitaryLayerConfig",
    "WaveletMixing2DConfig",
    "WaveletMixingConfig",
    "WaveletTransformerConfig",
    "build_component_from_config",
    "build_model_from_config",
    "load_yaml_config",
]
