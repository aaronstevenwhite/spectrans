"""Configuration system for spectrans.

This package provides comprehensive configuration management for spectrans
models and components using Pydantic for type safety and validation.

Modules
-------
core : Base configuration classes
models : Complete model configuration schemas
layers : Layer-specific configuration schemas
builder : YAML configuration loading and model building

Functions
---------
load_yaml_config : Load YAML configuration files
build_model_from_config : Build models from configuration dictionaries
build_component_from_config : Build components from configuration

Examples
--------
>>> from spectrans.config import ConfigBuilder, build_model_from_config
>>> builder = ConfigBuilder()
>>> model = builder.build_model("configs/fnet.yaml")
"""

from .builder import (
    ConfigBuilder,
    ConfigurationError,
    build_component_from_config,
    build_model_from_config,
    load_yaml_config,
)
from .core import (
    AttentionLayerConfig,
    BaseLayerConfig,
    FilterLayerConfig,
    UnitaryLayerConfig,
)
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

# ruff: noqa: RUF022
__all__ = [
    # Builder and utilities
    "ConfigBuilder",
    "ConfigurationError",
    "build_component_from_config",
    "build_model_from_config",
    "load_yaml_config",
    # Core configuration classes
    "AttentionLayerConfig",
    "BaseLayerConfig",
    "FilterLayerConfig",
    "UnitaryLayerConfig",
    # Layer configurations
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
    # Model configurations
    "AFNOModelConfig",
    "FNetModelConfig",
    "FNOTransformerConfig",
    "GFNetModelConfig",
    "HybridModelConfig",
    "LSTModelConfig",
    "ModelConfig",
    "SpectralAttentionModelConfig",
    "WaveletTransformerConfig",
]
