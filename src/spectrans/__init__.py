"""Spectrans: Modular spectral transformer implementations in PyTorch.

A comprehensive library for spectral transformers, providing efficient
alternatives to traditional attention mechanisms using Fourier transforms,
wavelets, and other spectral methods.
"""

__version__ = "0.1.0"

# Import core components
from .core.base import (
    AttentionLayer,
    BaseModel,
    MixingLayer,
    SpectralComponent,
    SpectralTransform,
    TransformerBlock,
)
from .core.registry import (
    create_component,
    get_component,
    list_components,
    register_component,
    registry,
)
from .core.types import (
    ActivationType,
    ComponentType,
    ConfigDict,
    ModelType,
    NormType,
    Tensor,
    TransformType,
    WaveletType,
)

# Public API
__all__: list[str] = [
    "ActivationType",
    "AttentionLayer",
    "BaseModel",
    "ComponentType",
    "ConfigDict",
    "MixingLayer",
    "ModelType",
    "NormType",
    # Base classes
    "SpectralComponent",
    "SpectralTransform",
    # Type exports
    "Tensor",
    "TransformType",
    "TransformerBlock",
    "WaveletType",
    # Version
    "__version__",
    "create_component",
    "get_component",
    "list_components",
    "register_component",
    # Registry functions
    "registry",
]
