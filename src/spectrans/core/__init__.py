"""Core components and interfaces for the spectrans library.

This module provides the fundamental building blocks for spectral transformer implementations,
including abstract base classes, type definitions, and the component registry system. All
spectral transformer components inherit from these base classes to ensure consistent APIs
and enable modular composition through the registry.

The core module establishes the mathematical foundations and software architecture that
allows for flexible experimentation with different spectral transform combinations while
maintaining type safety and performance.

Classes
-------
SpectralComponent
    Abstract base class for all spectral neural network components.
SpectralTransform
    Interface for spectral transform operations (FFT, DCT, DWT, etc.).
AttentionLayer
    Base class for spectral attention mechanisms.
TransformerBlock
    Base class for complete transformer blocks with residual connections.
BaseModel
    Base class for full spectral transformer models.
ComponentRegistry
    Registry system for dynamic component discovery and instantiation.

Functions
---------
register_component(category, name)
    Decorator to register components in the global registry.
create_component(category, name, **kwargs)
    Factory function to create registered component instances.
get_component(category, name)
    Retrieve component class from registry.
list_components(category)
    List all registered components in a category.

Examples
--------
Using the component registry system:

>>> from spectrans.core import register_component, create_component
>>> from spectrans.layers.mixing.base import MixingLayer
>>> @register_component('mixing', 'custom')
... class CustomMixing(MixingLayer):
...     def forward(self, x):
...         return x  # Custom implementation
>>> mixing = create_component('mixing', 'custom', hidden_dim=768)

Working with base classes for type safety:

>>> from spectrans.core import SpectralComponent
>>> def process_component(component: SpectralComponent):
...     complexity = component.complexity
...     return component(input_tensor)

Notes
-----
The core architecture follows these design principles:

1. **Abstract Interfaces**: All components implement consistent forward() and complexity methods
2. **Type Safety**: Comprehensive type hints with modern Python 3.13 syntax
3. **Modularity**: Registry system enables runtime component composition
4. **Mathematical Rigor**: Complexity analysis built into base classes
5. **Extensibility**: Easy to add new transforms and mixing strategies

The registry system supports six categories of components:
- transform: Spectral transforms (FFT, DCT, DWT, Hadamard)
- mixing: Token mixing layers (FourierMixing, GlobalFilter, etc.)
- attention: Spectral attention mechanisms
- block: Complete transformer blocks
- model: Full model implementations
- kernel: Kernel functions for attention approximation

See Also
--------
spectrans.core.base : Base class definitions and interfaces
spectrans.core.types : Type aliases and definitions
spectrans.core.registry : Component registration system
"""

from .base import (
    AttentionLayer,
    BaseModel,
    SpectralComponent,
    TransformerBlock,
)
from .registry import (
    ComponentRegistry,
    create_component,
    get_component,
    list_components,
    register_component,
    registry,
)
from .types import (
    ActivationType,
    AttentionMask,
    BatchDict,
    BatchSize,
    BatchTuple,
    BoolTensor,
    CausalMask,
    CheckpointDict,
    ComplexityInfo,
    ComplexTensor,
    ComponentClass,
    ComponentFactory,
    ComponentType,
    ConfigDict,
    Device,
    FeatureMapFunction,
    FFTNorm,
    FourierModes,
    GradientClipNorm,
    GradientClipValue,
    HeadDim,
    HiddenDim,
    InitializationType,
    IntermediateDim,
    KernelFunction,
    LearnableFilter,
    LocalRank,
    LongTensor,
    LossFunction,
    LossOutput,
    MetricFunction,
    MixedPrecisionDType,
    ModeIndices,
    ModelOutput,
    ModelType,
    ModeSelection,
    ModeTruncation,
    ModuleType,
    NormType,
    NumClasses,
    NumHeads,
    NumLayers,
    NumRandomFeatures,
    OptimizerConfig,
    OptionalModule,
    OptionalTensor,
    OutputHeadType,
    PaddingSize,
    PaddingType,
    ParamsDict,
    PoolingType,
    PositionalEncodingType,
    RandomSeed,
    Rank,
    RegistryDict,
    SchedulerConfig,
    SchedulerFunction,
    SequenceLength,
    Shape2D,
    Shape3D,
    Shape4D,
    SpectralFilter,
    StateDict,
    Tensor,
    TrainingConfig,
    TransformType,
    VocabSize,
    WaveletType,
    WindowFunction,
    WorldSize,
)

__all__: list[str] = [
    "ActivationType",
    "AttentionLayer",
    "AttentionMask",
    "BaseModel",
    "BatchSize",
    "BoolTensor",
    "ComplexTensor",
    "ComplexityInfo",
    # Registry
    "ComponentRegistry",
    "ComponentType",
    "ConfigDict",
    "Device",
    "HiddenDim",
    "LongTensor",
    "ModelType",
    "NormType",
    "NumHeads",
    "NumLayers",
    "OptionalTensor",
    "ParamsDict",
    "SequenceLength",
    # Base classes
    "SpectralComponent",
    "SpectralTransform",
    "StateDict",
    # Type exports (selected most commonly used)
    "Tensor",
    "TransformType",
    "TransformerBlock",
    "WaveletType",
    "create_component",
    "get_component",
    "list_components",
    "register_component",
    "registry",
]
