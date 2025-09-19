"""Core components and interfaces for the spectrans library.

This module provides the fundamental building blocks for spectral transformer implementations,
including abstract base classes, type definitions, and the component registry system. All
spectral transformer components inherit from these base classes to ensure consistent APIs
and enable modular composition through the registry.

The core module establishes the mathematical foundations and software architecture that
allows for flexible experimentation with different spectral transform combinations while
maintaining type safety and performance.

Modules
-------
base
    Base classes and interfaces for all spectral components.
registry
    Component registration and discovery system.
types
    Type definitions and aliases for the library.

Classes
-------
AttentionLayer
    Base class for spectral attention mechanisms.
BaseModel
    Base class for spectral transformer models.
ComponentRegistry
    Registry system for dynamic component discovery and instantiation.
SpectralComponent
    Abstract base class for all spectral neural network components.
TransformerBlock
    Base class for transformer blocks with residual connections.

Functions
---------
create_component
    Factory function to create registered component instances.
get_component
    Retrieve component class from registry.
list_components
    List all registered components in a category.
register_component
    Decorator to register components in the global registry.

Attributes
----------
registry : ComponentRegistry
    Global component registry instance.

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

1. **Abstract Interfaces**: All components implement consistent forward() methods
2. **Type Safety**: Type hints with Python 3.13 syntax
3. **Modularity**: Registry system enables runtime component composition
4. **Extensibility**: Easy to add new transforms and mixing strategies

The registry system supports six categories of components:
- transform: Spectral transforms (FFT, DCT, DWT, Hadamard)
- mixing: Token mixing layers (FourierMixing, GlobalFilter, etc.)
- attention: Spectral attention mechanisms
- block: Transformer blocks
- model: Model implementations
- kernel: Kernel functions for attention approximation

See Also
--------
[`spectrans.core.base`][] : Base class definitions and interfaces.
[`spectrans.core.types`][] : Type aliases and definitions.
[`spectrans.core.registry`][] : Component registration system.
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
    KernelType,
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
    TransformLSTType,
    TransformType,
    VocabSize,
    WaveletType,
    WindowFunction,
    WorldSize,
)

# Public API - alphabetically sorted
__all__ = [
    "ActivationType",
    "AttentionLayer",
    "AttentionMask",
    "BaseModel",
    "BatchDict",
    "BatchSize",
    "BatchTuple",
    "BoolTensor",
    "CausalMask",
    "CheckpointDict",
    "ComplexTensor",
    "ComponentClass",
    "ComponentFactory",
    "ComponentRegistry",
    "ComponentType",
    "ConfigDict",
    "Device",
    "FFTNorm",
    "FeatureMapFunction",
    "FourierModes",
    "GradientClipNorm",
    "GradientClipValue",
    "HeadDim",
    "HiddenDim",
    "InitializationType",
    "IntermediateDim",
    "KernelFunction",
    "KernelType",
    "LearnableFilter",
    "LocalRank",
    "LongTensor",
    "LossFunction",
    "LossOutput",
    "MetricFunction",
    "MixedPrecisionDType",
    "ModeIndices",
    "ModeSelection",
    "ModeTruncation",
    "ModelOutput",
    "ModelType",
    "ModuleType",
    "NormType",
    "NumClasses",
    "NumHeads",
    "NumLayers",
    "NumRandomFeatures",
    "OptimizerConfig",
    "OptionalModule",
    "OptionalTensor",
    "OutputHeadType",
    "PaddingSize",
    "PaddingType",
    "ParamsDict",
    "PoolingType",
    "PositionalEncodingType",
    "RandomSeed",
    "Rank",
    "RegistryDict",
    "SchedulerConfig",
    "SchedulerFunction",
    "SequenceLength",
    "Shape2D",
    "Shape3D",
    "Shape4D",
    "SpectralComponent",
    "SpectralFilter",
    "StateDict",
    "Tensor",
    "TrainingConfig",
    "TransformLSTType",
    "TransformType",
    "TransformerBlock",
    "VocabSize",
    "WaveletType",
    "WindowFunction",
    "WorldSize",
    "create_component",
    "get_component",
    "list_components",
    "register_component",
    "registry",
]
