"""Type definitions and aliases for the spectrans library.

This module provides comprehensive definitions, aliases, and literal types used
throughout the spectrans library. The system ensures safety, enables better
IDE support, and provides clear documentation of expected parameter types for all
spectral transformer components.

The types are organized into logical groups covering tensors, shapes, transforms,
neural network components, training configurations, and mathematical operations.
All definitions use modern Python 3.13 syntax with union operators and
the new statement for aliases.

Attributes
----------
Tensor : type
    Alias for torch.Tensor, the primary data type.
ComplexTensor : type
    Alias for complex-valued torch.Tensor.
BatchSize, SequenceLength, HiddenDim : int
    Common tensor dimension types.

Classes
-------
TransformType : Literal
    Valid spectral transform names ('fourier', 'cosine', 'hadamard', 'wavelet').
WaveletType : Literal
    Valid wavelet family names (Daubechies, Symlets, Coiflets, Biorthogonal).
ActivationType : Literal
    Valid activation function names ('relu', 'gelu', 'swish', etc.).
NormType : Literal
    Valid normalization layer types ('layernorm', 'batchnorm', etc.).
ModelType : Literal
    Valid model architecture names ('fnet', 'gfnet', 'afno', etc.).
ComponentType : Literal
    Component categories for the registry system.

Examples
--------
Using annotations in function signatures:

>>> from spectrans.core.types import Tensor, TransformType
>>> def apply_transform(x: Tensor, transform: TransformType) -> Tensor:
...     # Implementation with proper hints
...     pass

Working with configuration dictionaries:

>>> from spectrans.core.types import ConfigDict, ModelType
>>> config: ConfigDict = {
...     'model_type': 'fnet',
...     'hidden_dim': 768,
...     'num_layers': 12
... }

Complex tensor operations:

>>> from spectrans.core.types import ComplexTensor
>>> def process_spectral_component(x: ComplexTensor) -> ComplexTensor:
...     return x  # Process and return complex tensor

Notes
-----
Type System Philosophy:

1. **Explicit over Implicit**: All function signatures should include hints
2. **Literal Types**: Use Literal for enumerated options to prevent typos
3. **Type Aliases**: Semantic names (BatchSize) over raw types (int)
4. **Union Syntax**: Modern | syntax instead of Union[] for Python 3.13
5. **Optional Types**: T, None pattern instead of Optional[T]

Mathematical Type Categories:

- **Tensor Types**: Basic tensor operations and complex number support
- **Shape Types**: Common tensor dimension patterns (2D, 3D, 4D)
- **Transform Types**: Spectral transform varieties and parameters
- **Configuration Types**: Structured configuration and parameter passing

The system supports both runtime checking (where appropriate) and
static analysis with mypy, providing comprehensive safety for the library.

See Also
--------
spectrans.core.base : Base classes that use these definitions
spectrans.utils.complex : Complex tensor operations with proper typing
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeVar

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from spectrans.core.base import (
        AttentionLayer,
        BaseModel,
        TransformerBlock,
    )
    from spectrans.layers.mixing.base import MixingLayer
    from spectrans.transforms.base import SpectralTransform

# Tensor type aliases
type Tensor = torch.Tensor
type ComplexTensor = torch.Tensor  # Complex-valued tensor
type LongTensor = torch.LongTensor
type BoolTensor = torch.BoolTensor

# Shape type aliases
type BatchSize = int
type SequenceLength = int
type HiddenDim = int
type NumHeads = int
type HeadDim = int
type IntermediateDim = int
type NumLayers = int
type VocabSize = int
type NumClasses = int

# Common tensor shapes
type Shape2D = tuple[int, int]
type Shape3D = tuple[int, int, int]
type Shape4D = tuple[int, int, int, int]

# Transform types
TransformType = Literal[
    "fourier",
    "cosine",
    "sine",
    "hadamard",
    "wavelet",
]

# Wavelet types
WaveletType = Literal[
    "db1",  # Daubechies wavelets
    "db2",
    "db3",
    "db4",
    "db5",
    "db6",
    "db7",
    "db8",
    "sym2",  # Symlets
    "sym3",
    "sym4",
    "sym5",
    "sym6",
    "sym7",
    "sym8",
    "coif1",  # Coiflets
    "coif2",
    "coif3",
    "coif4",
    "coif5",
    "bior1.1",  # Biorthogonal
    "bior1.3",
    "bior1.5",
    "bior2.2",
    "bior2.4",
    "bior2.6",
    "bior2.8",
]

# Activation function types
ActivationType = Literal[
    "relu",
    "gelu",
    "swish",
    "silu",
    "mish",
    "tanh",
    "sigmoid",
    "identity",
]

# Type for filter activation in GFNet models
FilterActivationType = Literal["sigmoid", "tanh"]

# Normalization types
NormType = Literal[
    "layernorm",
    "batchnorm",
    "groupnorm",
    "rmsnorm",
    "none",
]

# Model types
ModelType = Literal[
    "fnet",
    "gfnet",
    "afno",
    "spectral_attention",
    "lst",
    "fno_transformer",
    "wavenet_transformer",
    "hybrid",
]

# Component types for registry
ComponentType = Literal[
    "transform",
    "mixing",
    "attention",
    "block",
    "model",
    "kernel",
    "operator",
]

# Configuration types
type ConfigDict = dict[str, int | float | str | bool | list[int | float | str | bool]]
type ParamsDict = dict[str, int | float | str | bool | list[int | float | str | bool]]


# Callback types
type LossFunction = Callable[[Tensor, Tensor], Tensor]
type MetricFunction = Callable[[Tensor, Tensor], float]
type SchedulerFunction = Callable[[int], float]

# Module types
ModuleType = TypeVar("ModuleType", bound=nn.Module)
TransformModuleType = TypeVar("TransformModuleType", bound="SpectralTransform")
MixingModuleType = TypeVar("MixingModuleType", bound="MixingLayer")
AttentionModuleType = TypeVar("AttentionModuleType", bound="AttentionLayer")
BlockModuleType = TypeVar("BlockModuleType", bound="TransformerBlock")
ModelModuleType = TypeVar("ModelModuleType", bound="BaseModel")

# Device types
type Device = torch.device | str | None

# Optional types
type OptionalTensor = Tensor | None
type OptionalModule = nn.Module | None

# Fourier mode types
type FourierModes = int  # Number of Fourier modes to keep
type ModeTruncation = tuple[int, ...]  # Mode truncation per dimension

# Random feature types
type NumRandomFeatures = int
type RandomSeed = int | None


# Training configuration
type OptimizerConfig = dict[str, int | float | str | bool]
type SchedulerConfig = dict[str, int | float | str | bool]
type TrainingConfig = dict[str, int | float | str | bool]

# Model state
type StateDict = dict[str, Tensor]
type CheckpointDict = dict[str, Tensor | int | float | str | bool | StateDict]

# Registry types
type ComponentClass = type[nn.Module]
type ComponentFactory = Callable[..., nn.Module]
type RegistryDict = dict[str, dict[str, ComponentClass]]

# Initialization types
InitializationType = Literal[
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "normal",
    "uniform",
    "ones",
    "zeros",
    "orthogonal",
]

# Padding types
PaddingType = Literal[
    "constant",
    "reflect",
    "replicate",
    "circular",
    "zeros",
]
type PaddingSize = int | tuple[int, ...]

# FFT normalization modes
FFTNorm = Literal["forward", "backward", "ortho"]

# Attention mask types
type AttentionMask = BoolTensor | None
type CausalMask = BoolTensor | None

# Position encoding types
PositionalEncodingType = Literal[
    "learned",
    "sinusoidal",
    "rotary",
    "alibi",
    "none",
]

# Output head types for models
OutputHeadType = Literal[
    "classification",
    "regression",
    "sequence",
    "lm",
    "none",
]

# Pooling strategies for sequence aggregation
PoolingType = Literal[
    "cls",
    "mean",
    "max",
]

# Output types for different model modes
type ModelOutput = Tensor | tuple[Tensor, ...]
type LossOutput = Tensor | tuple[Tensor, dict[str, Tensor]]

# Batch types
type BatchDict = dict[str, Tensor]
type BatchTuple = tuple[Tensor, ...]

# Gradient clipping types
type GradientClipValue = float | None
type GradientClipNorm = float | None

# Mixed precision types
type MixedPrecisionDType = Literal["float16", "bfloat16", "float32"]
type AutocastDType = torch.dtype | None

# Distributed training types
type WorldSize = int
type Rank = int
type LocalRank = int

# Kernel function types
type KernelFunction = Callable[[Tensor, Tensor], Tensor]
type FeatureMapFunction = Callable[[Tensor], Tensor]

# Kernel type aliases for attention layers
type KernelType = Literal["gaussian", "softmax"]
type SpectralKernelType = Literal["gaussian", "polynomial", "spectral"]

# Transform type aliases for attention layers
type TransformLSTType = Literal["dct", "dst", "hadamard", "mixed"]

# Filter types for spectral methods
type SpectralFilter = ComplexTensor
type LearnableFilter = nn.Parameter

# Mode selection for spectral methods
type ModeSelection = Literal["top", "random", "learned"]
type ModeIndices = LongTensor

# Window functions for spectral analysis
WindowFunction = Literal[
    "hann",
    "hamming",
    "blackman",
    "bartlett",
    "kaiser",
    "tukey",
    "none",
]

# Export all aliases
__all__: list[str] = [
    "ActivationType",
    "AttentionMask",
    "AttentionModuleType",
    "AutocastDType",
    "BatchDict",
    # Shape types
    "BatchSize",
    "BatchTuple",
    "BlockModuleType",
    "BoolTensor",
    "CausalMask",
    "CheckpointDict",
    "ComplexTensor",
    "ComponentClass",
    "ComponentFactory",
    "ComponentType",
    # Configuration types
    "ConfigDict",
    # Data types
    "Device",
    "FFTNorm",
    "FeatureMapFunction",
    "FilterActivationType",
    # Fourier and spectral types
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
    # Function types
    "LossFunction",
    "LossOutput",
    "MetricFunction",
    "MixedPrecisionDType",
    "MixingModuleType",
    "ModeIndices",
    "ModeSelection",
    "ModeTruncation",
    "ModelModuleType",
    "ModelOutput",
    "ModelType",
    # Module types
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
    "SpectralFilter",
    "SpectralKernelType",
    "StateDict",
    # Tensor types
    "Tensor",
    "TrainingConfig",
    "TransformLSTType",
    "TransformModuleType",
    # Transform and model types
    "TransformType",
    "VocabSize",
    "WaveletType",
    "WindowFunction",
    "WorldSize",
]
