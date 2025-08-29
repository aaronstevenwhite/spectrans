"""Type definitions and aliases for the spectrans library."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from spectrans.core.base import (
        AttentionLayer,
        BaseModel,
        MixingLayer,
        SpectralTransform,
        TransformerBlock,
    )

# Tensor type aliases
type Tensor = torch.Tensor
type ComplexTensor = torch.Tensor  # Complex-valued tensor
type LongTensor = torch.LongTensor
type BoolTensor = torch.BoolTensor

# Shape type aliases
BatchSize = int
SequenceLength = int
HiddenDim = int
NumHeads = int
HeadDim = int
IntermediateDim = int
NumLayers = int
VocabSize = int
NumClasses = int

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
type ConfigDict = dict[str, Any]
type ParamsDict = dict[str, Any]

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

# Note: Use torch.dtype directly for dtype type hints instead of creating an alias

# Device types
type Device = torch.device | str | None

# Optional types
type OptionalTensor = Tensor | None
type OptionalModule = nn.Module | None

# Fourier mode types
FourierModes = int  # Number of Fourier modes to keep
type ModeTruncation = tuple[int, ...]  # Mode truncation per dimension

# Random feature types
NumRandomFeatures = int
RandomSeed = int | None

# Complexity information
type ComplexityInfo = dict[Literal["time", "space"], str]

# Training configuration
type OptimizerConfig = dict[str, Any]
type SchedulerConfig = dict[str, Any]
type TrainingConfig = dict[str, Any]

# Model state
type StateDict = dict[str, Tensor]
type CheckpointDict = dict[str, Any]

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
PositionEncodingType = Literal[
    "learned",
    "sinusoidal",
    "rotary",
    "alibi",
    "none",
]

# Output types for different model modes
type ModelOutput = Tensor | tuple[Tensor, ...]
type LossOutput = Tensor | tuple[Tensor, dict[str, Tensor]]

# Batch types
type BatchDict = dict[str, Tensor]
type BatchTuple = tuple[Tensor, ...]

# Gradient clipping types
GradientClipValue = float | None
GradientClipNorm = float | None

# Mixed precision types
MixedPrecisionDType = Literal["float16", "bfloat16", "float32"]
type AutocastDType = torch.dtype | None

# Distributed training types
WorldSize = int
Rank = int
LocalRank = int

# Kernel function types
type KernelFunction = Callable[[Tensor, Tensor], Tensor]
type FeatureMapFunction = Callable[[Tensor], Tensor]

# Filter types for spectral methods
type SpectralFilter = ComplexTensor
type LearnableFilter = nn.Parameter

# Mode selection for spectral methods
ModeSelection = Literal["top", "random", "learned"]
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

# Export all type aliases
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
    # Other types
    "ComplexityInfo",
    "ComponentClass",
    "ComponentFactory",
    "ComponentType",
    # Configuration types
    "ConfigDict",
    # Data types
    "Device",
    "FFTNorm",
    "FeatureMapFunction",
    # Fourier and spectral types
    "FourierModes",
    "GradientClipNorm",
    "GradientClipValue",
    "HeadDim",
    "HiddenDim",
    "InitializationType",
    "IntermediateDim",
    "KernelFunction",
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
    "PaddingSize",
    "PaddingType",
    "ParamsDict",
    "PositionEncodingType",
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
    "StateDict",
    # Tensor types
    "Tensor",
    "TrainingConfig",
    "TransformModuleType",
    # Transform and model types
    "TransformType",
    "VocabSize",
    "WaveletType",
    "WindowFunction",
    "WorldSize",
]
