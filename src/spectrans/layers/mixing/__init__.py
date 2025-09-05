r"""Mixing layer implementations.

This module implements spectral mixing layers that serve as alternatives to attention
mechanisms. These layers operate in frequency domains using transforms like FFT,
maintaining linear or log-linear computational complexity for token mixing operations.

The mixing layers implement different mathematical approaches:
- Fourier mixing: Parameter-free frequency domain mixing (FNet style)
- Global filtering: Learnable complex filters in frequency domain (GFNet style)
- Advanced variants: Adaptive initialization, regularization, and multi-dimensional mixing

All mixing layers inherit from the base classes in .base and maintain consistent
interfaces for easy integration into transformer architectures.

Available Mixing Layers
-----------------------
Fourier-based mixing (parameter-free):
- FourierMixing: 2D FFT mixing for both sequence and feature dimensions
- FourierMixing1D: 1D FFT mixing along sequence dimension only
- RealFourierMixing: Memory-efficient real FFT variant
- SeparableFourierMixing: Configurable sequence and/or feature mixing

Global filter mixing (learnable parameters):
- GlobalFilterMixing: Learnable complex filters in frequency domain
- GlobalFilterMixing2D: 2D variant with filtering in both dimensions
- AdaptiveGlobalFilter: Enhanced with adaptive initialization and regularization

Base classes:
- MixingLayer: Base class for mixing operations
- UnitaryMixingLayer: Base for energy-preserving transforms
- FilterMixingLayer: Base for frequency domain filtering

Examples
--------
Basic Fourier mixing:

>>> from spectrans.layers.mixing import FourierMixing
>>> mixer = FourierMixing(hidden_dim=768)
>>> output = mixer(input_tensor)

Global filter with learnable parameters:

>>> from spectrans.layers.mixing import GlobalFilterMixing
>>> filter_mixer = GlobalFilterMixing(hidden_dim=768, sequence_length=512)
>>> filtered_output = filter_mixer(input_tensor)

Advanced adaptive filtering:

>>> from spectrans.layers.mixing import AdaptiveGlobalFilter
>>> adaptive_mixer = AdaptiveGlobalFilter(
...     hidden_dim=768, sequence_length=512,
...     adaptive_initialization=True, filter_regularization=0.01
... )
>>> adaptive_output = adaptive_mixer(input_tensor)

Notes
-----
Complexity Comparison:
- Traditional attention: :math:`O(n^2 d)`
- Fourier mixing: :math:`O(nd \log n)`
- Global filtering: :math:`O(nd \log n)` + learnable parameters

All mixing layers support:
- Batch processing with consistent behavior
- Gradient computation for end-to-end training
- Shape preservation (output shape = input shape)
- Mathematical property verification (energy, orthogonality)

See Also
--------
spectrans.layers.mixing.base : Base classes and interfaces
spectrans.transforms : Underlying spectral transform implementations
spectrans.blocks : Transformer blocks that use these mixing layers
"""

from .afno import AFNOMixing
from .base import FilterMixingLayer, MixingLayer, UnitaryMixingLayer

# Import Fourier-based mixing layers
from .fourier import (
    FourierMixing,
    FourierMixing1D,
    RealFourierMixing,
    SeparableFourierMixing,
)

# Import global filter mixing layers
from .global_filter import (
    AdaptiveGlobalFilter,
    GlobalFilterMixing,
    GlobalFilterMixing2D,
)
from .wavelet import WaveletMixing, WaveletMixing2D

__all__: list[str] = [
    "AFNOMixing",
    "AdaptiveGlobalFilter",
    "FilterMixingLayer",
    # Fourier mixing layers
    "FourierMixing",
    "FourierMixing1D",
    # Global filter mixing layers
    "GlobalFilterMixing",
    "GlobalFilterMixing2D",
    # Base classes
    "MixingLayer",
    "RealFourierMixing",
    "SeparableFourierMixing",
    "UnitaryMixingLayer",
    # Wavelet mixing layers
    "WaveletMixing",
    "WaveletMixing2D",
]
