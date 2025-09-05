r"""Comprehensive layer implementations for spectral transformers.

This module provides a complete collection of spectral transformer layers that
replace traditional attention mechanisms with efficient spectral operations.
The layers are organized into three main categories: mixing layers, attention
layers, and neural operators, each optimized for different use cases while
maintaining compatibility with standard transformer architectures.

Classes
-------
MixingLayer
    Base class for spectral mixing operations.
UnitaryMixingLayer
    Base class for energy-preserving mixing transforms.
FilterMixingLayer
    Base class for learnable frequency domain filters.
FourierMixing
    2D FFT mixing for both sequence and feature dimensions (FNet).
FourierMixing1D
    1D FFT mixing along sequence dimension only.
RealFourierMixing
    Memory-efficient real FFT variant for real-valued inputs.
SeparableFourierMixing
    Configurable sequence and/or feature mixing.
GlobalFilterMixing
    Learnable complex filters in frequency domain (GFNet).
GlobalFilterMixing2D
    2D variant with filtering in both dimensions.
AdaptiveGlobalFilter
    Enhanced global filter with adaptive initialization.
AFNOMixing
    Adaptive Fourier Neural Operator with mode truncation.
WaveletMixing
    1D wavelet mixing using discrete wavelet transform.
WaveletMixing2D
    2D wavelet mixing for spatial data processing.
SpectralAttention
    Multi-head spectral attention using random Fourier features.
PerformerAttention
    Performer-style attention with FAVOR+ algorithm.
KernelAttention
    General kernel-based attention with various kernel options.
LSTAttention
    Linear Spectral Transform attention with configurable transforms.
DCTAttention
    Specialized LST attention using discrete cosine transform.
HadamardAttention
    Fast attention using Hadamard transform operations.
MixedSpectralAttention
    Multi-transform attention combining multiple spectral methods.
FourierNeuralOperator
    Base FNO layer for learning operators in function spaces.
FNOBlock
    Complete FNO block with spectral convolution and feedforward.
SpectralConv1d
    1D spectral convolution operator for sequence data.
SpectralConv2d
    2D spectral convolution operator for image-like data.

Examples
--------
Basic Fourier mixing layer (FNet-style):

>>> import torch
>>> from spectrans.layers import FourierMixing
>>> 
>>> # Create Fourier mixing layer
>>> mixer = FourierMixing(hidden_dim=768)
>>> x = torch.randn(32, 512, 768)  # (batch, sequence, hidden)
>>> output = mixer(x)
>>> assert output.shape == x.shape

Global filter mixing with learnable parameters:

>>> from spectrans.layers import GlobalFilterMixing
>>> 
>>> # Create global filter with learnable complex weights
>>> filter_layer = GlobalFilterMixing(
...     hidden_dim=512,
...     sequence_length=1024,
...     activation='sigmoid'
... )
>>> x = torch.randn(16, 1024, 512)
>>> output = filter_layer(x)

Spectral attention with random Fourier features:

>>> from spectrans.layers import SpectralAttention
>>> 
>>> # Create spectral attention layer
>>> attention = SpectralAttention(
...     hidden_dim=768,
...     num_heads=12,
...     num_features=256
... )
>>> x = torch.randn(8, 256, 768)
>>> output = attention(x)

Fourier Neural Operator for function learning:

>>> from spectrans.layers import FourierNeuralOperator
>>> 
>>> # Create FNO layer for continuous function approximation
>>> fno = FourierNeuralOperator(
...     hidden_dim=256,
...     modes=32
... )
>>> x = torch.randn(16, 128, 256)
>>> output = fno(x)

Wavelet mixing for multiresolution analysis:

>>> from spectrans.layers import WaveletMixing
>>> 
>>> # Create wavelet mixing layer
>>> wavelet_layer = WaveletMixing(
...     hidden_dim=512,
...     wavelet='db4',
...     levels=3
... )
>>> x = torch.randn(32, 256, 512)
>>> output = wavelet_layer(x)

Hybrid mixing with multiple transforms:

>>> from spectrans.layers import MixedSpectralAttention
>>> 
>>> # Combine multiple spectral transforms
>>> hybrid_attention = MixedSpectralAttention(
...     hidden_dim=768,
...     num_heads=8,
...     transforms=['dct', 'dst', 'hadamard'],
...     mixing_weights=[0.4, 0.3, 0.3]
... )
>>> x = torch.randn(16, 512, 768)
>>> output = hybrid_attention(x)

Notes
-----
**Layer Categories and Complexity:**

1. **Mixing Layers** (:math:`O(n \log n)` or :math:`O(n)` complexity):
   - Parameter-free: FourierMixing variants using FFT operations
   - Learnable filters: Global filters and AFNO with trainable parameters
   - Multiresolution: Wavelet transforms for hierarchical processing

2. **Attention Layers** (Linear :math:`O(n)` complexity):
   - Kernel approximation: Random Fourier Features and orthogonal features
   - Transform-based: DCT, DST, and Hadamard transforms
   - Hybrid approaches: Multiple transforms with learnable mixing

3. **Neural Operators** (:math:`O(k \cdot d^2 + n \log n)` complexity):
   - Function space learning: Map between infinite-dimensional spaces
   - Resolution invariance: Learn operators independent of discretization
   - Spectral parameterization: Efficient representation in Fourier domain

**Mathematical Foundation:**

All layers leverage the convolution theorem for efficient global mixing:

.. math::
    \mathcal{F}[f \star g] = \mathcal{F}[f] \odot \mathcal{F}[g]

This enables replacement of quadratic attention :math:`O(n^2)` with logarithmic 
or linear complexity spectral operations.

**Key Advantages:**

- **Efficiency**: Subquadratic complexity compared to standard attention
- **Global Mixing**: All positions interact through spectral domain operations  
- **Mathematical Rigor**: Based on well-established signal processing principles
- **Hardware Optimization**: Leverage highly optimized FFT implementations
- **Memory Efficiency**: Reduced memory footprint for long sequences

**Integration Guidelines:**

All layers follow consistent interfaces and can be easily substituted for
standard attention in transformer architectures. They support:

- Batch processing with variable sequence lengths
- Gradient flow for end-to-end training  
- Mixed precision and distributed training
- Configuration-based instantiation via YAML

References
----------
.. [1] Lee-Thorp, J., et al., "FNet: Mixing Tokens with Fourier Transforms", 
       NAACL 2022.
.. [2] Rao, Y., et al., "Global Filter Networks for Image Classification",
       NeurIPS 2021.
.. [3] Guibas, J., et al., "Adaptive Fourier Neural Operators", ICLR 2022.
.. [4] Li, Z., et al., "Fourier Neural Operator for Parametric Partial
       Differential Equations", ICLR 2021.
.. [5] Choromanski, K., et al., "Rethinking Attention with Performers", 
       ICLR 2021.

See Also
--------
spectrans.transforms : Underlying spectral transform implementations.
spectrans.models : Complete model implementations using these layers.
spectrans.blocks : Transformer blocks that compose these layers.
"""

from .attention import (
    DCTAttention,
    HadamardAttention,
    KernelAttention,
    LSTAttention,
    MixedSpectralAttention,
    PerformerAttention,
    SpectralAttention,
)
from .mixing import (
    AFNOMixing,
    AdaptiveGlobalFilter,
    FilterMixingLayer,
    FourierMixing,
    FourierMixing1D,
    GlobalFilterMixing,
    GlobalFilterMixing2D,
    MixingLayer,
    RealFourierMixing,
    SeparableFourierMixing,
    UnitaryMixingLayer,
    WaveletMixing,
    WaveletMixing2D,
)
from .operators import (
    FNOBlock,
    FourierNeuralOperator,
    SpectralConv1d,
    SpectralConv2d,
)

__all__ = [
    # Base classes
    "MixingLayer",
    "UnitaryMixingLayer", 
    "FilterMixingLayer",
    # Fourier mixing layers
    "FourierMixing",
    "FourierMixing1D",
    "RealFourierMixing",
    "SeparableFourierMixing",
    # Global filter mixing layers
    "GlobalFilterMixing",
    "GlobalFilterMixing2D",
    "AdaptiveGlobalFilter",
    # Advanced mixing layers
    "AFNOMixing",
    "WaveletMixing",
    "WaveletMixing2D",
    # Spectral attention layers
    "SpectralAttention",
    "PerformerAttention",
    "KernelAttention",
    "LSTAttention",
    "DCTAttention",
    "HadamardAttention",
    "MixedSpectralAttention",
    # Neural operators
    "FourierNeuralOperator",
    "FNOBlock", 
    "SpectralConv1d",
    "SpectralConv2d",
]
