r"""Spectral mixing layer implementations for token mixing.

Provides spectral mixing layers as alternatives to attention mechanisms. These layers
operate in frequency domains using transforms like FFT, maintaining linear or log-linear
computational complexity for token mixing operations.

Mixing layers implement different mathematical approaches including parameter-free
Fourier mixing (FNet style), learnable complex filters in frequency domain (GFNet style),
and variants with adaptive initialization and multi-dimensional operations.

Modules
-------
afno
    Adaptive Fourier Neural Operator mixing implementations.
base
    Base classes and interfaces for mixing layers.
fourier
    Fourier transform-based mixing layers.
global_filter
    Global filter networks with learnable parameters.
wavelet
    Wavelet transform-based mixing layers.

Classes
-------
AFNOMixing
    Adaptive Fourier Neural Operator with mode truncation.
AdaptiveGlobalFilter
    Enhanced global filter with adaptive initialization.
FilterMixingLayer
    Base class for learnable frequency domain filters.
FourierMixing
    2D FFT mixing for both sequence and feature dimensions.
FourierMixing1D
    1D FFT mixing along sequence dimension only.
GlobalFilterMixing
    Learnable complex filters in frequency domain.
GlobalFilterMixing2D
    2D variant with filtering in both dimensions.
MixingLayer
    Base class for spectral mixing operations.
RealFourierMixing
    Memory-efficient real FFT variant.
SeparableFourierMixing
    Configurable sequence and/or feature mixing.
UnitaryMixingLayer
    Base class for energy-preserving mixing transforms.
WaveletMixing
    1D wavelet mixing using discrete wavelet transform.
WaveletMixing2D
    2D wavelet mixing for spatial data processing.

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

Adaptive filtering:

>>> from spectrans.layers.mixing import AdaptiveGlobalFilter
>>> adaptive_mixer = AdaptiveGlobalFilter(
...     hidden_dim=768, sequence_length=512,
...     adaptive_initialization=True, filter_regularization=0.01
... )
>>> adaptive_output = adaptive_mixer(input_tensor)

Notes
-----
Complexity Comparison:

Traditional attention has $O(n^2 d)$ complexity. Fourier mixing reduces this to
$O(nd \log n)$. Global filtering uses $O(nd \log n)$ complexity plus learnable parameters.

All mixing layers support batch processing with consistent behavior, gradient computation
for end-to-end training, shape preservation where output shape equals input shape, and
mathematical property verification for energy and orthogonality.

References
----------
James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon. 2022.
FNet: Mixing tokens with Fourier transforms. In Proceedings of the 2022 Conference
of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (NAACL-HLT), pages 4296-4313, Seattle.

Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou. 2021.
Global filter networks for image classification. In Advances in Neural
Information Processing Systems 34 (NeurIPS 2021), pages 980-993.

John Guibas, Morteza Mardani, Zongyi Li, Andrew Tao, Anima Anandkumar, and
Bryan Catanzaro. 2022. Adaptive Fourier neural operators: Efficient token
mixers for transformers. In Proceedings of the International Conference on
Learning Representations (ICLR).

See Also
--------
[`spectrans.layers.mixing.base`][] : Base classes and interfaces.
[`spectrans.transforms`][] : Underlying spectral transform implementations.
[`spectrans.blocks`][] : Transformer blocks that use these mixing layers.
"""

from .afno import AFNOMixing
from .base import FilterMixingLayer, MixingLayer, UnitaryMixingLayer

# Import Fourier-based mixing layers
from .fourier import FourierMixing, FourierMixing1D, RealFourierMixing, SeparableFourierMixing

# Import global filter mixing layers
from .global_filter import AdaptiveGlobalFilter, GlobalFilterMixing, GlobalFilterMixing2D
from .wavelet import WaveletMixing, WaveletMixing2D

# Public API - alphabetically sorted
__all__ = [
    "AFNOMixing",
    "AdaptiveGlobalFilter",
    "FilterMixingLayer",
    "FourierMixing",
    "FourierMixing1D",
    "GlobalFilterMixing",
    "GlobalFilterMixing2D",
    "MixingLayer",
    "RealFourierMixing",
    "SeparableFourierMixing",
    "UnitaryMixingLayer",
    "WaveletMixing",
    "WaveletMixing2D",
]
