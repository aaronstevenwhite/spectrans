r"""Layer implementations for spectral transformers.

Provides spectral transformer layers that replace traditional attention mechanisms
with spectral operations. The layers are organized into three categories: mixing layers,
attention layers, and neural operators for different use cases with standard
transformer architecture compatibility.

Modules
-------
attention
    Spectral attention mechanisms with linear complexity.
mixing
    Token mixing layers using spectral transforms.
operators
    Fourier neural operators for function space learning.

Classes
-------
AdaptiveGlobalFilter
    Enhanced global filter with adaptive initialization.
AFNOMixing
    Adaptive Fourier Neural Operator with mode truncation.
DCTAttention
    Specialized LST attention using discrete cosine transform.
FilterMixingLayer
    Base class for learnable frequency domain filters.
FNOBlock
    FNO block with spectral convolution and feedforward.
FourierMixing
    2D FFT mixing for both sequence and feature dimensions (FNet).
FourierMixing1D
    1D FFT mixing along sequence dimension only.
FourierNeuralOperator
    Base FNO layer for learning operators in function spaces.
GlobalFilterMixing
    Learnable complex filters in frequency domain (GFNet).
GlobalFilterMixing2D
    2D variant with filtering in both dimensions.
HadamardAttention
    Fast attention using Hadamard transform operations.
KernelAttention
    General kernel-based attention with various kernel options.
LSTAttention
    Linear Spectral Transform attention with configurable transforms.
MixedSpectralAttention
    Multi-transform attention combining multiple spectral methods.
MixingLayer
    Base class for spectral mixing operations.
PerformerAttention
    Performer-style attention with FAVOR+ algorithm.
RealFourierMixing
    Memory-efficient real FFT variant for real-valued inputs.
SeparableFourierMixing
    Configurable sequence and/or feature mixing.
SpectralAttention
    Multi-head spectral attention using random Fourier features.
SpectralConv1d
    1D spectral convolution operator for sequence data.
SpectralConv2d
    2D spectral convolution operator for image-like data.
UnitaryMixingLayer
    Base class for energy-preserving mixing transforms.
WaveletMixing
    1D wavelet mixing using discrete wavelet transform.
WaveletMixing2D
    2D wavelet mixing for spatial data processing.

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

Notes
-----
Layer Categories and Complexity:

Mixing layers have $O(n \log n)$ or $O(n)$ complexity. Parameter-free variants use FFT operations,
while learnable filters like global filters and AFNO include trainable parameters. Multiresolution
approaches use wavelet transforms for hierarchical processing.

Attention layers achieve linear $O(n)$ complexity through kernel approximation with Random Fourier
Features and orthogonal features, transform-based methods using DCT, DST, and Hadamard transforms,
or hybrid approaches combining multiple transforms with learnable mixing.

Neural operators have $O(k \cdot d^2 + n \log n)$ complexity where $k$ is the number of modes and
$d$ is the dimension. These operators map between infinite-dimensional function spaces with
resolution-invariant learning independent of discretization through spectral parameterization
in the Fourier domain.

All layers use the convolution theorem for global mixing:

$$
\mathcal{F}[f \star g] = \mathcal{F}[f] \odot \mathcal{F}[g]
$$

This replaces quadratic attention $O(n^2)$ with logarithmic or linear complexity spectral operations.

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

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik
Bhattacharya, Andrew Stuart, and Anima Anandkumar. 2021. Fourier neural
operator for parametric partial differential equations. In Proceedings of
the International Conference on Learning Representations (ICLR).

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song,
Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin,
Lukasz Kaiser, David Belanger, Lucy Colwell, and Adrian Weller. 2021.
Rethinking attention with performers. In Proceedings of the International
Conference on Learning Representations (ICLR).

See Also
--------
[`spectrans.transforms`][] : Underlying spectral transform implementations.
[`spectrans.models`][] : Model implementations using these layers.
[`spectrans.blocks`][] : Transformer blocks that compose these layers.
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
    AdaptiveGlobalFilter,
    AFNOMixing,
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

# Public API - alphabetically sorted
__all__ = [
    "AFNOMixing",
    "AdaptiveGlobalFilter",
    "DCTAttention",
    "FNOBlock",
    "FilterMixingLayer",
    "FourierMixing",
    "FourierMixing1D",
    "FourierNeuralOperator",
    "GlobalFilterMixing",
    "GlobalFilterMixing2D",
    "HadamardAttention",
    "KernelAttention",
    "LSTAttention",
    "MixedSpectralAttention",
    "MixingLayer",
    "PerformerAttention",
    "RealFourierMixing",
    "SeparableFourierMixing",
    "SpectralAttention",
    "SpectralConv1d",
    "SpectralConv2d",
    "UnitaryMixingLayer",
    "WaveletMixing",
    "WaveletMixing2D",
]
