r"""Spectral attention layer implementations with linear complexity.

Provides attention mechanisms based on spectral methods and kernel approximations,
achieving linear or logarithmic complexity compared to the quadratic complexity of
standard attention. Implementations include Random Fourier Features, orthogonal
transforms, and hybrid approaches.

Modules
-------
lst
    Linear Spectral Transform attention implementations.
spectral
    Kernel-based spectral attention mechanisms.

Classes
-------
DCTAttention
    Specialized LST attention using discrete cosine transform.
HadamardAttention
    Fast attention using Hadamard transform operations.
KernelAttention
    General kernel-based attention with various kernel options.
LSTAttention
    Linear Spectral Transform attention with configurable transforms.
MixedSpectralAttention
    Multi-transform attention combining multiple spectral methods.
PerformerAttention
    Performer-style attention with FAVOR+ algorithm.
SpectralAttention
    Multi-head spectral attention using random Fourier features.

Examples
--------
Using spectral attention with RFF:

>>> import torch
>>> from spectrans.layers.attention import SpectralAttention
>>>
>>> attn = SpectralAttention(hidden_dim=512, num_heads=8, num_features=256)
>>> x = torch.randn(32, 100, 512)
>>> output = attn(x)
>>> assert output.shape == x.shape

Using LST attention with DCT:

>>> from spectrans.layers.attention import DCTAttention
>>>
>>> attn = DCTAttention(hidden_dim=512, num_heads=8)
>>> x = torch.randn(16, 128, 512)
>>> output = attn(x)

Using Performer attention:

>>> from spectrans.layers.attention import PerformerAttention
>>>
>>> attn = PerformerAttention(
...     hidden_dim=768,
...     num_heads=12,
...     num_features=256,
...     use_orthogonal=True
... )
>>> output = attn(x)

Notes
-----
Complexity Analysis:

Standard attention requires $O(n^2 d)$ time and $O(n^2)$ memory. Spectral attention
reduces this to $O(n d k)$ time and $O(n k)$ memory, where $k$ is the number of random
features. LST attention achieves $O(n d \log n)$ time with $O(n d)$ memory. Performer
uses $O(n d k)$ time with orthogonal features. Here $n$ is sequence length and $d$ is
dimension.

Kernel approximation quality scales as $O(1/\sqrt{k})$ for random features.

References
----------
Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song,
Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin,
Lukasz Kaiser, David Belanger, Lucy Colwell, and Adrian Weller. 2021.
Rethinking attention with performers. In Proceedings of the International
Conference on Learning Representations (ICLR).

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020.
Transformers are RNNs: Fast autoregressive transformers with linear attention.
In Proceedings of the 37th International Conference on Machine Learning (ICML),
pages 5156-5165.

See Also
--------
[`spectrans.kernels`][] : Kernel functions used by attention mechanisms.
[`spectrans.transforms`][] : Spectral transforms used by LST attention.
[`spectrans.layers`][] : Parent module containing all layer implementations.
"""

from .lst import (
    DCTAttention,
    HadamardAttention,
    LSTAttention,
    MixedSpectralAttention,
)
from .spectral import (
    KernelAttention,
    PerformerAttention,
    SpectralAttention,
)

# Public API - alphabetically sorted
__all__ = [
    "DCTAttention",
    "HadamardAttention",
    "KernelAttention",
    "LSTAttention",
    "MixedSpectralAttention",
    "PerformerAttention",
    "SpectralAttention",
]
