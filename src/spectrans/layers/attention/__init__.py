"""Attention layer implementations for spectral transformers.

This module provides efficient attention mechanisms based on spectral methods
and kernel approximations, achieving linear or logarithmic complexity compared
to the quadratic complexity of standard attention.

Classes
-------
SpectralAttention
    Multi-head spectral attention using RFF approximation.
PerformerAttention
    Performer-style attention with FAVOR+ algorithm.
KernelAttention
    General kernel-based attention with various kernel options.
LSTAttention
    Linear Spectral Transform attention with orthogonal transforms.
DCTAttention
    Attention using Discrete Cosine Transform.
HadamardAttention
    Attention using fast Hadamard transform.
MixedSpectralAttention
    Mixed spectral attention using multiple transform types.

Examples
--------
Using spectral attention:

>>> from spectrans.layers.attention import SpectralAttention
>>> attn = SpectralAttention(hidden_dim=512, num_heads=8)
>>> x = torch.randn(32, 100, 512)
>>> output = attn(x)
>>> assert output.shape == x.shape

Using LST attention with DCT:

>>> from spectrans.layers.attention import DCTAttention
>>> attn = DCTAttention(hidden_dim=512, num_heads=8)
>>> output = attn(x)

See Also
--------
spectrans.kernels : Kernel functions used by attention mechanisms.
spectrans.transforms : Spectral transforms used by LST attention.
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

__all__ = [
    # Spectral attention variants
    "SpectralAttention",
    "PerformerAttention",
    "KernelAttention",
    # LST attention variants
    "LSTAttention",
    "DCTAttention",
    "HadamardAttention",
    "MixedSpectralAttention",
]
