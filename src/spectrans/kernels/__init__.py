"""Kernel functions for spectral transformers.

This module provides kernel functions and feature maps used in spectral
attention mechanisms and other kernel-based methods. It includes both
explicit kernel evaluations and implicit representations through random
feature maps.

The kernels support efficient approximations of attention mechanisms
with linear complexity, enabling scalable transformer architectures.

Classes
-------
KernelFunction
    Abstract base class for kernel functions.
RandomFeatureMap
    Abstract base class for random feature approximations.
ShiftInvariantKernel
    Base class for shift-invariant kernels.
PolynomialKernel
    Polynomial kernel implementation.
CosineKernel
    Cosine similarity kernel.
GaussianRFFKernel
    Gaussian kernel with RFF approximation.
LaplacianRFFKernel
    Laplacian kernel with RFF approximation.
OrthogonalRandomFeatures
    Orthogonal variant of random features.
RFFAttentionKernel
    RFF designed for attention mechanisms.
SpectralKernel
    Base class for spectral kernels.
PolynomialSpectralKernel
    Polynomial kernel with spectral decomposition.
TruncatedSVDKernel
    Kernel approximation via truncated SVD.
LearnableSpectralKernel
    Spectral kernel with learnable parameters.
FourierKernel
    Kernel defined in Fourier domain.

Examples
--------
Using Gaussian RFF kernel:

>>> from spectrans.kernels import GaussianRFFKernel
>>> kernel = GaussianRFFKernel(input_dim=64, num_features=256, sigma=1.0)
>>> x = torch.randn(32, 100, 64)
>>> features = kernel(x)
>>> assert features.shape == (32, 100, 256)

Using learnable spectral kernel:

>>> from spectrans.kernels import LearnableSpectralKernel
>>> kernel = LearnableSpectralKernel(input_dim=64, rank=16)
>>> K = kernel.compute(x, x)
>>> assert K.shape == (32, 100, 100)

See Also
--------
spectrans.layers.attention : Attention layers using these kernels.
"""

from .base import (
    CosineKernel,
    KernelFunction,
    KernelType,
    PolynomialKernel,
    RandomFeatureMap,
    ShiftInvariantKernel,
)
from .rff import (
    GaussianRFFKernel,
    LaplacianRFFKernel,
    OrthogonalRandomFeatures,
    RFFAttentionKernel,
)
from .spectral import (
    FourierKernel,
    LearnableSpectralKernel,
    PolynomialSpectralKernel,
    SpectralKernel,
    TruncatedSVDKernel,
)

__all__ = [
    # Base interfaces
    "KernelFunction",
    "KernelType",
    "RandomFeatureMap",
    "ShiftInvariantKernel",
    # Basic kernels
    "CosineKernel",
    "PolynomialKernel",
    # RFF kernels
    "GaussianRFFKernel",
    "LaplacianRFFKernel",
    "OrthogonalRandomFeatures",
    "RFFAttentionKernel",
    # Spectral kernels
    "FourierKernel",
    "LearnableSpectralKernel",
    "PolynomialSpectralKernel",
    "SpectralKernel",
    "TruncatedSVDKernel",
]
