r"""Kernel functions for spectral transformers.

This module provides kernel functions and feature maps used in spectral
attention mechanisms and other kernel-based methods. It includes both
explicit kernel evaluations and implicit representations through random
feature maps.

The kernels approximate attention mechanisms with linear complexity
through random feature expansions and spectral decompositions.

Modules
-------
base
    Base classes and interfaces for kernel functions.
rff
    Random Fourier Features implementations.
spectral
    Spectral kernel functions and decompositions.

Classes
-------
CosineKernel
    Cosine similarity kernel.
FourierKernel
    Kernel defined in Fourier domain.
GaussianRFFKernel
    Gaussian kernel with RFF approximation.
KernelFunction
    Abstract base class for kernel functions.
KernelType
    Type literal for kernel selection.
LaplacianRFFKernel
    Laplacian kernel with RFF approximation.
LearnableSpectralKernel
    Spectral kernel with learnable parameters.
OrthogonalRandomFeatures
    Orthogonal variant of random features.
PolynomialKernel
    Polynomial kernel implementation.
PolynomialSpectralKernel
    Polynomial kernel with spectral decomposition.
RFFAttentionKernel
    RFF designed for attention mechanisms.
RandomFeatureMap
    Abstract base class for random feature approximations.
ShiftInvariantKernel
    Base class for shift-invariant kernels.
SpectralKernel
    Base class for spectral kernels.
TruncatedSVDKernel
    Kernel approximation via truncated SVD.

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

Notes
-----
Kernel approximation achieves linear complexity attention mechanisms through
random feature expansions and spectral decompositions. Random Fourier Features,
based on Bochner's theorem, approximate shift-invariant kernels via the
factorization $k(\mathbf{x}, \mathbf{y}) \approx \varphi(\mathbf{x})^T \varphi(\mathbf{y})$
where $\varphi$ maps inputs to a feature space.

Spectral decomposition methods leverage eigendecomposition for kernel
computation through low-rank approximations, while orthogonal feature
variants apply orthogonalized random projections to reduce approximation
variance. The approximation error decreases with $O(1/\sqrt{D})$ where
$D$ is the number of random features.

References
----------
Ali Rahimi and Benjamin Recht. 2007. Random features for large-scale kernel machines.
In Advances in Neural Information Processing Systems 20 (NeurIPS 2007), pages 1177-1184.

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane,
Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger,
Lucy Colwell, and Adrian Weller. 2021. Rethinking attention with performers. In Proceedings
of the International Conference on Learning Representations (ICLR).

See Also
--------
[`spectrans.layers.attention`][] : Attention layers using these kernels.
[`spectrans.kernels.base`][] : Base kernel interfaces.
[`spectrans.kernels.rff`][] : Random Fourier Features implementations.
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

# Public API - alphabetically sorted
__all__ = [
    "CosineKernel",
    "FourierKernel",
    "GaussianRFFKernel",
    "KernelFunction",
    "KernelType",
    "LaplacianRFFKernel",
    "LearnableSpectralKernel",
    "OrthogonalRandomFeatures",
    "PolynomialKernel",
    "PolynomialSpectralKernel",
    "RFFAttentionKernel",
    "RandomFeatureMap",
    "ShiftInvariantKernel",
    "SpectralKernel",
    "TruncatedSVDKernel",
]
