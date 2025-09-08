r"""Base interfaces and classes for kernel functions and random feature maps.

This module defines the abstract base classes for kernel functions used in
spectral attention mechanisms and other kernel-based methods. It provides
interfaces for both explicit kernel evaluations and implicit feature map
representations through random features.

The kernel framework supports various approximation techniques including
Random Fourier Features (RFF), polynomial kernels, and spectral kernels,
enabling efficient computation of attention mechanisms with linear complexity.

Classes
-------
KernelFunction
    Abstract base class for kernel functions $k(x, y)$.
RandomFeatureMap
    Abstract base class for random feature approximations.
ShiftInvariantKernel
    Base class for shift-invariant (stationary) kernels.

Examples
--------
Implementing a custom kernel:

>>> import torch
>>> from spectrans.kernels.base import KernelFunction
>>> class LinearKernel(KernelFunction):
...     def compute(self, x, y):
...         return torch.matmul(x, y.transpose(-2, -1))

Using a random feature map:

>>> from spectrans.kernels.base import RandomFeatureMap
>>> class CustomFeatureMap(RandomFeatureMap):
...     def __init__(self, input_dim, num_features):
...         super().__init__(input_dim, num_features)
...         # Initialize random parameters
...     def forward(self, x):
...         # Return feature mapped tensor
...         pass

Notes
-----
Kernel Approximation Theory:

For shift-invariant kernels, Bochner's theorem states that:

$$
k(x - y) = \int p(\omega) \exp(i\omega^T(x-y)) d\omega
$$

This enables Random Fourier Features approximation:

$$
k(x, y) \approx \varphi(x)^T \varphi(y)
$$

Where:

$$
\varphi(x) = \sqrt{\frac{2}{D}} \left[\cos(\omega_1^Tx + b_1), \ldots, \cos(\omega_D^Tx + b_D)\right]
$$

The approximation quality improves with $O(1/\sqrt{D})$ where $D$ is the number
of random features.

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
spectrans.kernels.rff : Random Fourier Features implementation.
spectrans.kernels.spectral : Spectral kernel functions.
"""

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn

from ..core.types import Tensor


class KernelFunction(ABC):
    r"""Abstract base class for kernel functions.

    A kernel function $k(x, y)$ defines a similarity measure between
    inputs $x$ and $y$, satisfying positive semi-definiteness properties.
    This interface supports both explicit kernel evaluation and
    feature map representations.
    """

    @abstractmethod
    def compute(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Compute kernel values between x and y.

        Parameters
        ----------
        x : Tensor
            First input tensor of shape (..., n, d).
        y : Tensor
            Second input tensor of shape (..., m, d).

        Returns
        -------
        Tensor
            Kernel matrix of shape (..., n, m) where element $(i,j)$
            contains $k(x_i, y_j)$.
        """
        pass


    def gram_matrix(self, x: Tensor) -> Tensor:
        r"""Compute Gram matrix $K_{ij} = k(x_i, x_j)$.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., n, d).

        Returns
        -------
        Tensor
            Gram matrix of shape (..., n, n).
        """
        return self.compute(x, x)

    def is_positive_definite(self, x: Tensor, eps: float = 1e-6) -> bool:
        """Check if the kernel yields a positive definite Gram matrix.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., n, d).
        eps : float, default=1e-6
            Tolerance for eigenvalue positivity check.

        Returns
        -------
        bool
            True if all eigenvalues of Gram matrix are > eps.
        """
        gram = self.gram_matrix(x)
        eigenvalues = torch.linalg.eigvalsh(gram)
        return bool(torch.all(eigenvalues > eps).item())


class RandomFeatureMap(nn.Module, ABC):
    r"""Abstract base class for random feature map approximations.

    Random feature maps provide finite-dimensional approximations
    to kernel functions through the mapping:

    .. math::
        k(x, y) \approx \varphi(x)^T \varphi(y)

    This enables linear-time computation of kernel operations.

    Parameters
    ----------
    input_dim : int
        Dimension of input vectors.
    num_features : int
        Number of random features (D).
    kernel_scale : float, default=1.0
        Scaling parameter for the kernel.
    seed : int | None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    input_dim : int
        Input dimension.
    num_features : int
        Number of random features.
    kernel_scale : float
        Kernel scaling parameter.
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        kernel_scale: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.kernel_scale = kernel_scale

        if seed is not None:
            torch.manual_seed(seed)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply feature map to input.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., n, d).

        Returns
        -------
        Tensor
            Feature mapped tensor of shape (..., n, D) where D
            is the number of random features.
        """
        pass

    def kernel_approximation(self, x: Tensor, y: Tensor) -> Tensor:
        """Approximate kernel matrix using feature maps.

        Parameters
        ----------
        x : Tensor
            First input of shape (..., n, d).
        y : Tensor
            Second input of shape (..., m, d).

        Returns
        -------
        Tensor
            Approximated kernel matrix of shape (..., n, m).
        """
        phi_x = self.forward(x)  # (..., n, D)
        phi_y = self.forward(y)  # (..., m, D)
        return torch.matmul(phi_x, phi_y.transpose(-2, -1))



class ShiftInvariantKernel(KernelFunction):
    r"""Base class for shift-invariant (stationary) kernels.

    Shift-invariant kernels depend only on the difference $x - y$,
    i.e., $k(x, y) = k(x - y, 0) = \kappa(x - y)$ for some function $\kappa$.

    These kernels admit Random Fourier Features approximation
    via Bochner's theorem.

    Parameters
    ----------
    bandwidth : float, default=1.0
        Kernel bandwidth parameter (inverse of length scale).

    Attributes
    ----------
    bandwidth : float
        The bandwidth parameter.
    """

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth

    @abstractmethod
    def evaluate_difference(self, diff: Tensor) -> Tensor:
        r"""Evaluate kernel on difference vectors.

        Parameters
        ----------
        diff : Tensor
            Difference vectors $x - y$ of shape (..., d).

        Returns
        -------
        Tensor
            Kernel values $\kappa(\text{diff})$ of shape (...).
        """
        pass

    def compute(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute kernel matrix for shift-invariant kernel.

        Parameters
        ----------
        x : Tensor
            First input of shape (..., n, d).
        y : Tensor
            Second input of shape (..., m, d).

        Returns
        -------
        Tensor
            Kernel matrix of shape (..., n, m).
        """
        # Compute pairwise differences
        x_expanded = x.unsqueeze(-2)  # (..., n, 1, d)
        y_expanded = y.unsqueeze(-3)  # (..., 1, m, d)
        diff = x_expanded - y_expanded  # (..., n, m, d)

        # Evaluate kernel on differences
        return self.evaluate_difference(diff)

    @abstractmethod
    def spectral_density(self, omega: Tensor) -> Tensor:
        """Fourier transform of the kernel (spectral density).

        For shift-invariant kernels, this defines the sampling
        distribution for Random Fourier Features.

        Parameters
        ----------
        omega : Tensor
            Frequency vectors of shape (..., d).

        Returns
        -------
        Tensor
            Spectral density values of shape (...).
        """
        pass


class PolynomialKernel(KernelFunction):
    r"""Polynomial kernel $k(x, y) = (\alpha \langle x, y \rangle + c)^d$.

    Parameters
    ----------
    degree : int, default=2
        Polynomial degree.
    alpha : float, default=1.0
        Scaling of inner product.
    coef0 : float, default=0.0
        Constant term.

    Attributes
    ----------
    degree : int
        The polynomial degree.
    alpha : float
        Inner product scaling.
    coef0 : float
        Constant coefficient.
    """

    def __init__(
        self,
        degree: int = 2,
        alpha: float = 1.0,
        coef0: float = 0.0,
    ):
        self.degree = degree
        self.alpha = alpha
        self.coef0 = coef0

    def compute(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute polynomial kernel matrix.

        Parameters
        ----------
        x : Tensor
            First input of shape (..., n, d).
        y : Tensor
            Second input of shape (..., m, d).

        Returns
        -------
        Tensor
            Kernel matrix of shape (..., n, m).
        """
        inner_product = torch.matmul(x, y.transpose(-2, -1))
        return (self.alpha * inner_product + self.coef0) ** self.degree


class CosineKernel(KernelFunction):
    r"""Cosine similarity kernel $k(x, y) = \frac{\langle x, y \rangle}{\|x\| \|y\|}$.

    Parameters
    ----------
    eps : float, default=1e-8
        Small value for numerical stability.

    Attributes
    ----------
    eps : float
        Numerical stability parameter.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def compute(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute cosine similarity kernel matrix.

        Parameters
        ----------
        x : Tensor
            First input of shape (..., n, d).
        y : Tensor
            Second input of shape (..., m, d).

        Returns
        -------
        Tensor
            Kernel matrix of shape (..., n, m).
        """
        x_norm = torch.norm(x, dim=-1, keepdim=True)  # (..., n, 1)
        y_norm = torch.norm(y, dim=-1, keepdim=True)  # (..., m, 1)

        x_normalized = x / (x_norm + self.eps)
        y_normalized = y / (y_norm + self.eps)

        return torch.matmul(x_normalized, y_normalized.transpose(-2, -1))


# Kernel type literal for configuration
KernelType = Literal[
    "gaussian",
    "laplacian",
    "polynomial",
    "cosine",
    "linear",
]


__all__ = [
    "CosineKernel",
    "KernelFunction",
    "KernelType",
    "PolynomialKernel",
    "RandomFeatureMap",
    "ShiftInvariantKernel",
]
