r"""Random Fourier Features (RFF) for kernel approximation.

This module implements Random Fourier Features, a technique for approximating
shift-invariant kernels through explicit feature maps. RFF enables linear-time
computation of kernel operations that would normally require quadratic time,
making them suitable for large-scale machine learning applications.

The implementation supports various kernel types including Gaussian (RBF),
Laplacian, and other shift-invariant kernels. It also includes orthogonal
random features for improved approximation quality.

Classes
-------
GaussianRFFKernel
    Gaussian/RBF kernel with RFF approximation.
LaplacianRFFKernel
    Laplacian kernel with RFF approximation.
OrthogonalRandomFeatures
    Orthogonal variant of random features for better approximation.
RFFAttentionKernel
    Specialized RFF for attention mechanisms.

Examples
--------
Basic Gaussian RFF usage:

>>> import torch
>>> from spectrans.kernels.rff import GaussianRFFKernel
>>> kernel = GaussianRFFKernel(input_dim=64, num_features=256, sigma=1.0)
>>> x = torch.randn(32, 100, 64)  # (batch, sequence, dim)
>>> features = kernel.feature_map(x)
>>> assert features.shape == (32, 100, 256)

Computing approximate kernel matrix:

>>> y = torch.randn(32, 50, 64)
>>> K_approx = kernel.kernel_approximation(x, y)
>>> assert K_approx.shape == (32, 100, 50)

Using orthogonal features for better approximation:

>>> from spectrans.kernels.rff import OrthogonalRandomFeatures
>>> orf = OrthogonalRandomFeatures(input_dim=64, num_features=256)
>>> features = orf(x)

Notes
-----
Random Fourier Features Theory:

For a shift-invariant kernel :math:`k(x, y) = \kappa(x - y)` with Fourier transform :math:`p(\omega)`,
Bochner's theorem gives:

.. math::
    k(x, y) = \int p(\omega) \exp(i\omega^T(x-y)) d\omega

The RFF approximation samples :math:`\omega \sim p(\omega)` and uses:

.. math::
    \varphi(x) = \sqrt{\frac{2}{D}} \left[\cos(\omega_1^Tx + b_1), \ldots, \cos(\omega_D^Tx + b_D)\right]

This gives :math:`k(x, y) \approx \varphi(x)^T \varphi(y)` with approximation error :math:`O(1/\sqrt{D})`.

For Gaussian kernel: :math:`p(\omega) = \mathcal{N}(0, \sigma^2 I)`

For Laplacian kernel: :math:`p(\omega) = \text{Cauchy}(0, \sigma)`

References
----------
.. [1] Rahimi, A. & Recht, B., "Random Features for Large-Scale Kernel
       Machines", NeurIPS 2007.
.. [2] Yu, F. et al., "Orthogonal Random Features", NeurIPS 2016.
.. [3] Choromanski, K. et al., "Rethinking Attention with Performers",
       ICLR 2021.

See Also
--------
spectrans.kernels.base : Base kernel interfaces.
spectrans.layers.attention.spectral : Spectral attention using RFF.
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.registry import register_component
from ..core.types import Tensor
from .base import RandomFeatureMap, ShiftInvariantKernel


@register_component("kernel", "gaussian_rff")
class GaussianRFFKernel(ShiftInvariantKernel, RandomFeatureMap):
    r"""Gaussian (RBF) kernel with Random Fourier Features approximation.

    Implements :math:`k(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)` using RFF.

    Parameters
    ----------
    input_dim : int
        Dimension of input vectors.
    num_features : int
        Number of random Fourier features.
    sigma : float, default=1.0
        Kernel bandwidth (standard deviation).
    use_cos_sin : bool, default=False
        If True, use both cos and sin features (doubles feature dimension).
    orthogonal : bool, default=False
        If True, use orthogonal random features.
    trainable : bool, default=False
        If True, make random parameters trainable.
    seed : int | None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    omega : nn.Parameter or Tensor
        Random frequencies of shape (input_dim, num_features).
    bias : nn.Parameter or Tensor
        Random phase shifts of shape (num_features,).
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        sigma: float = 1.0,
        use_cos_sin: bool = False,
        orthogonal: bool = False,
        trainable: bool = False,
        seed: int | None = None,
    ):
        ShiftInvariantKernel.__init__(self, bandwidth=1.0 / sigma)
        RandomFeatureMap.__init__(self, input_dim, num_features, kernel_scale=sigma, seed=seed)

        self.sigma = sigma
        self.use_cos_sin = use_cos_sin
        self.orthogonal = orthogonal
        self.trainable = trainable

        # Effective number of output features
        self.output_features = num_features * 2 if use_cos_sin else num_features

        # Initialize random parameters
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize random frequencies and biases."""
        if self.orthogonal:
            # Orthogonal random features
            omega = self._sample_orthogonal_gaussian(
                self.input_dim, self.num_features
            ) / self.sigma
        else:
            # Standard Gaussian random features
            omega = torch.randn(self.input_dim, self.num_features) / self.sigma

        # Random phase shifts
        bias = torch.rand(self.num_features) * 2 * math.pi

        if self.trainable:
            self.omega = nn.Parameter(omega)
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer('omega', omega)
            self.register_buffer('bias', bias)

    def _sample_orthogonal_gaussian(self, n: int, m: int) -> Tensor:
        """Sample from orthogonal Gaussian distribution.

        Uses QR decomposition to generate orthogonal random features.

        Parameters
        ----------
        n : int
            Number of rows (input dimension).
        m : int
            Number of columns (features).

        Returns
        -------
        Tensor
            Orthogonal random matrix of shape (n, m).
        """
        # Handle case where m > n
        if m > n:
            # Sample multiple blocks and concatenate
            num_blocks = (m + n - 1) // n
            blocks = []
            for _ in range(num_blocks):
                G = torch.randn(n, n)
                Q, _ = torch.linalg.qr(G)
                blocks.append(Q)
            W = torch.cat(blocks, dim=1)[:, :m]
        else:
            G = torch.randn(n, m)
            Q, _ = torch.linalg.qr(G)
            W = Q

        # Scale to match Gaussian distribution
        W = W * math.sqrt(n)
        return W

    def forward(self, x: Tensor) -> Tensor:
        """Apply random Fourier feature map.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., n, d).

        Returns
        -------
        Tensor
            Feature mapped tensor of shape (..., n, D) where D is
            self.output_features.
        """
        # Linear projection: (..., n, d) @ (d, m) -> (..., n, m)
        projection = torch.matmul(x, self.omega)

        # Add phase shifts
        projection = projection + self.bias

        if self.use_cos_sin:
            # Use both cos and sin features
            cos_features = torch.cos(projection)
            sin_features = torch.sin(projection)
            features = torch.cat([cos_features, sin_features], dim=-1)
            # Normalization factor for cos+sin
            scale = math.sqrt(1.0 / self.num_features)
        else:
            # Use only cos features
            features = torch.cos(projection)
            # Normalization factor for cos only
            scale = math.sqrt(2.0 / self.num_features)

        return features * scale

    def evaluate_difference(self, diff: Tensor) -> Tensor:
        """Evaluate Gaussian kernel on difference vectors.

        Parameters
        ----------
        diff : Tensor
            Difference vectors of shape (..., d).

        Returns
        -------
        Tensor
            Kernel values of shape (...).
        """
        squared_norm = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-squared_norm / (2 * self.sigma ** 2))

    def spectral_density(self, omega: Tensor) -> Tensor:
        """Spectral density for Gaussian kernel (Gaussian distribution).

        Parameters
        ----------
        omega : Tensor
            Frequency vectors of shape (..., d).

        Returns
        -------
        Tensor
            Spectral density values of shape (...).
        """
        d = omega.shape[-1]
        norm_squared = torch.sum(omega ** 2, dim=-1)
        # Gaussian spectral density
        result: Tensor = ((2 * math.pi * self.sigma ** 2) ** (d / 2) *
                          torch.exp(-0.5 * self.sigma ** 2 * norm_squared))
        return result


@register_component("kernel", "laplacian_rff")
class LaplacianRFFKernel(ShiftInvariantKernel, RandomFeatureMap):
    r"""Laplacian kernel with Random Fourier Features approximation.

    Implements :math:`k(x, y) = \exp\left(-\frac{\|x - y\|_1}{\sigma}\right)` using RFF with Cauchy
    distribution for sampling frequencies.

    Parameters
    ----------
    input_dim : int
        Dimension of input vectors.
    num_features : int
        Number of random Fourier features.
    sigma : float, default=1.0
        Kernel bandwidth parameter.
    use_cos_sin : bool, default=False
        If True, use both cos and sin features.
    trainable : bool, default=False
        If True, make random parameters trainable.
    seed : int | None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        sigma: float = 1.0,
        use_cos_sin: bool = False,
        trainable: bool = False,
        seed: int | None = None,
    ):
        ShiftInvariantKernel.__init__(self, bandwidth=1.0 / sigma)
        RandomFeatureMap.__init__(self, input_dim, num_features, kernel_scale=sigma, seed=seed)

        self.sigma = sigma
        self.use_cos_sin = use_cos_sin
        self.trainable = trainable

        self.output_features = num_features * 2 if use_cos_sin else num_features

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize random frequencies from Cauchy distribution."""
        # Sample from Cauchy distribution for Laplacian kernel
        # Cauchy(0, 1/sigma) using inverse transform sampling
        uniform = torch.rand(self.input_dim, self.num_features)
        omega = torch.tan(math.pi * (uniform - 0.5)) / self.sigma

        bias = torch.rand(self.num_features) * 2 * math.pi

        if self.trainable:
            self.omega = nn.Parameter(omega)
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer('omega', omega)
            self.register_buffer('bias', bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply random Fourier feature map.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., n, d).

        Returns
        -------
        Tensor
            Feature mapped tensor of shape (..., n, D).
        """
        projection = torch.matmul(x, self.omega) + self.bias

        if self.use_cos_sin:
            cos_features = torch.cos(projection)
            sin_features = torch.sin(projection)
            features = torch.cat([cos_features, sin_features], dim=-1)
            scale = math.sqrt(1.0 / self.num_features)
        else:
            features = torch.cos(projection)
            scale = math.sqrt(2.0 / self.num_features)

        return features * scale

    def evaluate_difference(self, diff: Tensor) -> Tensor:
        """Evaluate Laplacian kernel on difference vectors.

        Parameters
        ----------
        diff : Tensor
            Difference vectors of shape (..., d).

        Returns
        -------
        Tensor
            Kernel values of shape (...).
        """
        l1_norm = torch.sum(torch.abs(diff), dim=-1)
        return torch.exp(-l1_norm / self.sigma)

    def spectral_density(self, omega: Tensor) -> Tensor:
        """Spectral density for Laplacian kernel (Cauchy distribution).

        Parameters
        ----------
        omega : Tensor
            Frequency vectors of shape (..., d).

        Returns
        -------
        Tensor
            Spectral density values of shape (...).
        """
        d = omega.shape[-1]
        # Product of 1D Cauchy densities
        density = torch.ones_like(omega[..., 0])
        for i in range(d):
            density = density * (2 * self.sigma /
                                (math.pi * (1 + (self.sigma * omega[..., i]) ** 2)))
        return density


@register_component("kernel", "orthogonal_rff")
class OrthogonalRandomFeatures(RandomFeatureMap):
    """Orthogonal Random Features for improved kernel approximation.

    Uses structured orthogonal matrices for better approximation quality
    compared to standard i.i.d. Gaussian features.

    Parameters
    ----------
    input_dim : int
        Dimension of input vectors.
    num_features : int
        Number of random features.
    kernel_type : Literal["gaussian", "laplacian"], default="gaussian"
        Type of kernel to approximate.
    sigma : float, default=1.0
        Kernel bandwidth parameter.
    use_hadamard : bool, default=False
        If True, use fast Hadamard transform for efficiency.
    trainable : bool, default=False
        If True, make scaling parameters trainable.
    seed : int | None, default=None
        Random seed.
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        kernel_type: Literal["gaussian", "laplacian"] = "gaussian",
        sigma: float = 1.0,
        use_hadamard: bool = False,
        trainable: bool = False,
        seed: int | None = None,
    ):
        super().__init__(input_dim, num_features, kernel_scale=sigma, seed=seed)

        self.kernel_type = kernel_type
        self.sigma = sigma
        self.use_hadamard = use_hadamard
        self.trainable = trainable

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize orthogonal random features."""
        if self.use_hadamard:
            # Use Hadamard matrix with random diagonal
            self._setup_hadamard_features()
        else:
            # Use QR decomposition for orthogonal features
            self._setup_qr_features()

        # Random bias
        bias = torch.rand(self.num_features) * 2 * math.pi
        if self.trainable:
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer('bias', bias)

    def _setup_qr_features(self) -> None:
        """Setup orthogonal features using QR decomposition."""
        # Generate blocks of orthogonal matrices
        num_blocks = (self.num_features + self.input_dim - 1) // self.input_dim

        blocks = []
        for _ in range(num_blocks):
            if self.kernel_type == "gaussian":
                G = torch.randn(self.input_dim, self.input_dim)
            else:  # laplacian
                uniform = torch.rand(self.input_dim, self.input_dim)
                G = torch.tan(math.pi * (uniform - 0.5))

            Q, _ = torch.linalg.qr(G)
            blocks.append(Q)

        W = torch.cat(blocks, dim=1)[:, :self.num_features]

        # Scale based on kernel type
        if self.kernel_type == "gaussian":
            W = W * math.sqrt(self.input_dim) / self.sigma
        else:
            W = W / self.sigma

        if self.trainable:
            self.projection = nn.Parameter(W)
        else:
            self.register_buffer('projection', W)

    def _setup_hadamard_features(self) -> None:
        """Setup features using fast Hadamard transform."""
        # Find next power of 2
        d_padded = 2 ** math.ceil(math.log2(max(self.input_dim, self.num_features)))

        # Random diagonal matrices for HD HD HD structure
        num_blocks = 3
        diagonals = []

        for _ in range(num_blocks):
            if self.kernel_type == "gaussian":
                diag = torch.randn(d_padded) / self.sigma
            else:
                uniform = torch.rand(d_padded)
                diag = torch.tan(math.pi * (uniform - 0.5)) / self.sigma

            diag = diag / diag.norm() * math.sqrt(d_padded)
            diagonals.append(diag)

        if self.trainable:
            self.diagonals = nn.ParameterList([nn.Parameter(d) for d in diagonals])
        else:
            for i, diag in enumerate(diagonals):
                self.register_buffer(f'diagonal_{i}', diag)

        self.d_padded = d_padded

    def _hadamard_transform(self, x: Tensor) -> Tensor:
        """Apply fast Hadamard transform."""
        # This is a placeholder for actual Hadamard transform
        # In practice, would use fast Walsh-Hadamard transform
        n = x.shape[-1]
        h = torch.ones(n, n, device=x.device, dtype=x.dtype) / math.sqrt(n)
        # Simplified - actual implementation would use butterfly operations
        return torch.matmul(x, h)

    def forward(self, x: Tensor) -> Tensor:
        """Apply orthogonal random feature map.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., n, d).

        Returns
        -------
        Tensor
            Feature mapped tensor of shape (..., n, D).
        """
        if self.use_hadamard:
            # Pad input if necessary
            if x.shape[-1] < self.d_padded:
                padding = self.d_padded - x.shape[-1]
                x = F.pad(x, (0, padding))

            # Apply HD HD HD structure
            z = x
            for i in range(3):
                if hasattr(self, 'diagonals'):
                    diag = self.diagonals[i]
                else:
                    diag = getattr(self, f'diagonal_{i}')
                z = z * diag
                z = self._hadamard_transform(z)

            # Truncate to desired number of features
            projection = z[..., :self.num_features]
        else:
            projection = torch.matmul(x, self.projection)

        # Add bias and apply cosine
        projection = projection + self.bias
        features = torch.cos(projection)

        # Normalize
        scale = math.sqrt(2.0 / self.num_features)
        return features * scale


@register_component("kernel", "rff_attention")
class RFFAttentionKernel(RandomFeatureMap):
    """Random Fourier Features specifically designed for attention mechanisms.

    Implements positive random features for use in linear attention,
    following the Performer architecture.

    Parameters
    ----------
    input_dim : int
        Dimension of input vectors (typically head_dim).
    num_features : int
        Number of random features.
    kernel_type : Literal["softmax", "relu", "elu"], default="softmax"
        Type of kernel approximation.
    use_orthogonal : bool, default=True
        If True, use orthogonal random features.
    redraw : bool, default=False
        If True, redraw random features at each forward pass.
    seed : int | None, default=None
        Random seed.
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        kernel_type: Literal["softmax", "relu", "elu"] = "softmax",
        use_orthogonal: bool = True,
        redraw: bool = False,
        seed: int | None = None,
    ):
        super().__init__(input_dim, num_features, seed=seed)

        self.kernel_type = kernel_type
        self.use_orthogonal = use_orthogonal
        self.redraw = redraw

        if not redraw:
            self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize random projection matrix."""
        if self.use_orthogonal:
            # Orthogonal Gaussian features
            projection = self._sample_orthogonal_gaussian()
        else:
            # Standard Gaussian features
            projection = torch.randn(self.input_dim, self.num_features)

        projection = projection / math.sqrt(self.input_dim)
        self.register_buffer('projection', projection)
        self.projection: Tensor  # Type hint for mypy

    def _sample_orthogonal_gaussian(self) -> Tensor:
        """Sample orthogonal Gaussian matrix."""
        num_blocks = (self.num_features + self.input_dim - 1) // self.input_dim
        blocks = []

        for _ in range(num_blocks):
            G = torch.randn(self.input_dim, self.input_dim)
            Q, _ = torch.linalg.qr(G)
            blocks.append(Q)

        W = torch.cat(blocks, dim=1)[:, :self.num_features]
        return W * math.sqrt(self.input_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply random feature map for attention.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., n, d).

        Returns
        -------
        Tensor
            Positive feature mapped tensor of shape (..., n, D).
        """
        if self.redraw:
            # Redraw random features (useful for training)
            projection = self._sample_orthogonal_gaussian() if self.use_orthogonal \
                        else torch.randn(self.input_dim, self.num_features, device=x.device)
            projection = projection / math.sqrt(self.input_dim)
        else:
            projection = self.projection

        # Linear projection
        z = torch.matmul(x, projection)

        if self.kernel_type == "softmax":
            # Positive features for softmax kernel approximation
            # :math:`\varphi(x) = \exp(x^T \omega - \|x\|^2/2) / \sqrt{m}`
            x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True) / 2
            features = torch.exp(z - x_norm_sq)
            scale = 1.0 / math.sqrt(self.num_features)

        elif self.kernel_type == "relu":
            # ReLU kernel: :math:`\max(0, x^T \omega)`
            features = F.relu(z)
            scale = math.sqrt(2.0 / self.num_features)

        else:  # elu
            # ELU kernel for smooth approximation
            features = F.elu(z) + 1
            scale = 1.0 / math.sqrt(self.num_features)

        return features * scale


__all__ = [
    "GaussianRFFKernel",
    "LaplacianRFFKernel",
    "OrthogonalRandomFeatures",
    "RFFAttentionKernel",
]
