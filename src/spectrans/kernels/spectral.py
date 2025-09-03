"""Spectral kernel functions for attention mechanisms.

This module implements kernel functions based on spectral decomposition
and eigenfunction expansions. These kernels are particularly useful for
spectral attention mechanisms and provide alternatives to RFF-based
approximations.

The implementations include polynomial spectral kernels, eigenvalue-based
decompositions, and learnable spectral filters that can be optimized
during training.

Classes
-------
SpectralKernel
    Base class for spectral kernel functions.
PolynomialSpectralKernel
    Polynomial kernel with spectral decomposition.
TruncatedSVDKernel
    Kernel approximation via truncated SVD.
LearnableSpectralKernel
    Spectral kernel with learnable eigenvalues.
FourierKernel
    Kernel defined in Fourier domain.

Examples
--------
Using polynomial spectral kernel:

>>> import torch
>>> from spectrans.kernels.spectral import PolynomialSpectralKernel
>>> kernel = PolynomialSpectralKernel(rank=32, degree=3)
>>> Q, K = torch.randn(2, 100, 64), torch.randn(2, 100, 64)
>>> attention_weights = kernel.compute_attention(Q, K)
>>> assert attention_weights.shape == (2, 100, 100)

Learnable spectral kernel:

>>> from spectrans.kernels.spectral import LearnableSpectralKernel
>>> kernel = LearnableSpectralKernel(input_dim=64, rank=16)
>>> features = kernel.extract_features(Q)
>>> assert features.shape == (2, 100, 16)

Notes
-----
Spectral kernels leverage eigendecomposition for efficient computation:
    K(X, Y) = Φ(X) Λ Φ(Y)^T

Where Φ are eigenfunctions and Λ are eigenvalues. This allows:
1. Low-rank approximations via truncation
2. Learnable spectral filters via trainable Λ
3. Efficient computation in eigenspace

The rank parameter controls the tradeoff between approximation
quality and computational efficiency.

References
----------
.. [1] Chen, Y. et al., "Scatterbrain: Unifying Sparse and Low-rank 
       Attention Approximation", NeurIPS 2021.
.. [2] Wang, S. et al., "Linformer: Self-Attention with Linear 
       Complexity", 2020.

See Also
--------
spectrans.kernels.base : Base kernel interfaces.
spectrans.kernels.rff : Random Fourier Features.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.registry import register_component
from ..core.types import ComplexityInfo, Tensor
from .base import KernelFunction


class SpectralKernel(KernelFunction):
    """Base class for spectral kernel functions.
    
    Spectral kernels use eigendecomposition or spectral analysis
    for efficient kernel computation.
    
    Parameters
    ----------
    rank : int
        Rank of spectral approximation.
    normalize : bool, default=True
        Whether to normalize kernel values.
        
    Attributes
    ----------
    rank : int
        Approximation rank.
    normalize : bool
        Normalization flag.
    """

    def __init__(self, rank: int, normalize: bool = True):
        self.rank = rank
        self.normalize = normalize

    def spectral_decomposition(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute spectral decomposition of input.
        
        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., n, d).
            
        Returns
        -------
        eigenvectors : Tensor
            Eigenvectors of shape (..., n, rank).
        eigenvalues : Tensor
            Eigenvalues of shape (..., rank).
        """
        # Compute Gram matrix
        gram = torch.matmul(x, x.transpose(-2, -1))

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)

        # Keep top-k eigenvalues/vectors
        eigenvalues = eigenvalues[..., -self.rank:]
        eigenvectors = eigenvectors[..., -self.rank:]

        if self.normalize:
            # Normalize by trace
            trace = eigenvalues.sum(dim=-1, keepdim=True)
            eigenvalues = eigenvalues / (trace + 1e-8)

        return eigenvectors, eigenvalues

    @property
    def complexity(self) -> ComplexityInfo:
        """Computational complexity."""
        return {
            'time': f'O(n²r) for rank r={self.rank}',
            'space': f'O(nr) for rank r={self.rank}'
        }


@register_component("kernel", "polynomial_spectral")
class PolynomialSpectralKernel(SpectralKernel):
    """Polynomial kernel with spectral decomposition.
    
    Computes (X Y^T + c)^d using eigendecomposition for efficiency.
    
    Parameters
    ----------
    rank : int
        Rank of spectral approximation.
    degree : int, default=2
        Polynomial degree.
    coef0 : float, default=1.0
        Constant coefficient.
    alpha : float, default=1.0
        Scaling factor.
    normalize : bool, default=True
        Whether to normalize.
        
    Attributes
    ----------
    degree : int
        Polynomial degree.
    coef0 : float
        Constant term.
    alpha : float
        Scale factor.
    """

    def __init__(
        self,
        rank: int,
        degree: int = 2,
        coef0: float = 1.0,
        alpha: float = 1.0,
        normalize: bool = True,
    ):
        super().__init__(rank, normalize)
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha

    def compute(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute polynomial spectral kernel.
        
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
        # Standard polynomial kernel
        inner = torch.matmul(x, y.transpose(-2, -1))
        kernel = (self.alpha * inner + self.coef0) ** self.degree

        if self.normalize:
            # Normalize by geometric mean of norms
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            y_norm = torch.norm(y, dim=-1, keepdim=True)
            norm_matrix = torch.matmul(x_norm, y_norm.transpose(-2, -1))
            kernel = kernel / (norm_matrix + 1e-8)

        return kernel

    def compute_attention(self, q: Tensor, k: Tensor) -> Tensor:
        """Compute attention weights using spectral decomposition.
        
        Parameters
        ----------
        q : Tensor
            Queries of shape (..., n, d).
        k : Tensor
            Keys of shape (..., m, d).
            
        Returns
        -------
        Tensor
            Attention weights of shape (..., n, m).
        """
        # Low-rank approximation via SVD
        # Q = U_q S_q V_q^T, K = U_k S_k V_k^T

        # Compute QK^T approximately
        q_reduced = self._reduce_rank(q)  # (..., n, r)
        k_reduced = self._reduce_rank(k)  # (..., m, r)

        # Polynomial kernel in reduced space
        inner = torch.matmul(q_reduced, k_reduced.transpose(-2, -1))
        attention = (self.alpha * inner + self.coef0) ** self.degree

        if self.normalize:
            attention = F.softmax(attention, dim=-1)

        return attention

    def _reduce_rank(self, x: Tensor) -> Tensor:
        """Reduce dimensionality via PCA/SVD.
        
        Parameters
        ----------
        x : Tensor
            Input of shape (..., n, d).
            
        Returns  
        -------
        Tensor
            Reduced tensor of shape (..., n, r).
        """
        # Center the data
        mean = x.mean(dim=-2, keepdim=True)
        x_centered = x - mean

        # Compute covariance matrix
        cov = torch.matmul(x_centered.transpose(-2, -1), x_centered)
        cov = cov / x.shape[-2]

        # Eigendecomposition of covariance
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Keep top-r components
        top_components = eigenvectors[..., -self.rank:]

        # Project data
        x_reduced = torch.matmul(x_centered, top_components)

        return x_reduced


@register_component("kernel", "truncated_svd")
class TruncatedSVDKernel(SpectralKernel):
    """Kernel approximation via truncated SVD.
    
    Uses SVD to compute low-rank approximation of kernel matrix.
    
    Parameters
    ----------
    rank : int
        Truncation rank.
    normalize : bool, default=True
        Whether to normalize.
    use_randomized : bool, default=False
        Use randomized SVD for efficiency.
        
    Attributes
    ----------
    use_randomized : bool
        Whether to use randomized algorithms.
    """

    def __init__(
        self,
        rank: int,
        normalize: bool = True,
        use_randomized: bool = False,
    ):
        super().__init__(rank, normalize)
        self.use_randomized = use_randomized

    def compute(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute kernel via truncated SVD.
        
        Parameters
        ----------
        x : Tensor
            First input of shape (..., n, d).
        y : Tensor
            Second input of shape (..., m, d).
            
        Returns
        -------
        Tensor
            Approximate kernel matrix of shape (..., n, m).
        """
        # Compute full kernel matrix
        kernel_full = torch.matmul(x, y.transpose(-2, -1))

        if self.use_randomized:
            # Randomized SVD (faster for large matrices)
            kernel_approx = self._randomized_svd_approximation(kernel_full)
        else:
            # Standard SVD
            U, S, Vt = torch.linalg.svd(kernel_full, full_matrices=False)

            # Truncate to rank
            U_r = U[..., :self.rank]
            S_r = S[..., :self.rank]
            Vt_r = Vt[..., :self.rank, :]

            # Reconstruct
            kernel_approx = torch.matmul(U_r * S_r.unsqueeze(-2), Vt_r)

        if self.normalize:
            # Normalize rows
            row_norms = kernel_approx.norm(dim=-1, keepdim=True)
            kernel_approx = kernel_approx / (row_norms + 1e-8)

        return kernel_approx

    def _randomized_svd_approximation(self, matrix: Tensor) -> Tensor:
        """Randomized SVD for fast approximation.
        
        Parameters
        ----------
        matrix : Tensor
            Matrix to approximate of shape (..., n, m).
            
        Returns
        -------
        Tensor
            Low-rank approximation of shape (..., n, m).
        """
        n, m = matrix.shape[-2:]

        # Random sampling matrix
        omega = torch.randn(m, self.rank + 10, device=matrix.device)

        # Range finding: Y = A @ Omega
        Y = torch.matmul(matrix, omega)

        # Orthogonalize
        Q, _ = torch.linalg.qr(Y)
        Q = Q[..., :self.rank]

        # Project and compute SVD of smaller matrix
        B = torch.matmul(Q.transpose(-2, -1), matrix)
        U_b, S_b, Vt_b = torch.linalg.svd(B, full_matrices=False)

        # Recover approximate SVD of original matrix
        U = torch.matmul(Q, U_b)

        # Reconstruct
        approx = torch.matmul(U * S_b.unsqueeze(-2), Vt_b)

        return approx


@register_component("kernel", "learnable_spectral")
class LearnableSpectralKernel(nn.Module, SpectralKernel):
    """Spectral kernel with learnable eigenvalues and eigenfunctions.
    
    Parameters
    ----------
    input_dim : int
        Input dimension.
    rank : int
        Number of spectral components.
    init_scale : float, default=1.0
        Initialization scale.
    trainable_eigenvectors : bool, default=True
        Whether eigenvectors are trainable.
    normalize : bool, default=True
        Whether to normalize.
        
    Attributes
    ----------
    eigenvectors : nn.Parameter
        Learnable eigenvectors of shape (input_dim, rank).
    eigenvalues : nn.Parameter
        Learnable eigenvalues of shape (rank,).
    """

    def __init__(
        self,
        input_dim: int,
        rank: int,
        init_scale: float = 1.0,
        trainable_eigenvectors: bool = True,
        normalize: bool = True,
    ):
        nn.Module.__init__(self)
        SpectralKernel.__init__(self, rank, normalize)

        self.input_dim = input_dim
        self.trainable_eigenvectors = trainable_eigenvectors

        # Initialize eigenvectors (orthogonal)
        eigenvectors = torch.randn(input_dim, rank) * init_scale
        eigenvectors, _ = torch.linalg.qr(eigenvectors)

        if trainable_eigenvectors:
            self.eigenvectors = nn.Parameter(eigenvectors)
        else:
            self.register_buffer('eigenvectors', eigenvectors)

        # Initialize eigenvalues (positive, decreasing)
        eigenvalues = torch.linspace(1.0, 0.1, rank) * init_scale
        self.eigenvalues = nn.Parameter(eigenvalues)

    def compute(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute learnable spectral kernel.
        
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
        # Project to eigenspace
        x_proj = torch.matmul(x, self.eigenvectors)  # (..., n, r)
        y_proj = torch.matmul(y, self.eigenvectors)  # (..., m, r)

        # Apply eigenvalue weighting
        x_weighted = x_proj * torch.sqrt(torch.abs(self.eigenvalues) + 1e-8)
        y_weighted = y_proj * torch.sqrt(torch.abs(self.eigenvalues) + 1e-8)

        # Compute kernel
        kernel = torch.matmul(x_weighted, y_weighted.transpose(-2, -1))

        if self.normalize:
            # Row normalization
            kernel = F.normalize(kernel, p=2, dim=-1)

        return kernel

    def extract_features(self, x: Tensor) -> Tensor:
        """Extract spectral features.
        
        Parameters
        ----------
        x : Tensor
            Input of shape (..., n, d).
            
        Returns
        -------
        Tensor
            Spectral features of shape (..., n, r).
        """
        # Project to eigenspace
        features = torch.matmul(x, self.eigenvectors)

        # Weight by eigenvalues
        features = features * torch.sqrt(torch.abs(self.eigenvalues) + 1e-8)

        return features

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Forward pass for nn.Module compatibility.
        
        Parameters
        ----------
        x : Tensor
            First input of shape (..., n, d).
        y : Tensor | None, default=None
            Second input. If None, returns features.
            
        Returns
        -------
        Tensor
            Kernel matrix or features.
        """
        if y is None:
            return self.extract_features(x)
        else:
            return self.compute(x, y)

    def orthogonalize_eigenvectors(self) -> None:
        """Orthogonalize eigenvectors via Gram-Schmidt."""
        if self.trainable_eigenvectors:
            with torch.no_grad():
                Q, _ = torch.linalg.qr(self.eigenvectors)
                self.eigenvectors.data = Q


@register_component("kernel", "fourier_kernel")
class FourierKernel(SpectralKernel):
    """Kernel defined in Fourier domain.
    
    Defines kernel through spectral filters in frequency space.
    
    Parameters
    ----------
    rank : int
        Number of Fourier modes.
    input_dim : int
        Input dimension.
    learnable_filter : bool, default=True
        Whether filter is learnable.
    filter_type : Literal["gaussian", "butterworth", "ideal"], default="gaussian"
        Type of spectral filter.
    cutoff_freq : float, default=0.5
        Normalized cutoff frequency.
        
    Attributes
    ----------
    filter : nn.Parameter or Tensor
        Spectral filter of shape (rank,).
    """

    def __init__(
        self,
        rank: int,
        input_dim: int,
        learnable_filter: bool = True,
        filter_type: Literal["gaussian", "butterworth", "ideal"] = "gaussian",
        cutoff_freq: float = 0.5,
    ):
        super().__init__(rank, normalize=True)

        self.input_dim = input_dim
        self.filter_type = filter_type
        self.cutoff_freq = cutoff_freq

        # Initialize spectral filter
        filter_vals = self._init_filter()

        if learnable_filter:
            self.filter = nn.Parameter(filter_vals)
        else:
            self.register_buffer('filter', filter_vals)

    def _init_filter(self) -> Tensor:
        """Initialize spectral filter."""
        freqs = torch.linspace(0, 1, self.rank)

        if self.filter_type == "gaussian":
            # Gaussian low-pass
            filter_vals = torch.exp(-(freqs / self.cutoff_freq) ** 2)

        elif self.filter_type == "butterworth":
            # Butterworth filter
            order = 4
            filter_vals = 1 / (1 + (freqs / self.cutoff_freq) ** (2 * order))

        else:  # ideal
            # Ideal low-pass
            filter_vals = (freqs <= self.cutoff_freq).float()

        return filter_vals

    def compute(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute Fourier kernel.
        
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
        # Compute FFT of inputs
        x_freq = torch.fft.rfft(x, dim=-1)
        y_freq = torch.fft.rfft(y, dim=-1)

        # Truncate to rank modes
        x_freq = x_freq[..., :self.rank]
        y_freq = y_freq[..., :self.rank]

        # Apply spectral filter
        x_filtered = x_freq * self.filter
        y_filtered = y_freq * self.filter

        # Compute kernel in frequency domain
        # K(x,y) = Real(IFFT(X_filtered * conj(Y_filtered)))
        kernel_freq = x_filtered.unsqueeze(-2) * y_filtered.unsqueeze(-3).conj()

        # Average over frequency dimension
        kernel = kernel_freq.real.mean(dim=-1)

        return kernel


__all__ = [
    "FourierKernel",
    "LearnableSpectralKernel",
    "PolynomialSpectralKernel",
    "SpectralKernel",
    "TruncatedSVDKernel",
]
