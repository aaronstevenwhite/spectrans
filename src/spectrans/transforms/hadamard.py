"""Hadamard transform implementations."""

import math

import torch

from ..core.registry import register_component
from ..core.types import Tensor
from .base import OrthogonalTransform, SpectralTransform2D


@register_component("transform", "hadamard")
class HadamardTransform(OrthogonalTransform):
    """Fast Walsh-Hadamard Transform.

    The Hadamard transform is an orthogonal transform using only +1 and -1
    values, making it computationally efficient. The transform size must
    be a power of 2.

    Parameters
    ----------
    normalized : bool, default=True
        Whether to normalize by 1/sqrt(n) for orthogonality.
    """

    def __init__(self, normalized: bool = True):
        self.normalized = normalized

    def transform(self, x: Tensor, dim: int = -1) -> Tensor:
        """Apply Fast Walsh-Hadamard Transform.

        Parameters
        ----------
        x : Tensor
            Input tensor. Size along dim must be power of 2.
        dim : int, default=-1
            Dimension along which to apply transform.

        Returns
        -------
        Tensor
            Hadamard transformed tensor.

        Raises
        ------
        ValueError
            If size along dim is not a power of 2.
        """
        n = x.shape[dim]

        # Check if n is power of 2
        if n & (n - 1) != 0:
            raise ValueError(f"Hadamard transform requires size to be power of 2, got {n}")

        # Move dimension to last for easier processing
        if dim != -1 and dim != x.ndim - 1:
            x = x.transpose(dim, -1)

        # Apply Fast Walsh-Hadamard Transform
        result = self._fwht(x)

        # Normalize if requested
        if self.normalized:
            result = result / math.sqrt(n)

        # Move dimension back
        if dim != -1 and dim != x.ndim - 1:
            result = result.transpose(dim, -1)

        return result

    def inverse_transform(self, x: Tensor, dim: int = -1) -> Tensor:
        """Apply inverse Hadamard transform.

        The Hadamard transform is self-inverse (up to normalization).

        Parameters
        ----------
        x : Tensor
            Hadamard coefficients.
        dim : int, default=-1
            Dimension along which to apply inverse transform.

        Returns
        -------
        Tensor
            Inverse transformed tensor.
        """
        n = x.shape[dim]

        # For orthogonal Hadamard, inverse is same as forward
        if self.normalized:
            return self.transform(x, dim)
        else:
            # Without normalization, need to scale by 1/n
            result = self.transform(x, dim)
            return result / n

    def _fwht(self, x: Tensor) -> Tensor:
        """Fast Walsh-Hadamard Transform (in-place style).

        Parameters
        ----------
        x : Tensor
            Input tensor with last dimension being power of 2.

        Returns
        -------
        Tensor
            Transformed tensor.
        """
        x = x.clone()  # Don't modify input
        n = x.shape[-1]

        # Iterative FWHT using butterfly operations
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    # Butterfly operation
                    x_j = x[..., j].clone()
                    x_jh = x[..., j + h].clone()
                    x[..., j] = x_j + x_jh
                    x[..., j + h] = x_j - x_jh
            h *= 2

        return x


@register_component("transform", "hadamard_2d")
class HadamardTransform2D(SpectralTransform2D):
    """2D Fast Walsh-Hadamard Transform.

    Applies Hadamard transform along two dimensions.

    Parameters
    ----------
    normalized : bool, default=True
        Whether to normalize for orthogonality.
    """

    def __init__(self, normalized: bool = True):
        self.normalized = normalized
        self.hadamard = HadamardTransform(normalized=False)  # Handle normalization here

    def transform(self, x: Tensor, dim: tuple[int, int] = (-2, -1)) -> Tensor:
        """Apply 2D Hadamard transform.

        Parameters
        ----------
        x : Tensor
            Input tensor. Sizes along both dims must be powers of 2.
        dim : tuple[int, int], default=(-2, -1)
            Dimensions along which to apply transform.

        Returns
        -------
        Tensor
            2D Hadamard transformed tensor.
        """
        # Apply along first dimension
        result = self.hadamard.transform(x, dim=dim[0])
        # Apply along second dimension
        result = self.hadamard.transform(result, dim=dim[1])

        if self.normalized:
            n1 = x.shape[dim[0]]
            n2 = x.shape[dim[1]]
            result = result / math.sqrt(n1 * n2)

        return result

    def inverse_transform(self, x: Tensor, dim: tuple[int, int] = (-2, -1)) -> Tensor:
        """Apply inverse 2D Hadamard transform.

        Parameters
        ----------
        x : Tensor
            Hadamard coefficients.
        dim : tuple[int, int], default=(-2, -1)
            Dimensions along which to apply inverse transform.

        Returns
        -------
        Tensor
            Inverse transformed tensor.
        """
        if self.normalized:
            return self.transform(x, dim)
        else:
            result = self.transform(x, dim)
            n1 = x.shape[dim[0]]
            n2 = x.shape[dim[1]]
            return result / (n1 * n2)


@register_component("transform", "sequency_hadamard")
class SequencyHadamardTransform(OrthogonalTransform):
    """Sequency-ordered Hadamard Transform.

    The sequency ordering arranges basis functions by number of
    zero-crossings, similar to frequency ordering in Fourier transforms.

    Parameters
    ----------
    normalized : bool, default=True
        Whether to normalize for orthogonality.
    """

    def __init__(self, normalized: bool = True):
        self.normalized = normalized
        self.hadamard = HadamardTransform(normalized=normalized)

    def transform(self, x: Tensor, dim: int = -1) -> Tensor:
        """Apply sequency-ordered Hadamard transform.

        Parameters
        ----------
        x : Tensor
            Input tensor. Size along dim must be power of 2.
        dim : int, default=-1
            Dimension along which to apply transform.

        Returns
        -------
        Tensor
            Sequency-ordered Hadamard coefficients.
        """
        # Apply standard Hadamard transform
        result = self.hadamard.transform(x, dim)

        # Reorder to sequency ordering
        n = x.shape[dim]
        indices = self._get_sequency_indices(n).to(x.device)

        result = result[..., indices] if dim == -1 else torch.index_select(result, dim, indices)

        return result

    def inverse_transform(self, x: Tensor, dim: int = -1) -> Tensor:
        """Apply inverse sequency-ordered Hadamard transform.

        Parameters
        ----------
        x : Tensor
            Sequency-ordered Hadamard coefficients.
        dim : int, default=-1
            Dimension along which to apply inverse transform.

        Returns
        -------
        Tensor
            Inverse transformed tensor.
        """
        n = x.shape[dim]

        # Get inverse permutation
        indices = self._get_sequency_indices(n).to(x.device)
        inverse_indices = torch.zeros_like(indices)
        inverse_indices[indices] = torch.arange(n, device=x.device)

        # Reorder from sequency to natural ordering
        if dim == -1:
            x_reordered = x[..., inverse_indices]
        else:
            x_reordered = torch.index_select(x, dim, inverse_indices)

        # Apply inverse Hadamard
        return self.hadamard.inverse_transform(x_reordered, dim)

    def _get_sequency_indices(self, n: int) -> Tensor:
        """Get permutation indices for sequency ordering.

        Parameters
        ----------
        n : int
            Transform size (must be power of 2).

        Returns
        -------
        Tensor
            Permutation indices for sequency ordering.
        """
        # Generate Gray code sequence for sequency ordering
        indices = torch.arange(n)
        gray_code = indices ^ (indices >> 1)

        # Sort by Gray code to get sequency order
        _, sequency_indices = torch.sort(gray_code)

        return sequency_indices


@register_component("transform", "slant")
class SlantTransform(OrthogonalTransform):
    """Slant Transform.

    The Slant transform is similar to Hadamard but with varying
    basis function slopes, providing better energy compaction
    for certain signals.

    Parameters
    ----------
    normalized : bool, default=True
        Whether to normalize for orthogonality.
    """

    def __init__(self, normalized: bool = True):
        self.normalized = normalized

    def transform(self, x: Tensor, dim: int = -1) -> Tensor:
        """Apply Slant transform.

        Parameters
        ----------
        x : Tensor
            Input tensor. Size along dim should be power of 2.
        dim : int, default=-1
            Dimension along which to apply transform.

        Returns
        -------
        Tensor
            Slant transformed tensor.
        """
        n = x.shape[dim]

        # Check if n is power of 2
        if n & (n - 1) != 0:
            raise ValueError(f"Slant transform works best with size as power of 2, got {n}")

        # Create Slant matrix
        slant_matrix = self._create_slant_matrix(n, x.device, x.dtype)

        # Apply transform via matrix multiplication
        if dim == -1 or dim == x.ndim - 1:
            result = torch.matmul(x, slant_matrix.T)
        else:
            x_moved = x.transpose(dim, -1)
            result = torch.matmul(x_moved, slant_matrix.T)
            result = result.transpose(dim, -1)

        return result

    def inverse_transform(self, x: Tensor, dim: int = -1) -> Tensor:
        """Apply inverse Slant transform.

        Parameters
        ----------
        x : Tensor
            Slant coefficients.
        dim : int, default=-1
            Dimension along which to apply inverse transform.

        Returns
        -------
        Tensor
            Inverse transformed tensor.
        """
        n = x.shape[dim]

        # Create Slant matrix (orthogonal, so inverse is transpose)
        slant_matrix = self._create_slant_matrix(n, x.device, x.dtype)

        # Apply inverse transform
        if dim == -1 or dim == x.ndim - 1:
            result = torch.matmul(x, slant_matrix)
        else:
            x_moved = x.transpose(dim, -1)
            result = torch.matmul(x_moved, slant_matrix)
            result = result.transpose(dim, -1)

        return result

    def _create_slant_matrix(self, n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Create the Slant transform matrix.

        Parameters
        ----------
        n : int
            Size of transform.
        device : torch.device
            Device for the matrix.
        dtype : torch.dtype
            Data type for the matrix.

        Returns
        -------
        Tensor
            Slant transformation matrix of shape (n, n).
        """
        # Simplified Slant matrix construction
        # For full implementation, use recursive construction
        matrix = torch.zeros(n, n, device=device, dtype=dtype)

        # First row: constant
        matrix[0, :] = 1.0 / math.sqrt(n)

        if n > 1:
            # Second row: linear slant
            for j in range(n):
                matrix[1, j] = (2 * j - n + 1) / math.sqrt(n * (n - 1) * n / 3)

        # For simplicity, fill remaining with Hadamard-like pattern
        # Full implementation would use proper Slant recursion
        h_matrix = self._simple_hadamard_matrix(n, device, dtype)
        if n > 2:
            matrix[2:, :] = h_matrix[2:, :]

        if self.normalized:
            # Orthonormalize via QR decomposition
            q, _ = torch.linalg.qr(matrix.T)
            matrix = q.T

        return matrix

    def _simple_hadamard_matrix(self, n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Create a simple Hadamard matrix for fallback.

        Parameters
        ----------
        n : int
            Matrix size.
        device : torch.device
            Device for the matrix.
        dtype : torch.dtype
            Data type.

        Returns
        -------
        Tensor
            Simple Hadamard-like matrix.
        """
        if n == 1:
            return torch.ones(1, 1, device=device, dtype=dtype)

        # Recursive construction (simplified)
        h_half = self._simple_hadamard_matrix(n // 2, device, dtype) if n > 2 else torch.ones(1, 1, device=device, dtype=dtype)
        h_n = torch.zeros(n, n, device=device, dtype=dtype)

        # Fill quadrants
        h_n[:n//2, :n//2] = h_half
        h_n[:n//2, n//2:] = h_half
        h_n[n//2:, :n//2] = h_half
        h_n[n//2:, n//2:] = -h_half

        return h_n / math.sqrt(2)


__all__: list[str] = [
    "HadamardTransform",
    "HadamardTransform2D",
    "SequencyHadamardTransform",
    "SlantTransform",
]
