"""Base classes and interfaces for spectral transforms.

This module defines the foundational interfaces for all spectral transform implementations
used in spectral transformer neural networks. The class hierarchy accommodates the
mathematical diversity of different transform types while maintaining consistent APIs
for integration with neural network layers.

The transforms are categorized by their mathematical properties and interface requirements,
enabling type-safe composition and clear separation of concerns between different
spectral analysis approaches.

Classes
-------
Transform
    Minimal base class for all transforms.
SpectralTransform
    Interface for simple 1D spectral transforms (FFT, DCT, DST, Hadamard).
SpectralTransform2D
    Interface for 2D spectral transforms with tuple dimension parameters.
MultiResolutionTransform
    Interface for transforms returning multi-resolution coefficients (DWT).
MultiResolutionTransform2D
    Interface for 2D multi-resolution transforms.
OrthogonalTransform
    Base class for orthogonal transforms preserving inner products.
UnitaryTransform
    Base class for unitary transforms preserving complex inner products.
NeuralSpectralTransform
    Base class for learnable transforms with neural network parameters.
InvertibleTransform
    Base class for transforms guaranteeing exact invertibility.
AdaptiveTransform
    Base class for transforms with learnable basis functions.

Examples
--------
Implementing a custom 1D spectral transform:

>>> from spectrans.transforms.base import SpectralTransform
>>> import torch
>>> class IdentityTransform(SpectralTransform):
...     def transform(self, x, dim=-1):
...         return x
...     def inverse_transform(self, x, dim=-1):
...         return x

Using the invertibility checker:

>>> from spectrans.transforms.base import InvertibleTransform
>>> class MyTransform(InvertibleTransform):
...     def transform(self, x, dim=-1):
...         return torch.fft.fft(x, dim=dim)
...     def inverse_transform(self, x, dim=-1):
...         return torch.fft.ifft(x, dim=dim)
>>> transform = MyTransform()
>>> test_input = torch.randn(10, 256, dtype=torch.complex64)
>>> is_invertible = transform.check_invertibility(test_input)

Implementing a multi-resolution transform:

>>> from spectrans.transforms.base import MultiResolutionTransform
>>> class SimpleWavelet(MultiResolutionTransform):
...     def decompose(self, x, levels=None, dim=-1):
...         # Implementation returns (approximation, [details])
...         approx = x[..., ::2]  # Simplified downsampling
...         detail = x[..., 1::2]
...         return approx, [detail]
...     def reconstruct(self, coeffs, dim=-1):
...         # Simplified reconstruction
...         approx, details = coeffs
...         return torch.stack([approx, details[0]], dim=-1).flatten(-2)

Notes
-----
Design Principles:

1. **Mathematical Correctness**: Each base class enforces the mathematical
   properties of its transform family (orthogonality, unitarity, etc.)

2. **Interface Segregation**: Different transform types have separate interfaces
   to avoid forcing incompatible operations into the same signature

3. **Composition Support**: All transforms can be composed and chained while
   maintaining proper mathematical properties

4. **Gradient Compatibility**: All transforms support automatic differentiation
   for end-to-end neural network training

Transform Categories by Mathematical Properties:

**Simple Spectral Transforms** (SpectralTransform):
- Map Tensor → Tensor along a single dimension
- Examples: 1D FFT, DCT, DST, Hadamard
- Interface: transform(), inverse_transform() with dim parameter

**2D Spectral Transforms** (SpectralTransform2D):
- Operate on 2D data with tuple dimension parameters
- Examples: 2D FFT for AFNO, 2D DCT
- Interface: transform(), inverse_transform() with dim tuple

**Multi-Resolution Transforms** (MultiResolutionTransform):
- Decompose into multiple resolution levels
- Examples: Discrete Wavelet Transform (DWT)
- Interface: decompose(), reconstruct() returning coefficient tuples

**Property-Based Classifications**:
- **Orthogonal**: Preserve real inner products (DCT, Hadamard)
- **Unitary**: Preserve complex inner products (FFT)
- **Invertible**: Guarantee numerical invertibility with tolerance checking
- **Adaptive**: Learn transform parameters during training

See Also
--------
spectrans.transforms.fourier : FFT implementations
spectrans.transforms.cosine : DCT/DST implementations
spectrans.transforms.hadamard : Hadamard transform implementations
spectrans.transforms.wavelet : Wavelet transform implementations
spectrans.core.types : Type definitions for transform interfaces
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..core.types import Tensor


class Transform(nn.Module, ABC):
    """Minimal common base class for all transforms.

    This provides the minimal interface that all transforms share,
    allowing for flexible composition through the registry system
    without forcing incompatible mathematical operations into the
    same interface signatures.
    """

    pass


class SpectralTransform(Transform):
    """Base class for simple spectral transforms.

    For transforms that map Tensor → Tensor along a specified dimension,
    such as FFT, DCT, DST, and Hadamard transforms. These transforms
    operate on a single dimension and return tensors of the same shape.

    Mathematical operations supported:
    - Fourier transforms (FFT, RFFT)
    - Discrete Cosine Transform (DCT)
    - Discrete Sine Transform (DST)
    - Hadamard transform
    """

    @abstractmethod
    def transform(self, x: Tensor, dim: int = -1) -> Tensor:
        """Apply forward transform along specified dimension.

        Parameters
        ----------
        x : Tensor
            Input tensor to transform.
        dim : int, default=-1
            Dimension along which to apply the transform.

        Returns
        -------
        Tensor
            Transformed tensor with same shape as input.
        """
        pass

    @abstractmethod
    def inverse_transform(self, x: Tensor, dim: int = -1) -> Tensor:
        """Apply inverse transform along specified dimension.

        Parameters
        ----------
        x : Tensor
            Transformed tensor to invert.
        dim : int, default=-1
            Dimension along which to apply the inverse transform.

        Returns
        -------
        Tensor
            Inverse transformed tensor with same shape as input.
        """
        pass

    @property
    def is_orthogonal(self) -> bool:
        """Whether the transform is orthogonal.

        Returns
        -------
        bool
            True if the transform preserves inner products.
        """
        return False

    @property
    def is_unitary(self) -> bool:
        """Whether the transform is unitary.

        Returns
        -------
        bool
            True if the transform preserves complex inner products.
        """
        return False


class SpectralTransform2D(Transform):
    """Base class for 2D spectral transforms.

    For transforms that operate on 2D data with tuple dimension parameters,
    such as 2D FFT, 2D DCT, or other transforms that require operating
    along multiple dimensions simultaneously.

    These transforms are essential for AFNO and other architectures that
    perform 2D Fourier operations as specified in the implementation plan.
    """

    @abstractmethod
    def transform(self, x: Tensor, dim: tuple[int, int] = (-2, -1)) -> Tensor:
        """Apply forward 2D transform along specified dimensions.

        Parameters
        ----------
        x : Tensor
            Input tensor to transform.
        dim : tuple[int, int], default=(-2, -1)
            Dimensions along which to apply the transform.

        Returns
        -------
        Tensor
            Transformed tensor with same shape as input.
        """
        pass

    @abstractmethod
    def inverse_transform(self, x: Tensor, dim: tuple[int, int] = (-2, -1)) -> Tensor:
        """Apply inverse 2D transform along specified dimensions.

        Parameters
        ----------
        x : Tensor
            Transformed tensor to invert.
        dim : tuple[int, int], default=(-2, -1)
            Dimensions along which to apply the inverse transform.

        Returns
        -------
        Tensor
            Inverse transformed tensor with same shape as input.
        """
        pass


class MultiResolutionTransform(Transform):
    """Base class for multi-resolution transforms.

    For transforms that decompose signals into multiple components at
    different resolution levels, such as Discrete Wavelet Transform (DWT).

    These transforms are mathematically different from simple spectral
    transforms as they return multiple components:
    - Approximation coefficients at the coarsest level
    - Detail coefficients at each level

    This matches the mathematical formulation:
    DWT(x) = {c_{A_J}, {c_{D_j}}_{j=1}^J}

    Parameters
    ----------
    levels : int, default=1
        Number of decomposition levels.
    """

    def __init__(self, levels: int = 1):
        super().__init__()
        self.levels = levels

    @abstractmethod
    def decompose(
        self, x: Tensor, levels: int | None = None, dim: int = -1
    ) -> tuple[Tensor, list[Tensor]]:
        """Decompose signal into multiple resolution levels.

        Parameters
        ----------
        x : Tensor
            Input tensor to decompose.
        levels : int | None, default=None
            Number of levels. If None, use self.levels.
        dim : int, default=-1
            Dimension along which to apply decomposition.

        Returns
        -------
        tuple[Tensor, list[Tensor]]
            Tuple of (approximation_coefficients, detail_coefficients_list)
            where detail_coefficients_list contains coefficients from
            coarsest to finest level.
        """
        pass

    @abstractmethod
    def reconstruct(self, coeffs: tuple[Tensor, list[Tensor]], dim: int = -1) -> Tensor:
        """Reconstruct signal from multi-resolution coefficients.

        Parameters
        ----------
        coeffs : tuple[Tensor, list[Tensor]]
            Tuple of (approximation_coefficients, detail_coefficients_list).
        dim : int, default=-1
            Dimension along which to apply reconstruction.

        Returns
        -------
        Tensor
            Reconstructed tensor.
        """
        pass


class MultiResolutionTransform2D(Transform):
    """Base class for 2D multi-resolution transforms.

    For 2D wavelet transforms and other multi-resolution transforms
    that operate on 2D data. Returns coefficients in the standard
    2D wavelet format: (LL, [(LH, HL, HH) per level]).

    Parameters
    ----------
    levels : int, default=1
        Number of decomposition levels.
    """

    def __init__(self, levels: int = 1):
        super().__init__()
        self.levels = levels

    @abstractmethod
    def decompose(
        self, x: Tensor, levels: int | None = None, dim: tuple[int, int] = (-2, -1)
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor, Tensor]]]:
        """Decompose 2D signal into multiple resolution levels.

        Parameters
        ----------
        x : Tensor
            Input 2D tensor to decompose.
        levels : int | None, default=None
            Number of levels. If None, use self.levels.
        dim : tuple[int, int], default=(-2, -1)
            Dimensions along which to apply decomposition.

        Returns
        -------
        tuple[Tensor, list[tuple[Tensor, Tensor, Tensor]]]
            Tuple of (LL_coefficients, [(LH, HL, HH) per level])
            where each tuple contains the three high-frequency subbands.
        """
        pass

    @abstractmethod
    def reconstruct(
        self,
        coeffs: tuple[Tensor, list[tuple[Tensor, Tensor, Tensor]]],
        dim: tuple[int, int] = (-2, -1),
    ) -> Tensor:
        """Reconstruct 2D signal from multi-resolution coefficients.

        Parameters
        ----------
        coeffs : tuple[Tensor, list[tuple[Tensor, Tensor, Tensor]]]
            Tuple of (LL_coefficients, [(LH, HL, HH) per level]).
        dim : tuple[int, int], default=(-2, -1)
            Dimensions along which to apply reconstruction.

        Returns
        -------
        Tensor
            Reconstructed 2D tensor.
        """
        pass


class OrthogonalTransform(SpectralTransform):
    """Base class for orthogonal transforms.

    Orthogonal transforms preserve inner products and have
    the property that their inverse is their transpose.
    This includes DCT, DST, and Hadamard transforms.
    """

    @property
    def is_orthogonal(self) -> bool:
        """Orthogonal transforms preserve inner products."""
        return True


class UnitaryTransform(SpectralTransform):
    """Base class for unitary transforms.

    Unitary transforms preserve complex inner products and have
    the property that their inverse is their conjugate transpose.
    This includes the Discrete Fourier Transform (DFT/FFT).
    """

    @property
    def is_unitary(self) -> bool:
        """Unitary transforms preserve complex inner products."""
        return True


class NeuralSpectralTransform(SpectralTransform):
    """Base class for learnable spectral transforms.

    This class is for transforms that can learn their parameters
    during training, such as learnable filters in the frequency domain.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the neural spectral transform.

        By default, applies the transform operation. Subclasses can
        override this for more complex learned behaviors.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        return self.transform(x)


class InvertibleTransform(SpectralTransform):
    """Base class for transforms that guarantee exact invertibility.

    These transforms ensure that inverse_transform(transform(x)) == x
    within numerical precision, which is important for certain
    architectures that require perfect reconstruction.
    """

    def check_invertibility(
        self, x: Tensor, dim: int = -1, rtol: float = 1e-5, atol: float = 1e-8
    ) -> bool:
        """Check if transform is invertible for given input.

        Parameters
        ----------
        x : Tensor
            Input tensor to test.
        dim : int, default=-1
            Dimension along which to test invertibility.
        rtol : float, default=1e-5
            Relative tolerance for comparison.
        atol : float, default=1e-8
            Absolute tolerance for comparison.

        Returns
        -------
        bool
            True if transform is invertible within tolerance.
        """
        transformed = self.transform(x, dim=dim)
        reconstructed = self.inverse_transform(transformed, dim=dim)
        return bool(torch.allclose(x, reconstructed, rtol=rtol, atol=atol))


class AdaptiveTransform(NeuralSpectralTransform):
    """Base class for adaptive transforms with learnable parameters.

    Adaptive transforms can learn their basis functions or
    transformation parameters from data. This is useful for
    applications where the optimal spectral representation
    depends on the specific data distribution.

    Parameters
    ----------
    input_dim : int
        Input dimension size.
    learnable : bool, default=True
        Whether transform parameters are learnable.
    """

    def __init__(self, input_dim: int, learnable: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.learnable = learnable
