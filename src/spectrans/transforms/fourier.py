r"""Fourier transform implementations for spectral neural networks.

This module provides Fourier transform implementations for
spectral transformer architectures. The transforms are built on PyTorch's native
FFT operations for GPU acceleration and automatic differentiation support.

All Fourier transforms in this module are unitary, preserving complex inner products
and maintaining energy conservation (Parseval's theorem). They support various
normalization modes and handle both real and complex inputs efficiently.

Classes
-------
FFT1D
    1D Fast Fourier Transform with configurable normalization.
FFT2D
    2D Fast Fourier Transform for AFNO-style 2D operations.
RFFT
    Real-input Fast Fourier Transform.
RFFT2D
    2D Real-input Fast Fourier Transform.
SpectralPooling
    Spectral domain pooling operation via frequency truncation.

Examples
--------
Basic 1D FFT usage:

>>> import torch
>>> from spectrans.transforms.fourier import FFT1D
>>> fft = FFT1D(norm='ortho')
>>> signal = torch.randn(32, 512, dtype=torch.complex64)
>>> freq_domain = fft.transform(signal, dim=-1)
>>> reconstructed = fft.inverse_transform(freq_domain, dim=-1)

Real-input FFT:

>>> from spectrans.transforms.fourier import RFFT
>>> rfft = RFFT(norm='ortho')
>>> real_signal = torch.randn(32, 512)
>>> freq_domain = rfft.transform(real_signal)  # Returns complex output
>>> # Note: inverse returns real values for real-input FFTs

2D FFT for AFNO operations:

>>> from spectrans.transforms.fourier import FFT2D
>>> fft2d = FFT2D(norm='ortho')
>>> tensor_2d = torch.randn(32, 64, 64, dtype=torch.complex64)
>>> freq_2d = fft2d.transform(tensor_2d, dim=(-2, -1))

Spectral pooling for downsampling:

>>> from spectrans.transforms.fourier import SpectralPooling
>>> pool = SpectralPooling(output_size=256, input_size=512)
>>> downsampled = pool.transform(freq_domain)

Notes
-----
Mathematical Properties:

All Fourier transforms maintain unitarity:

- Energy conservation: $\|\mathcal{F}(\mathbf{x})\|^2 = \|\mathbf{x}\|^2$
- Parseval's theorem: $\langle \mathbf{x}, \mathbf{y} \rangle = \langle \mathcal{F}(\mathbf{x}), \mathcal{F}(\mathbf{y}) \rangle$
- Perfect reconstruction: $\mathcal{F}^{-1}(\mathcal{F}(\mathbf{x})) = \mathbf{x}$

Normalization Modes:

- 'forward': No scaling on forward transform, $\frac{1}{n}$ scaling on inverse
- 'backward': $\frac{1}{n}$ scaling on forward transform, no scaling on inverse
- 'ortho': $\frac{1}{\sqrt{n}}$ scaling on both directions (unitary)

The 'ortho' mode is recommended for neural networks as it preserves numerical
stability and maintains consistent scaling throughout the network.

Real-Input FFT:
RFFT and RFFT2D exploit Hermitian symmetry of real-input FFTs, storing only
the non-redundant frequency components for real-valued inputs.

GPU Acceleration:
All transforms utilize PyTorch's cuFFT backend when tensors are on GPU.

Gradient Support:
All transforms support automatic differentiation through PyTorch's autograd system,
enabling end-to-end training of spectral neural networks.

References
----------
James W. Cooley and John W. Tukey. 1965. An algorithm for the machine calculation
of complex Fourier series. Mathematics of Computation, 19(90):297-301.

Michael T. Heideman, Don H. Johnson, and C. Sidney Burrus. 1984. Gauss and the
history of the fast Fourier transform. IEEE ASSP Magazine, 1(4):14-21.

Steven G. Johnson and Matteo Frigo. 2007. A modified split-radix FFT with fewer
arithmetic operations. IEEE Transactions on Signal Processing, 55(1):111-119.

See Also
--------
spectrans.transforms.base : Base classes for transform interfaces
spectrans.utils.complex : Complex tensor utility functions
spectrans.layers.mixing.fourier : Neural layers using these transforms
"""

from ..core.registry import register_component
from ..core.types import ComplexTensor, FFTNorm, Tensor
from ..utils.fft import (
    safe_fft,
    safe_fft2,
    safe_ifft,
    safe_ifft2,
    safe_irfft,
    safe_irfft2,
    safe_irfftn,
    safe_rfft,
    safe_rfft2,
    safe_rfftn,
)
from .base import SpectralTransform2D, UnitaryTransform


@register_component("transform", "fft1d")
class FFT1D(UnitaryTransform):
    """1D Fast Fourier Transform.

    Applies 1D FFT along a specified dimension of the input tensor.

    Parameters
    ----------
    norm : FFTNorm, default="ortho"
        Normalization mode: "forward", "backward", or "ortho".
    """

    def __init__(self, norm: FFTNorm = "ortho"):
        self.norm = norm

    def transform(self, x: Tensor, dim: int = -1) -> ComplexTensor:
        """Apply 1D FFT.

        Parameters
        ----------
        x : Tensor
            Input tensor of real or complex values.
        dim : int, default=-1
            Dimension along which to apply FFT.

        Returns
        -------
        ComplexTensor
            Complex-valued FFT result.
        """
        return safe_fft(x, dim=dim, norm=self.norm)

    def inverse_transform(self, x: ComplexTensor, dim: int = -1) -> Tensor:
        """Apply inverse 1D FFT.

        Parameters
        ----------
        x : ComplexTensor
            Complex-valued FFT coefficients.
        dim : int, default=-1
            Dimension along which to apply inverse FFT.

        Returns
        -------
        Tensor
            Inverse FFT result (may be complex if input was complex).
        """
        return safe_ifft(x, dim=dim, norm=self.norm)


@register_component("transform", "fft2d")
class FFT2D(SpectralTransform2D):
    """2D Fast Fourier Transform.

    Applies 2D FFT along the last two dimensions of the input tensor.

    Parameters
    ----------
    norm : FFTNorm, default="ortho"
        Normalization mode: "forward", "backward", or "ortho".
    """

    def __init__(self, norm: FFTNorm = "ortho"):
        self.norm = norm

    def transform(self, x: Tensor, dim: tuple[int, int] = (-2, -1)) -> ComplexTensor:
        """Apply 2D FFT.

        Parameters
        ----------
        x : Tensor
            Input tensor of real or complex values.
        dim : tuple[int, int], default=(-2, -1)
            Dimensions along which to apply 2D FFT.

        Returns
        -------
        ComplexTensor
            Complex-valued 2D FFT result.
        """
        return safe_fft2(x, dim=dim, norm=self.norm)

    def inverse_transform(self, x: ComplexTensor, dim: tuple[int, int] = (-2, -1)) -> Tensor:
        """Apply inverse 2D FFT.

        Parameters
        ----------
        x : ComplexTensor
            Complex-valued FFT coefficients.
        dim : tuple[int, int], default=(-2, -1)
            Dimensions along which to apply inverse FFT.

        Returns
        -------
        Tensor
            Inverse FFT result.
        """
        return safe_ifft2(x, dim=dim, norm=self.norm)


@register_component("transform", "rfft")
class RFFT(UnitaryTransform):
    """Real Fast Fourier Transform.

    Applies FFT to real-valued inputs, returning only the positive
    frequency components.

    Parameters
    ----------
    norm : FFTNorm, default="ortho"
        Normalization mode: "forward", "backward", or "ortho".
    """

    def __init__(self, norm: FFTNorm = "ortho"):
        self.norm = norm

    def transform(self, x: Tensor, dim: int = -1) -> ComplexTensor:
        """Apply real FFT.

        Parameters
        ----------
        x : Tensor
            Real-valued input tensor.
        dim : int, default=-1
            Dimension along which to apply RFFT.

        Returns
        -------
        ComplexTensor
            Complex-valued RFFT result (positive frequencies only).
        """
        return safe_rfft(x, dim=dim, norm=self.norm)

    def inverse_transform(self, x: ComplexTensor, dim: int = -1, n: int | None = None) -> Tensor:
        """Apply inverse real FFT.

        Parameters
        ----------
        x : ComplexTensor
            Complex-valued RFFT coefficients.
        dim : int, default=-1
            Dimension along which to apply inverse RFFT.
        n : int | None, default=None
            Length of the output signal. If None, inferred from input.

        Returns
        -------
        Tensor
            Real-valued inverse RFFT result.
        """
        return safe_irfft(x, n=n, dim=dim, norm=self.norm)


@register_component("transform", "rfft2d")
class RFFT2D(SpectralTransform2D):
    """2D Real Fast Fourier Transform.

    Applies 2D FFT to real-valued inputs.

    Parameters
    ----------
    norm : FFTNorm, default="ortho"
        Normalization mode: "forward", "backward", or "ortho".
    """

    def __init__(self, norm: FFTNorm = "ortho"):
        self.norm = norm

    def transform(self, x: Tensor, dim: tuple[int, int] = (-2, -1)) -> ComplexTensor:
        """Apply 2D real FFT.

        Parameters
        ----------
        x : Tensor
            Real-valued input tensor.
        dim : tuple[int, int], default=(-2, -1)
            Dimensions along which to apply 2D RFFT.

        Returns
        -------
        ComplexTensor
            Complex-valued 2D RFFT result.
        """
        return safe_rfft2(x, dim=dim, norm=self.norm)

    def inverse_transform(
        self, x: ComplexTensor, dim: tuple[int, int] = (-2, -1), s: tuple[int, int] | None = None
    ) -> Tensor:
        """Apply inverse 2D real FFT.

        Parameters
        ----------
        x : ComplexTensor
            Complex-valued RFFT coefficients.
        dim : tuple[int, int], default=(-2, -1)
            Dimensions along which to apply inverse RFFT.
        s : tuple[int, int] | None, default=None
            Output signal size. If None, inferred from input.

        Returns
        -------
        Tensor
            Real-valued inverse RFFT result.
        """
        return safe_irfft2(x, s=s, dim=dim, norm=self.norm)


@register_component("transform", "spectral_pool")
class SpectralPooling(UnitaryTransform):
    """Spectral pooling via frequency domain truncation.

    Reduces spatial dimensions by truncating high-frequency components
    in the Fourier domain.

    Parameters
    ----------
    output_size : int | tuple[int, ...]
        Target output size after pooling.
    norm : FFTNorm, default="ortho"
        Normalization mode for FFT operations.
    """

    def __init__(self, output_size: int | tuple[int, ...], norm: FFTNorm = "ortho"):
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size,)
        self.norm = norm

    def transform(self, x: Tensor, dim: int | tuple[int, ...] = -1) -> Tensor:
        """Apply spectral pooling.

        Parameters
        ----------
        x : Tensor
            Input tensor to pool.
        dim : int | tuple[int, ...], default=-1
            Dimensions to pool along.

        Returns
        -------
        Tensor
            Spectrally pooled tensor.
        """
        # Convert to frequency domain
        if isinstance(dim, int):
            x_freq = safe_rfft(x, dim=dim, norm=self.norm)
        else:
            x_freq = safe_rfftn(x, dim=dim, norm=self.norm)

        # Truncate frequencies
        if isinstance(dim, int):
            truncated = x_freq[..., : self.output_size[0] // 2 + 1]
        else:
            # Handle multi-dimensional truncation
            slices = [slice(None)] * x_freq.ndim
            for i, d in enumerate(dim):
                size = self.output_size[i] if i < len(self.output_size) else x_freq.shape[d]
                slices[d] = slice(0, size // 2 + 1) if d == dim[-1] else slice(0, size)
            truncated = x_freq[tuple(slices)]

        # Convert back to spatial domain
        if isinstance(dim, int):
            return safe_irfft(truncated, n=self.output_size[0], dim=dim, norm=self.norm)
        else:
            return safe_irfftn(truncated, s=self.output_size, dim=dim, norm=self.norm)

    def inverse_transform(self, x: Tensor, dim: int | tuple[int, ...] = -1) -> Tensor:
        """Inverse is not well-defined for pooling operations."""
        raise NotImplementedError("Spectral pooling is not invertible due to information loss")


__all__: list[str] = [
    "FFT1D",
    "FFT2D",
    "RFFT",
    "RFFT2D",
    "SpectralPooling",
]
