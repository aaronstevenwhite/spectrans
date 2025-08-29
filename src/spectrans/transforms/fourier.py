"""Fourier transform implementations."""

import torch

from ..core.registry import register_component
from ..core.types import ComplexTensor, FFTNorm, Tensor
from .base import SpectralTransform2D, UnitaryTransform

# PyTorch FFT functions have Any return types in current mypy stubs
# These are known to return tensors, so we suppress the warnings


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
        return torch.fft.fft(x, dim=dim, norm=self.norm)  # type: ignore[no-any-return]

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
        return torch.fft.ifft(x, dim=dim, norm=self.norm)  # type: ignore[no-any-return]


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
        return torch.fft.fft2(x, dim=dim, norm=self.norm)  # type: ignore[no-any-return]

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
        return torch.fft.ifft2(x, dim=dim, norm=self.norm)  # type: ignore[no-any-return]


@register_component("transform", "rfft")
class RFFT(UnitaryTransform):
    """Real Fast Fourier Transform.

    Applies FFT to real-valued inputs, returning only the positive
    frequency components (more efficient than full FFT).

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
        return torch.fft.rfft(x, dim=dim, norm=self.norm)  # type: ignore[no-any-return]

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
        return torch.fft.irfft(x, n=n, dim=dim, norm=self.norm)  # type: ignore[no-any-return]


@register_component("transform", "rfft2d")
class RFFT2D(SpectralTransform2D):
    """2D Real Fast Fourier Transform.

    Applies 2D FFT to real-valued inputs, efficient for real signals.

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
        return torch.fft.rfft2(x, dim=dim, norm=self.norm)  # type: ignore[no-any-return]

    def inverse_transform(
        self,
        x: ComplexTensor,
        dim: tuple[int, int] = (-2, -1),
        s: tuple[int, int] | None = None
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
        return torch.fft.irfft2(x, s=s, dim=dim, norm=self.norm)  # type: ignore[no-any-return]


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
            x_freq = torch.fft.rfft(x, dim=dim, norm=self.norm)
        else:
            x_freq = torch.fft.rfftn(x, dim=dim, norm=self.norm)

        # Truncate frequencies
        if isinstance(dim, int):
            truncated = x_freq[..., :self.output_size[0] // 2 + 1]
        else:
            # Handle multi-dimensional truncation
            slices = [slice(None)] * x_freq.ndim
            for i, d in enumerate(dim):
                size = self.output_size[i] if i < len(self.output_size) else x_freq.shape[d]
                slices[d] = slice(0, size // 2 + 1) if d == dim[-1] else slice(0, size)
            truncated = x_freq[tuple(slices)]

        # Convert back to spatial domain
        if isinstance(dim, int):
            return torch.fft.irfft(truncated, n=self.output_size[0], dim=dim, norm=self.norm)  # type: ignore[no-any-return]
        else:
            return torch.fft.irfftn(truncated, s=self.output_size, dim=dim, norm=self.norm)  # type: ignore[no-any-return]

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
