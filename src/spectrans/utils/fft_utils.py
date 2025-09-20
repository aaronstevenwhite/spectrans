"""FFT utilities for handling MKL compatibility issues.

This module provides wrapper functions for FFT operations that handle
known compatibility issues with Intel MKL, particularly in backward passes.
"""

import os
import warnings

import torch


def safe_rfft2(
    input: torch.Tensor,
    s: tuple[int, int] | None = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    """Safe wrapper for torch.fft.rfft2 that handles MKL errors.

    This function wraps torch.fft.rfft2 with error handling for known
    MKL compatibility issues in backward passes.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    s : tuple[int, int] | None, optional
        Signal size in the transformed dimensions.
    dim : tuple[int, int], optional
        Dimensions to transform. Default is (-2, -1).
    norm : str | None, optional
        Normalization mode. Can be "forward", "backward", or "ortho".

    Returns
    -------
    torch.Tensor
        The FFT of the input tensor.
    """
    # Check if we should use alternative FFT implementation
    use_fallback = os.environ.get("SPECTRANS_DISABLE_MKL_FFT", "0") == "1"

    if use_fallback:
        # Use a workaround for MKL issues
        # Split into 1D FFTs which are more stable
        result = torch.fft.rfft(input, n=s[1] if s else None, dim=dim[1], norm=norm)
        result = torch.fft.fft(result, n=s[0] if s else None, dim=dim[0], norm=norm)
        return result  # type: ignore[no-any-return]

    try:
        return torch.fft.rfft2(input, s=s, dim=dim, norm=norm)  # type: ignore[no-any-return]
    except RuntimeError as e:
        if "MKL FFT error" in str(e) and "Inconsistent configuration" in str(e):
            warnings.warn(
                "MKL FFT error detected. Falling back to alternative implementation. "
                "Set SPECTRANS_DISABLE_MKL_FFT=1 to always use fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Fallback: Use sequential 1D FFTs
            result = torch.fft.rfft(input, n=s[1] if s else None, dim=dim[1], norm=norm)
            result = torch.fft.fft(result, n=s[0] if s else None, dim=dim[0], norm=norm)
            return result  # type: ignore[no-any-return]
        raise


def safe_irfft2(
    input: torch.Tensor,
    s: tuple[int, int] | None = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    """Safe wrapper for torch.fft.irfft2 that handles MKL errors.

    This function wraps torch.fft.irfft2 with error handling for known
    MKL compatibility issues in backward passes.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor (complex).
    s : tuple[int, int] | None, optional
        Signal size in the transformed dimensions.
    dim : tuple[int, int], optional
        Dimensions to transform. Default is (-2, -1).
    norm : str | None, optional
        Normalization mode. Can be "forward", "backward", or "ortho".

    Returns
    -------
    torch.Tensor
        The inverse FFT of the input tensor.
    """
    # Check if we should use alternative FFT implementation
    use_fallback = os.environ.get("SPECTRANS_DISABLE_MKL_FFT", "0") == "1"

    if use_fallback:
        # Use a workaround for MKL issues
        # Split into 1D FFTs which are more stable
        result = torch.fft.ifft(input, n=s[0] if s else None, dim=dim[0], norm=norm)
        result = torch.fft.irfft(result, n=s[1] if s else None, dim=dim[1], norm=norm)
        return result  # type: ignore[no-any-return]

    try:
        return torch.fft.irfft2(input, s=s, dim=dim, norm=norm)  # type: ignore[no-any-return]
    except RuntimeError as e:
        if "MKL FFT error" in str(e) and "Inconsistent configuration" in str(e):
            warnings.warn(
                "MKL FFT error detected. Falling back to alternative implementation. "
                "Set SPECTRANS_DISABLE_MKL_FFT=1 to always use fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Fallback: Use sequential 1D FFTs
            result = torch.fft.ifft(input, n=s[0] if s else None, dim=dim[0], norm=norm)
            result = torch.fft.irfft(result, n=s[1] if s else None, dim=dim[1], norm=norm)
            return result  # type: ignore[no-any-return]
        raise


def safe_rfft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> torch.Tensor:
    """Safe wrapper for torch.fft.rfft that handles MKL errors.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor.
    n : int | None, optional
        Signal length.
    dim : int, optional
        Dimension to transform. Default is -1.
    norm : str | None, optional
        Normalization mode.

    Returns
    -------
    torch.Tensor
        The FFT of the input tensor.
    """
    # For 1D FFT, MKL issues are less common but we still handle them
    try:
        return torch.fft.rfft(input, n=n, dim=dim, norm=norm)  # type: ignore[no-any-return]
    except RuntimeError as e:
        if "MKL FFT error" in str(e):
            # For 1D, we can try using the full complex FFT
            warnings.warn(
                "MKL FFT error in rfft. Using complex FFT fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            result = torch.fft.fft(input, n=n, dim=dim, norm=norm)
            # Keep only positive frequencies
            n_out = n if n else input.shape[dim]
            return result[..., : n_out // 2 + 1]  # type: ignore[no-any-return]
        raise


def safe_irfft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> torch.Tensor:
    """Safe wrapper for torch.fft.irfft that handles MKL errors.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor (complex).
    n : int | None, optional
        Signal length.
    dim : int, optional
        Dimension to transform. Default is -1.
    norm : str | None, optional
        Normalization mode.

    Returns
    -------
    torch.Tensor
        The inverse FFT of the input tensor.
    """
    try:
        return torch.fft.irfft(input, n=n, dim=dim, norm=norm)  # type: ignore[no-any-return]
    except RuntimeError as e:
        if "MKL FFT error" in str(e):
            warnings.warn(
                "MKL FFT error in irfft. Using complex IFFT fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Reconstruct full spectrum from half spectrum
            n_out = n if n else 2 * (input.shape[dim] - 1)
            full_spectrum = torch.zeros(
                (*input.shape[:dim], n_out, *input.shape[dim + 1 :]),
                dtype=input.dtype,
                device=input.device,
            )
            # Copy positive frequencies
            full_spectrum[..., : input.shape[dim]] = input
            # Mirror negative frequencies (conjugate)
            if n_out > 1:
                full_spectrum[..., -1 : input.shape[dim] : -1] = input[..., 1:].conj()
            result = torch.fft.ifft(full_spectrum, n=n_out, dim=dim, norm=norm)
            return result.real  # type: ignore[no-any-return]
        raise
