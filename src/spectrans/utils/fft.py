r"""FFT utilities with MKL compatibility fallbacks.

This module provides wrapper functions for FFT operations that handle
MKL compatibility issues on certain platforms. When MKL FFT operations
fail due to configuration inconsistencies, these functions automatically
fall back to DFT matrix multiplication or sequential 1D transforms.

The fallback implementations maintain mathematical correctness while
avoiding MKL-specific issues. All transforms preserve the standard
FFT properties including linearity, unitarity (with appropriate
normalization), and the convolution theorem.

Environment Variables
---------------------
SPECTRANS_DISABLE_MKL_FFT : str
    Set to "1" to force use of fallback implementations.

Notes
-----
The fallback algorithms trade computational efficiency for compatibility.
DFT matrix multiplication has $O(n^2)$ complexity compared to FFT's
$O(n \log n)$, but ensures consistent behavior across platforms.

For 2D transforms, the module decomposes operations into sequential
1D transforms following the separability property:

$$
\mathcal{F}_{2D}[f(x,y)] = \mathcal{F}_y[\mathcal{F}_x[f(x,y)]]
$$

where $\mathcal{F}_x$ and $\mathcal{F}_y$ denote 1D transforms along
the respective dimensions.

References
----------
James W. Cooley and John W. Tukey. 1965. An algorithm for the machine
calculation of complex Fourier series. Mathematics of Computation,
19(90):297-301.
"""

import os
import warnings
from typing import cast

import torch


def safe_rfft2(
    input: torch.Tensor,
    s: tuple[int, int] | None = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    r"""2D real FFT with MKL fallback support.

    Computes the 2D discrete Fourier transform of real-valued input,
    exploiting Hermitian symmetry to store only non-redundant frequencies.
    Falls back to sequential 1D transforms when MKL errors occur.

    Parameters
    ----------
    input : torch.Tensor
        Real-valued input tensor with at least 2 dimensions.
    s : tuple[int, int] | None, optional
        Output sizes along transformed dimensions. If None, uses input sizes.
    dim : tuple[int, int], default=(-2, -1)
        Dimensions over which to compute the FFT.
    norm : str | None, optional
        Normalization mode. Options are:
        - "forward": normalize by 1/n on forward transform
        - "backward": normalize by 1/n on inverse transform
        - "ortho": normalize by 1/sqrt(n) for unitarity

    Returns
    -------
    torch.Tensor
        Complex tensor containing FFT coefficients. Due to Hermitian
        symmetry, the last dimension contains only positive frequencies
        with size (n//2 + 1) for input size n along that dimension.

    Notes
    -----
    The 2D real FFT computes:

    $$
    X[k_1, k_2] = \sum_{n_1=0}^{N_1-1} \sum_{n_2=0}^{N_2-1} x[n_1, n_2]
                  e^{-2\pi i (k_1 n_1/N_1 + k_2 n_2/N_2)}
    $$

    For real input, the output satisfies the Hermitian symmetry property:
    $X[k_1, k_2] = X^*[N_1-k_1, N_2-k_2]$, allowing storage of only
    positive frequencies along the last dimension.

    The fallback implementation decomposes the 2D transform into sequential
    1D operations: first an RFFT along dim[1], then an FFT along dim[0].
    This order ensures RFFT operates on real input as required.
    """
    # Check if we should use alternative FFT implementation
    use_fallback = os.environ.get("SPECTRANS_DISABLE_MKL_FFT", "0") == "1"

    if use_fallback:
        # Check for empty transform dimensions - match PyTorch FFT behavior
        if input.shape[dim[0]] == 0 or input.shape[dim[1]] == 0:
            raise RuntimeError("FFT operations do not support empty tensors")
        # Use a workaround for MKL issues
        # For 2D RFFT, we decompose into 1D operations
        # Must apply RFFT first (requires real input), then FFT on complex output
        # First apply RFFT along the second dimension (dim[1])
        result = safe_rfft(input, n=s[1] if s else None, dim=dim[1], norm=norm)
        # Then apply FFT along the first dimension (dim[0]) on complex result
        result = safe_fft(result, n=s[0] if s else None, dim=dim[0], norm=norm)
        return result

    try:
        return cast(torch.Tensor, torch.fft.rfft2(input, s=s, dim=dim, norm=norm))
    except RuntimeError as e:
        if "MKL FFT error" in str(e) and "Inconsistent configuration" in str(e):
            warnings.warn(
                "MKL FFT error detected. Falling back to alternative implementation. "
                "Set SPECTRANS_DISABLE_MKL_FFT=1 to always use fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Fallback: Use sequential 1D FFTs
            # Must apply RFFT first (requires real input), then FFT on complex output
            result = safe_rfft(input, n=s[1] if s else None, dim=dim[1], norm=norm)
            result = safe_fft(result, n=s[0] if s else None, dim=dim[0], norm=norm)
            return result
        raise


def safe_irfft2(
    input: torch.Tensor,
    s: tuple[int, int] | None = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    r"""2D inverse real FFT with MKL fallback support.

    Computes the 2D inverse discrete Fourier transform for real-valued
    output from Hermitian-symmetric frequency domain input.

    Parameters
    ----------
    input : torch.Tensor
        Complex tensor containing Hermitian-symmetric FFT coefficients.
        The last dimension should contain only positive frequencies.
    s : tuple[int, int] | None, optional
        Output spatial sizes. If None, inferred from input:
        s[1] = 2*(input.shape[dim[1]]-1) for the last dimension.
    dim : tuple[int, int], default=(-2, -1)
        Dimensions over which to compute the inverse FFT.
    norm : str | None, optional
        Normalization mode matching forward transform.

    Returns
    -------
    torch.Tensor
        Real-valued spatial domain tensor.

    Notes
    -----
    The inverse transform reconstructs the spatial signal:

    $$
    x[n_1, n_2] = \frac{1}{N_1 N_2} \sum_{k_1=0}^{N_1-1} \sum_{k_2=0}^{N_2-1}
                  X[k_1, k_2] e^{2\pi i (k_1 n_1/N_1 + k_2 n_2/N_2)}
    $$

    where the normalization factor depends on the norm parameter.

    The fallback implementation reverses the forward transform decomposition:
    first IFFT along dim[0], then IRFFT along dim[1]. This ensures the final
    IRFFT produces real-valued output as expected.
    """
    # Check if we should use alternative FFT implementation
    use_fallback = os.environ.get("SPECTRANS_DISABLE_MKL_FFT", "0") == "1"

    if use_fallback:
        # Use a workaround for MKL issues
        # For 2D IRFFT, reverse the order of forward transform
        # First apply IFFT along the first dimension (dim[0])
        result = safe_ifft(input, n=s[0] if s else None, dim=dim[0], norm=norm)
        # Then apply IRFFT along the second dimension (dim[1]) to get real output
        result = safe_irfft(result, n=s[1] if s else None, dim=dim[1], norm=norm)
        # IRFFT already returns real values
        return result

    try:
        return cast(torch.Tensor, torch.fft.irfft2(input, s=s, dim=dim, norm=norm))
    except RuntimeError as e:
        if "MKL FFT error" in str(e) and "Inconsistent configuration" in str(e):
            warnings.warn(
                "MKL FFT error detected. Falling back to alternative implementation. "
                "Set SPECTRANS_DISABLE_MKL_FFT=1 to always use fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Fallback: Use sequential 1D FFTs (reverse of forward transform)
            # First apply IFFT along the first dimension, then IRFFT along the second
            result = safe_ifft(input, n=s[0] if s else None, dim=dim[0], norm=norm)
            result = safe_irfft(result, n=s[1] if s else None, dim=dim[1], norm=norm)
            # IRFFT already returns real values
            return result
        raise


def safe_rfft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> torch.Tensor:
    r"""1D real FFT with MKL fallback support.

    Computes the 1D discrete Fourier transform of real-valued input along
    a single dimension, storing only positive frequency components.

    Parameters
    ----------
    input : torch.Tensor
        Real-valued input tensor.
    n : int | None, optional
        Signal length along transform dimension. Input is truncated or
        zero-padded to match. If None, uses input size.
    dim : int, default=-1
        Dimension along which to compute the FFT.
    norm : str | None, optional
        Normalization mode for scaling FFT outputs.

    Returns
    -------
    torch.Tensor
        Complex tensor of shape (..., n//2+1, ...) containing positive
        frequency coefficients along the specified dimension.

    Notes
    -----
    For real input of length $n$, the FFT satisfies Hermitian symmetry:
    $X[k] = X^*[n-k]$. This function returns only frequencies $k \in [0, n/2]$,
    reducing memory usage by nearly half.

    The fallback uses full complex FFT and extracts positive frequencies,
    maintaining mathematical equivalence while avoiding MKL issues.
    """
    use_fallback = os.environ.get("SPECTRANS_DISABLE_MKL_FFT", "0") == "1"

    if use_fallback:
        # Check for empty transform dimension - match PyTorch FFT behavior
        if input.shape[dim] == 0:
            raise RuntimeError("FFT operations do not support empty tensors")
        # Use complex FFT and keep only positive frequencies
        result = safe_fft(input, n=n, dim=dim, norm=norm)
        n_out = n if n else input.shape[dim]
        # Slice along the specified dimension, not always the last
        indices = [slice(None)] * result.ndim
        indices[dim] = slice(0, n_out // 2 + 1)
        return result[tuple(indices)]

    # For 1D FFT, MKL issues are less common but we still handle them
    try:
        return cast(torch.Tensor, torch.fft.rfft(input, n=n, dim=dim, norm=norm))
    except RuntimeError as e:
        if "MKL FFT error" in str(e):
            # For 1D, we can try using the full complex FFT
            warnings.warn(
                "MKL FFT error in rfft. Using complex FFT fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            result = safe_fft(input, n=n, dim=dim, norm=norm)
            # Keep only positive frequencies
            n_out = n if n else input.shape[dim]
            # Slice along the specified dimension, not always the last
            indices = [slice(None)] * result.ndim
            indices[dim] = slice(0, n_out // 2 + 1)
            return result[tuple(indices)]
        raise


def safe_irfft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> torch.Tensor:
    r"""1D inverse real FFT with MKL fallback support.

    Reconstructs real-valued signal from Hermitian-symmetric frequency
    domain representation containing only positive frequencies.

    Parameters
    ----------
    input : torch.Tensor
        Complex tensor containing positive frequency coefficients.
        Shape along dim should be n//2+1 for output length n.
    n : int | None, optional
        Output signal length. If None, inferred as 2*(input_size-1).
    dim : int, default=-1
        Dimension along which to compute inverse FFT.
    norm : str | None, optional
        Normalization mode matching the forward transform.

    Returns
    -------
    torch.Tensor
        Real-valued reconstructed signal of length n along specified dimension.

    Notes
    -----
    This function reconstructs the full Hermitian-symmetric spectrum from
    positive frequencies before applying inverse FFT. For input containing
    $m = n//2 + 1$ frequencies, the reconstruction satisfies:

    $$
    x[j] = \frac{1}{n} \sum_{k=0}^{n-1} X[k] e^{2\pi i jk/n}
    $$

    where negative frequencies are constructed via conjugate symmetry:
    $X[n-k] = X^*[k]$ for $k > 0$.

    The fallback explicitly constructs the full spectrum and uses complex
    IFFT, ensuring correctness when MKL operations fail.
    """
    use_fallback = os.environ.get("SPECTRANS_DISABLE_MKL_FFT", "0") == "1"

    if use_fallback:
        # Reconstruct full spectrum from half spectrum
        n_out = n if n else 2 * (input.shape[dim] - 1)

        # Build indices for creating full spectrum
        shape = list(input.shape)
        shape[dim] = n_out
        full_spectrum = torch.zeros(shape, dtype=input.dtype, device=input.device)

        # Copy positive frequencies
        n_positive = input.shape[dim]
        indices = [slice(None)] * len(input.shape)
        indices[dim] = slice(0, n_positive)
        full_spectrum[tuple(indices)] = input

        # Mirror negative frequencies (conjugate) - skip DC and Nyquist
        if n_out > n_positive:
            indices_pos: list[slice | int] = [slice(None)] * len(input.shape)
            indices_neg: list[slice | int] = [slice(None)] * len(input.shape)

            # Copy from indices 1 to n_positive-1 (skip DC)
            # to indices -1 to -(n_positive-1) (reverse order, conjugate)
            for i in range(1, min(n_positive - 1, n_out - n_positive + 1)):
                indices_pos[dim] = i
                indices_neg[dim] = -i
                full_spectrum[tuple(indices_neg)] = input[tuple(indices_pos)].conj()

        result = safe_ifft(full_spectrum, n=n_out, dim=dim, norm=norm)
        return result.real

    try:
        return cast(torch.Tensor, torch.fft.irfft(input, n=n, dim=dim, norm=norm))
    except RuntimeError as e:
        if "MKL FFT error" in str(e):
            warnings.warn(
                "MKL FFT error in irfft. Using complex IFFT fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Reconstruct full spectrum from half spectrum
            n_out = n if n else 2 * (input.shape[dim] - 1)

            # Build indices for creating full spectrum
            shape = list(input.shape)
            shape[dim] = n_out
            full_spectrum = torch.zeros(shape, dtype=input.dtype, device=input.device)

            # Copy positive frequencies
            n_positive = input.shape[dim]
            indices = [slice(None)] * len(input.shape)
            indices[dim] = slice(0, n_positive)
            full_spectrum[tuple(indices)] = input

            # Mirror negative frequencies (conjugate) - skip DC and Nyquist
            if n_out > n_positive:
                idx_pos: list[slice | int] = [slice(None)] * len(input.shape)
                idx_neg: list[slice | int] = [slice(None)] * len(input.shape)

                # Copy from indices 1 to n_positive-1 (skip DC)
                # to indices -1 to -(n_positive-1) (reverse order, conjugate)
                for i in range(1, min(n_positive - 1, n_out - n_positive + 1)):
                    idx_pos[dim] = i
                    idx_neg[dim] = -i
                    full_spectrum[tuple(idx_neg)] = input[tuple(idx_pos)].conj()

            result = safe_ifft(full_spectrum, n=n_out, dim=dim, norm=norm)
            return result.real
        raise


def safe_fft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> torch.Tensor:
    r"""1D complex FFT with MKL fallback support.

    Computes the 1D discrete Fourier transform along a single dimension,
    supporting both real and complex inputs.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor (real or complex).
    n : int | None, optional
        Signal length along transform dimension. Input is truncated or
        zero-padded to match. If None, uses input size.
    dim : int, default=-1
        Dimension along which to compute the FFT.
    norm : str | None, optional
        Normalization mode affecting output scaling.

    Returns
    -------
    torch.Tensor
        Complex tensor containing frequency domain representation.

    Notes
    -----
    The discrete Fourier transform is defined as:

    $$
    X[k] = \sum_{n=0}^{N-1} x[n] e^{-2\pi i kn/N}
    $$

    The fallback implementation uses explicit DFT matrix multiplication
    with $O(n^2)$ complexity when MKL FFT fails. While computationally
    less efficient than FFT's $O(n \log n)$, it guarantees correctness
    across all platforms.
    """
    use_fallback = os.environ.get("SPECTRANS_DISABLE_MKL_FFT", "0") == "1"

    if use_fallback:
        # Check for empty transform dimension - match PyTorch FFT behavior
        if input.shape[dim] == 0:
            raise RuntimeError("FFT operations do not support empty tensors")
        # For complex FFT, we can use DFT matrix multiplication as fallback
        # This is slower but avoids MKL entirely
        return _fft_fallback(input, n=n, dim=dim, norm=norm, inverse=False)

    try:
        return cast(torch.Tensor, torch.fft.fft(input, n=n, dim=dim, norm=norm))
    except RuntimeError as e:
        if "MKL FFT error" in str(e):
            warnings.warn(
                "MKL FFT error in fft. Using DFT matrix fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _fft_fallback(input, n=n, dim=dim, norm=norm, inverse=False)
        raise


def safe_ifft(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
) -> torch.Tensor:
    r"""1D inverse complex FFT with MKL fallback support.

    Computes the 1D inverse discrete Fourier transform, reconstructing
    signals from frequency domain representation.

    Parameters
    ----------
    input : torch.Tensor
        Complex tensor containing frequency coefficients.
    n : int | None, optional
        Output signal length. Input is truncated or zero-padded.
        If None, uses input size.
    dim : int, default=-1
        Dimension along which to compute inverse FFT.
    norm : str | None, optional
        Normalization mode matching the forward transform.

    Returns
    -------
    torch.Tensor
        Complex tensor containing reconstructed signal. For originally
        real signals, imaginary components represent numerical error.

    Notes
    -----
    The inverse transform is:

    $$
    x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{2\pi i kn/N}
    $$

    where the normalization factor depends on the norm parameter.
    The fallback uses DFT matrix multiplication with conjugate transpose
    for the inverse operation.
    """
    use_fallback = os.environ.get("SPECTRANS_DISABLE_MKL_FFT", "0") == "1"

    if use_fallback:
        # For complex IFFT, we can use DFT matrix multiplication as fallback
        return _fft_fallback(input, n=n, dim=dim, norm=norm, inverse=True)

    try:
        return cast(torch.Tensor, torch.fft.ifft(input, n=n, dim=dim, norm=norm))
    except RuntimeError as e:
        if "MKL FFT error" in str(e):
            warnings.warn(
                "MKL FFT error in ifft. Using DFT matrix fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _fft_fallback(input, n=n, dim=dim, norm=norm, inverse=True)
        raise


def _fft_fallback(
    input: torch.Tensor,
    n: int | None = None,
    dim: int = -1,
    norm: str | None = None,
    inverse: bool = False,
) -> torch.Tensor:
    r"""DFT matrix multiplication fallback.

    Implements discrete Fourier transform via explicit matrix multiplication,
    providing platform-independent fallback when MKL FFT fails.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor to transform.
    n : int | None, optional
        Transform length. Input is resized if needed.
    dim : int, default=-1
        Transform dimension.
    norm : str | None, optional
        Normalization mode.
    inverse : bool, default=False
        If True, compute inverse transform.

    Returns
    -------
    torch.Tensor
        Transformed tensor via DFT matrix multiplication.

    Notes
    -----
    Constructs the DFT matrix $\mathbf{W}$ with elements:

    $$
    W_{kn} = e^{\pm 2\pi i kn/N}
    $$

    where the sign depends on inverse. The transform is computed as
    $\mathbf{y} = \mathbf{W} \mathbf{x}$ with $O(N^2)$ complexity.

    This implementation trades computational efficiency for guaranteed
    correctness, useful when platform-specific FFT libraries fail.
    """
    # Get the size along the transform dimension
    n_transform = n if n is not None else input.shape[dim]

    # Pad or truncate if needed
    if n is not None and n != input.shape[dim]:
        # Create padded/truncated tensor
        shape = list(input.shape)
        shape[dim] = n
        padded = torch.zeros(shape, dtype=input.dtype, device=input.device)

        # Copy available data
        indices = [slice(None)] * len(input.shape)
        indices[dim] = slice(0, min(n, input.shape[dim]))
        padded[tuple(indices)] = input[tuple(indices)]
        input = padded

    # Move dimension to last for easier computation
    input = input.moveaxis(dim, -1)

    # Create DFT matrix
    device = input.device
    # Preserve dtype: float32->complex64, float64->complex128
    if input.is_complex():
        dtype = input.dtype
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    else:
        real_dtype = input.dtype
        dtype = torch.complex64 if input.dtype == torch.float32 else torch.complex128

    # Create frequency grid
    k = torch.arange(n_transform, device=device, dtype=real_dtype)
    n_grid = k.reshape(-1, 1)
    k_grid = k.reshape(1, -1)

    # DFT matrix using explicit cos/sin for better numerical stability
    angle = 2 * torch.pi * n_grid * k_grid / n_transform
    if inverse:
        # For inverse transform, use positive angle
        W = torch.complex(torch.cos(angle), torch.sin(angle))
    else:
        # For forward transform, use negative angle
        W = torch.complex(torch.cos(angle), -torch.sin(angle))

    # Apply normalization to DFT matrix
    if norm == "ortho":
        # Orthonormal transform: scale by 1/sqrt(n)
        W = W / torch.sqrt(torch.tensor(n_transform, dtype=real_dtype))
    elif norm == "forward" and not inverse:
        # Normalize forward transform by 1/n
        W = W / n_transform
    elif norm == "backward" and inverse:
        # Normalize inverse transform by 1/n
        W = W / n_transform
    elif norm is None and inverse:
        # Default: normalize inverse transform
        W = W / n_transform

    # Convert input to complex if needed
    if not input.is_complex():
        input = input.to(dtype)

    # Apply DFT via matrix multiplication
    result = torch.matmul(input, W.T)

    # Move dimension back
    result = result.moveaxis(-1, dim)

    return result


def safe_fft2(
    input: torch.Tensor,
    s: tuple[int, int] | None = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    r"""2D complex FFT with MKL fallback support.

    Computes the 2D discrete Fourier transform for complex or real input.
    Falls back to sequential 1D transforms when MKL errors occur.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor (real or complex).
    s : tuple[int, int] | None, optional
        Output sizes along transformed dimensions. If None, uses input sizes.
    dim : tuple[int, int], default=(-2, -1)
        Dimensions over which to compute the FFT.
    norm : str | None, optional
        Normalization mode.

    Returns
    -------
    torch.Tensor
        Complex tensor containing 2D FFT coefficients.
    """
    use_fallback = os.environ.get("SPECTRANS_DISABLE_MKL_FFT", "0") == "1"

    if use_fallback:
        # Check for empty transform dimensions
        if input.shape[dim[0]] == 0 or input.shape[dim[1]] == 0:
            raise RuntimeError("FFT operations do not support empty tensors")
        # Decompose into sequential 1D FFTs
        result = safe_fft(input, n=s[1] if s else None, dim=dim[1], norm=norm)
        result = safe_fft(result, n=s[0] if s else None, dim=dim[0], norm=norm)
        return result

    try:
        return cast(torch.Tensor, torch.fft.fft2(input, s=s, dim=dim, norm=norm))
    except RuntimeError as e:
        if "MKL FFT error" in str(e):
            warnings.warn(
                "MKL FFT error in fft2. Using sequential 1D FFT fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Fallback: Use sequential 1D FFTs
            result = safe_fft(input, n=s[1] if s else None, dim=dim[1], norm=norm)
            result = safe_fft(result, n=s[0] if s else None, dim=dim[0], norm=norm)
            return result
        raise


def safe_ifft2(
    input: torch.Tensor,
    s: tuple[int, int] | None = None,
    dim: tuple[int, int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    r"""2D inverse complex FFT with MKL fallback support.

    Computes the 2D inverse discrete Fourier transform.
    Falls back to sequential 1D transforms when MKL errors occur.

    Parameters
    ----------
    input : torch.Tensor
        Complex tensor containing FFT coefficients.
    s : tuple[int, int] | None, optional
        Output spatial sizes. If None, uses input sizes.
    dim : tuple[int, int], default=(-2, -1)
        Dimensions over which to compute the inverse FFT.
    norm : str | None, optional
        Normalization mode matching forward transform.

    Returns
    -------
    torch.Tensor
        Complex tensor containing spatial domain result.
    """
    use_fallback = os.environ.get("SPECTRANS_DISABLE_MKL_FFT", "0") == "1"

    if use_fallback:
        # Decompose into sequential 1D inverse FFTs (reverse order of forward)
        result = safe_ifft(input, n=s[0] if s else None, dim=dim[0], norm=norm)
        result = safe_ifft(result, n=s[1] if s else None, dim=dim[1], norm=norm)
        return result

    try:
        return cast(torch.Tensor, torch.fft.ifft2(input, s=s, dim=dim, norm=norm))
    except RuntimeError as e:
        if "MKL FFT error" in str(e):
            warnings.warn(
                "MKL FFT error in ifft2. Using sequential 1D IFFT fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Fallback: Use sequential 1D inverse FFTs
            result = safe_ifft(input, n=s[0] if s else None, dim=dim[0], norm=norm)
            result = safe_ifft(result, n=s[1] if s else None, dim=dim[1], norm=norm)
            return result
        raise


def safe_rfftn(
    input: torch.Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
    norm: str | None = None,
) -> torch.Tensor:
    r"""N-dimensional real FFT with MKL fallback support.

    Computes the N-dimensional discrete Fourier transform of real input.
    Falls back to sequential transforms when MKL errors occur.

    Parameters
    ----------
    input : torch.Tensor
        Real-valued input tensor.
    s : tuple[int, ...] | None, optional
        Output sizes along transformed dimensions.
    dim : tuple[int, ...] | None, optional
        Dimensions to transform. If None, transforms all dimensions.
    norm : str | None, optional
        Normalization mode.

    Returns
    -------
    torch.Tensor
        Complex tensor with positive frequencies only in the last dimension.
    """
    use_fallback = os.environ.get("SPECTRANS_DISABLE_MKL_FFT", "0") == "1"

    if use_fallback:
        if dim is None:
            dim = tuple(range(input.ndim))
        # Check for empty dimensions
        for d in dim:
            if input.shape[d] == 0:
                raise RuntimeError("FFT operations do not support empty tensors")

        # For n-dimensional RFFT, we need to handle this differently
        # RFFT is only applied to real input, so we apply RFFT first, then FFT to other dims
        # This matches what torch.fft.rfftn does internally

        # First apply RFFT to the last dimension (real -> complex with half spectrum)
        n_last = s[-1] if s else None
        result = safe_rfft(input, n=n_last, dim=dim[-1], norm=norm)

        # Then apply FFT to the remaining dimensions (complex -> complex)
        for i, d in enumerate(dim[:-1]):
            n = s[i] if s else None
            result = safe_fft(result, n=n, dim=d, norm=norm)

        return result

    try:
        return cast(torch.Tensor, torch.fft.rfftn(input, s=s, dim=dim, norm=norm))
    except RuntimeError as e:
        if "MKL FFT error" in str(e):
            warnings.warn(
                "MKL FFT error in rfftn. Using sequential FFT fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            if dim is None:
                dim = tuple(range(input.ndim))
            # Fallback implementation
            result = input
            for i, d in enumerate(dim[:-1]):
                n = s[i] if s else None
                result = safe_fft(result, n=n, dim=d, norm=norm)
            # Apply RFFT to last dimension
            n_last = s[-1] if s else None
            result = safe_rfft(result, n=n_last, dim=dim[-1], norm=norm)
            return result
        raise


def safe_irfftn(
    input: torch.Tensor,
    s: tuple[int, ...] | None = None,
    dim: tuple[int, ...] | None = None,
    norm: str | None = None,
) -> torch.Tensor:
    r"""N-dimensional inverse real FFT with MKL fallback support.

    Computes the N-dimensional inverse discrete Fourier transform for real output.
    Falls back to sequential transforms when MKL errors occur.

    Parameters
    ----------
    input : torch.Tensor
        Complex tensor with Hermitian symmetry.
    s : tuple[int, ...] | None, optional
        Output spatial sizes.
    dim : tuple[int, ...] | None, optional
        Dimensions to transform. If None, transforms all dimensions.
    norm : str | None, optional
        Normalization mode.

    Returns
    -------
    torch.Tensor
        Real-valued spatial domain tensor.
    """
    use_fallback = os.environ.get("SPECTRANS_DISABLE_MKL_FFT", "0") == "1"

    if use_fallback:
        if dim is None:
            dim = tuple(range(input.ndim))
        # For n-dimensional IRFFT, we reverse the operations of RFFTN
        # First apply IFFT to all but the last dimension (complex -> complex)
        # Then apply IRFFT to the last dimension (complex half spectrum -> real)

        result = input
        # Apply IFFT to all dimensions except the last (in reverse order for consistency)
        for i in range(len(dim) - 2, -1, -1):
            n = s[i] if s else None
            result = safe_ifft(result, n=n, dim=dim[i], norm=norm)

        # Finally apply IRFFT to the last dimension
        n_last = s[-1] if s else None
        result = safe_irfft(result, n=n_last, dim=dim[-1], norm=norm)

        return result

    try:
        return cast(torch.Tensor, torch.fft.irfftn(input, s=s, dim=dim, norm=norm))
    except RuntimeError as e:
        if "MKL FFT error" in str(e):
            warnings.warn(
                "MKL FFT error in irfftn. Using sequential IFFT fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            if dim is None:
                dim = tuple(range(input.ndim))
            # Fallback implementation
            n_last = s[-1] if s else None
            result = safe_irfft(input, n=n_last, dim=dim[-1], norm=norm)
            for i in range(len(dim) - 2, -1, -1):
                n = s[i] if s else None
                result = safe_ifft(result, n=n, dim=dim[i], norm=norm)
            return result.real
        raise
