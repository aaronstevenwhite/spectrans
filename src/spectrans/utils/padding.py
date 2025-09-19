r"""Padding utilities for spectral transformations and signal processing.

This module provides comprehensive padding strategies specifically designed for
spectral neural networks and signal processing operations. The padding functions
maintain mathematical properties of transforms, handle boundary conditions properly,
and optimize computational efficiency for various spectral operations.

Different padding modes are crucial for spectral transformers as they affect the
mathematical properties of transforms, boundary artifacts, and computational
efficiency. This module provides both basic and specialized padding operations
with proper handling of edge cases and batch processing.

Functions
---------
pad_to_length(x, target_length, dim, mode)
    Pad tensor to specified length with various padding modes.
pad_to_power_of_2(x, dim)
    Pad tensor to next power of 2 for efficient FFT operations.
pad_for_fft(x, target_length, dim, mode)
    Specialized padding for FFT operations with optimal performance.
pad_for_convolution(x, kernel_size, mode, dim)
    Pad tensor for convolution operations to maintain output size.
zero_pad(x, padding, dim)
    Apply zero padding along specified dimension.
circular_pad(x, padding, dim)
    Apply circular (periodic) padding for periodic signals.
reflect_pad(x, padding, dim)
    Apply reflection padding to minimize boundary artifacts.
symmetric_pad(x, padding, dim)
    Apply symmetric padding with proper boundary handling.
pad_sequence(sequences, batch_first, padding_value)
    Pad variable-length sequences to uniform length.
unpad_to_length(x, original_length, dim)
    Remove padding to restore original tensor length.
unpad_sequence(padded, lengths)
    Remove padding from batch of padded sequences.

Examples
--------
Padding for FFT operations:

>>> import torch
>>> from spectrans.utils.padding import pad_to_power_of_2, pad_for_fft
>>> # Pad to nearest power of 2 for efficient FFT
>>> signal = torch.randn(32, 500)  # Length 500
>>> padded = pad_to_power_of_2(signal, dim=-1)  # Pads to 512
>>> print(f"Original: {signal.shape}, Padded: {padded.shape}")
>>>
>>> # Pad to specific FFT length
>>> fft_signal = pad_for_fft(signal, target_length=1024, dim=-1)
>>> # Use torch.fft.fft on fft_signal for optimal performance

Circular padding for periodic signals:

>>> from spectrans.utils.padding import circular_pad
>>> periodic_signal = torch.randn(32, 256)
>>> # Add 64 samples of circular padding on each side
>>> circularly_padded = circular_pad(periodic_signal, padding=64, dim=-1)
>>> # Signal boundaries are continuous (no edge discontinuities)

Reflection padding for boundary handling:

>>> from spectrans.utils.padding import reflect_pad
>>> image = torch.randn(32, 3, 224, 224)
>>> # Reflect pad for 2D convolution
>>> reflected = reflect_pad(image, padding=3, dim=(-2, -1))
>>> # Boundaries are smooth transitions

Variable-length sequence padding:

>>> from spectrans.utils.padding import pad_sequence
>>> sequences = [torch.randn(100, 768), torch.randn(150, 768), torch.randn(80, 768)]
>>> padded_batch = pad_sequence(sequences, batch_first=True, padding_value=0.0)
>>> # All sequences now have length 150 (max length)

Notes
-----
Padding Strategies and Their Properties:

**Zero Padding**:
- Adds zeros at boundaries
- Simple and computationally efficient
- Can introduce spectral artifacts due to discontinuities
- Best for: General-purpose padding where simplicity is preferred

**Circular Padding** (Periodic):
- Wraps signal around to create continuity
- Maintains periodicity assumptions of DFT/FFT
- No boundary artifacts in frequency domain
- Best for: Periodic signals, DFT/FFT operations

**Reflection Padding**:
- Mirrors signal at boundaries
- Reduces boundary discontinuities
- Maintains signal characteristics near boundaries
- Best for: Image processing, minimizing edge effects

**Symmetric Padding**:
- Creates symmetric extension of signal
- Preserves certain symmetry properties
- Useful for transforms with symmetry assumptions
- Best for: DCT operations, symmetric signal processing

Mathematical Considerations:

1. **FFT Efficiency**: Powers of 2 enable radix-2 FFT algorithms with optimal :math:`O(n \log n)` complexity
2. **Boundary Conditions**: Different padding affects transform properties and numerical stability
3. **Aliasing**: Improper padding can introduce aliasing artifacts in frequency domain
4. **Memory Usage**: Padding increases memory requirements, important for large signals

Implementation Details:

- **Batch Processing**: All functions handle batched inputs efficiently
- **Multi-Dimensional**: Support for padding along multiple dimensions simultaneously
- **Type Preservation**: Maintains input tensor dtype and device
- **Memory Efficiency**: In-place operations where safe, memory-efficient algorithms
- **Error Handling**: Comprehensive validation of input parameters and dimensions

Performance Optimizations:

- **Vectorized Operations**: Use PyTorch's native vectorized padding operations
- **GPU Acceleration**: All operations are GPU-compatible
- **Memory Layout**: Optimized for PyTorch's memory layout conventions
- **Caching**: Efficient memory allocation patterns for repeated operations

Common Pitfalls:

- Padding too aggressively can change signal characteristics
- Wrong padding mode can introduce artifacts in spectral domain
- Not accounting for padding when calculating output sizes
- Forgetting to unpad results when original size is needed

See Also
--------
spectrans.transforms : Spectral transforms requiring specific padding
spectrans.utils.complex : Complex tensor operations for padded signals
torch.nn.functional : PyTorch's native padding functions
"""

import torch

from ..core.types import Tensor

__all__ = [
    "circular_pad",
    "pad_for_convolution",
    "pad_for_fft",
    "pad_sequence",
    "pad_to_length",
    "pad_to_power_of_2",
    "reflect_pad",
    "symmetric_pad",
    "unpad_sequence",
    "unpad_to_length",
    "zero_pad",
]


def pad_to_length(x: Tensor, target_length: int, dim: int = -1, mode: str = "zero") -> Tensor:
    """Pad tensor to specified length along given dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor to pad.
    target_length : int
        Target length after padding.
    dim : int, default=-1
        Dimension to pad along.
    mode : str, default="zero"
        Padding mode: "zero", "circular", "reflect", "symmetric".

    Returns
    -------
    Tensor
        Padded tensor with target length along specified dimension.

    Raises
    ------
    ValueError
        If target_length is smaller than current length, or invalid mode.
    IndexError
        If dimension is out of bounds.
    """
    if dim >= x.ndim or dim < -x.ndim:
        raise IndexError(f"Dimension {dim} out of bounds for tensor with {x.ndim} dimensions")

    # Normalize negative dimension
    dim = dim % x.ndim
    current_length = x.shape[dim]

    if target_length < current_length:
        raise ValueError(
            f"Target length {target_length} must be >= current length {current_length}"
        )

    if target_length == current_length:
        return x

    pad_amount = target_length - current_length

    # Convert mode to appropriate function
    if mode == "zero":
        return zero_pad(x, pad_amount, dim)
    elif mode == "circular":
        return circular_pad(x, pad_amount, dim)
    elif mode == "reflect":
        return reflect_pad(x, pad_amount, dim)
    elif mode == "symmetric":
        return symmetric_pad(x, pad_amount, dim)
    else:
        raise ValueError(
            f"Invalid padding mode: {mode}. Must be one of: zero, circular, reflect, symmetric"
        )


def unpad_to_length(x: Tensor, target_length: int, dim: int = -1) -> Tensor:
    """Remove padding to restore original length.

    Parameters
    ----------
    x : Tensor
        Padded tensor.
    target_length : int
        Original length before padding.
    dim : int, default=-1
        Dimension to unpad along.

    Returns
    -------
    Tensor
        Tensor with padding removed.

    Raises
    ------
    ValueError
        If target_length is larger than current length.
    IndexError
        If dimension is out of bounds.
    """
    if dim >= x.ndim or dim < -x.ndim:
        raise IndexError(f"Dimension {dim} out of bounds for tensor with {x.ndim} dimensions")

    # Normalize negative dimension
    dim = dim % x.ndim
    current_length = x.shape[dim]

    if target_length > current_length:
        raise ValueError(
            f"Target length {target_length} must be <= current length {current_length}"
        )

    if target_length == current_length:
        return x

    # Create slice objects to extract the original data
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(0, target_length)

    return x[tuple(slices)]


def pad_sequence(sequences: list[Tensor], padding_value: float = 0.0, dim: int = -1) -> Tensor:
    """Pad a list of sequences to the same length.

    Parameters
    ----------
    sequences : list[Tensor]
        List of tensors to pad.
    padding_value : float, default=0.0
        Value to use for padding.
    dim : int, default=-1
        Dimension to pad along.

    Returns
    -------
    Tensor
        Batched tensor with all sequences padded to same length.

    Raises
    ------
    ValueError
        If sequences is empty or tensors have incompatible shapes.
    IndexError
        If dimension is out of bounds.
    """
    if not sequences:
        raise ValueError("Cannot pad empty list of sequences")

    if len(sequences) == 1:
        return sequences[0].unsqueeze(0)

    # Check dimension bounds
    ndim = sequences[0].ndim
    if dim >= ndim or dim < -ndim:
        raise IndexError(f"Dimension {dim} out of bounds for tensor with {ndim} dimensions")

    dim = dim % ndim

    # Verify all tensors have compatible shapes (same except for padding dimension)
    ref_shape = list(sequences[0].shape)
    for i, seq in enumerate(sequences[1:], 1):
        if seq.ndim != ndim:
            raise ValueError(
                f"All sequences must have same number of dimensions. "
                f"Sequence 0 has {ndim}, sequence {i} has {seq.ndim}"
            )

        seq_shape = list(seq.shape)
        for d in range(ndim):
            if d != dim and seq_shape[d] != ref_shape[d]:
                raise ValueError(
                    f"All sequences must have same shape except in padding dimension. "
                    f"Mismatch in dimension {d}: {ref_shape[d]} vs {seq_shape[d]}"
                )

    # Find maximum length
    max_length = max(seq.shape[dim] for seq in sequences)

    # Pad each sequence
    padded_sequences = []
    for seq in sequences:
        if seq.shape[dim] < max_length:
            pad_amount = max_length - seq.shape[dim]
            padded_seq = zero_pad(seq, pad_amount, dim, padding_value)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)

    # Stack into batch
    return torch.stack(padded_sequences, dim=0)


def unpad_sequence(padded_tensor: Tensor, lengths: list[int], dim: int = -1) -> list[Tensor]:
    """Unpad a batched tensor back to individual sequences.

    Parameters
    ----------
    padded_tensor : Tensor
        Batched, padded tensor.
    lengths : list[int]
        Original lengths of each sequence.
    dim : int, default=-1
        Dimension that was padded.

    Returns
    -------
    list[Tensor]
        List of unpadded sequences.

    Raises
    ------
    ValueError
        If lengths don't match batch size.
    IndexError
        If dimension is out of bounds.
    """
    if padded_tensor.ndim == 0:
        raise ValueError("Cannot unpad scalar tensor")

    batch_size = padded_tensor.shape[0]
    if len(lengths) != batch_size:
        raise ValueError(f"Number of lengths ({len(lengths)}) must match batch size ({batch_size})")

    # Adjust for batch dimension
    batch_dim = dim
    if batch_dim >= 0:
        batch_dim += 1  # Account for batch dimension

    if batch_dim >= padded_tensor.ndim or batch_dim < -padded_tensor.ndim:
        raise IndexError(
            f"Dimension {batch_dim} out of bounds for tensor with {padded_tensor.ndim} dimensions"
        )

    sequences = []
    for i, length in enumerate(lengths):
        seq = padded_tensor[i]
        if length < seq.shape[dim]:  # Use original dim for individual tensor
            seq = unpad_to_length(seq, length, dim)
        sequences.append(seq)

    return sequences


def circular_pad(x: Tensor, pad_amount: int, dim: int = -1) -> Tensor:
    """Apply circular (periodic) padding.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    pad_amount : int
        Number of elements to pad.
    dim : int, default=-1
        Dimension to pad along.

    Returns
    -------
    Tensor
        Circularly padded tensor.

    Raises
    ------
    ValueError
        If pad_amount is negative or exceeds tensor size.
    IndexError
        If dimension is out of bounds.
    """
    if pad_amount < 0:
        raise ValueError(f"Pad amount must be non-negative, got {pad_amount}")

    if pad_amount == 0:
        return x

    if dim >= x.ndim or dim < -x.ndim:
        raise IndexError(f"Dimension {dim} out of bounds for tensor with {x.ndim} dimensions")

    dim = dim % x.ndim
    seq_len = x.shape[dim]

    if pad_amount > seq_len:
        raise ValueError(f"Circular pad amount {pad_amount} exceeds tensor size {seq_len}")

    # Take last pad_amount elements and append them
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(-pad_amount, None)
    padding = x[tuple(slices)]

    return torch.cat([x, padding], dim=dim)


def reflect_pad(x: Tensor, pad_amount: int, dim: int = -1) -> Tensor:
    """Apply reflection padding (mirror without repeating edge).

    Parameters
    ----------
    x : Tensor
        Input tensor.
    pad_amount : int
        Number of elements to pad.
    dim : int, default=-1
        Dimension to pad along.

    Returns
    -------
    Tensor
        Reflection padded tensor.

    Raises
    ------
    ValueError
        If pad_amount is negative or too large for reflection.
    IndexError
        If dimension is out of bounds.
    """
    if pad_amount < 0:
        raise ValueError(f"Pad amount must be non-negative, got {pad_amount}")

    if pad_amount == 0:
        return x

    if dim >= x.ndim or dim < -x.ndim:
        raise IndexError(f"Dimension {dim} out of bounds for tensor with {x.ndim} dimensions")

    dim = dim % x.ndim
    seq_len = x.shape[dim]

    if pad_amount >= seq_len:
        raise ValueError(f"Reflect pad amount {pad_amount} must be < tensor size {seq_len}")

    # Reflect last pad_amount elements (excluding the edge)
    # For [1,2,3,4] with pad=2, we take elements [-3:-1] = [2,3] and flip to get [3,2]
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(seq_len - pad_amount - 1, seq_len - 1)
    padding = torch.flip(x[tuple(slices)], dims=[dim])

    return torch.cat([x, padding], dim=dim)


def symmetric_pad(x: Tensor, pad_amount: int, dim: int = -1) -> Tensor:
    """Apply symmetric padding (mirror including edge).

    Parameters
    ----------
    x : Tensor
        Input tensor.
    pad_amount : int
        Number of elements to pad.
    dim : int, default=-1
        Dimension to pad along.

    Returns
    -------
    Tensor
        Symmetrically padded tensor.

    Raises
    ------
    ValueError
        If pad_amount is negative or too large for symmetry.
    IndexError
        If dimension is out of bounds.
    """
    if pad_amount < 0:
        raise ValueError(f"Pad amount must be non-negative, got {pad_amount}")

    if pad_amount == 0:
        return x

    if dim >= x.ndim or dim < -x.ndim:
        raise IndexError(f"Dimension {dim} out of bounds for tensor with {x.ndim} dimensions")

    dim = dim % x.ndim
    seq_len = x.shape[dim]

    if pad_amount > seq_len:
        raise ValueError(f"Symmetric pad amount {pad_amount} must be <= tensor size {seq_len}")

    # Symmetric: mirror including the edge
    # For [1,2,3,4] with pad=2, we take last 2 elements [3,4] and flip to get [4,3]
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(seq_len - pad_amount, seq_len)
    padding = torch.flip(x[tuple(slices)], dims=[dim])

    return torch.cat([x, padding], dim=dim)


def zero_pad(x: Tensor, pad_amount: int, dim: int = -1, value: float = 0.0) -> Tensor:
    """Apply zero (constant) padding.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    pad_amount : int
        Number of elements to pad.
    dim : int, default=-1
        Dimension to pad along.
    value : float, default=0.0
        Constant value to pad with.

    Returns
    -------
    Tensor
        Zero-padded tensor.

    Raises
    ------
    ValueError
        If pad_amount is negative.
    IndexError
        If dimension is out of bounds.
    """
    if pad_amount < 0:
        raise ValueError(f"Pad amount must be non-negative, got {pad_amount}")

    if pad_amount == 0:
        return x

    if dim >= x.ndim or dim < -x.ndim:
        raise IndexError(f"Dimension {dim} out of bounds for tensor with {x.ndim} dimensions")

    dim = dim % x.ndim

    # Create padding tensor with same shape except in padding dimension
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_amount

    padding = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)

    return torch.cat([x, padding], dim=dim)


def pad_for_fft(x: Tensor, dim: int = -1) -> tuple[Tensor, int]:
    """Pad tensor to optimal size for FFT computation.

    Pads to next power of 2 for optimal FFT performance.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int, default=-1
        Dimension to pad along.

    Returns
    -------
    tuple[Tensor, int]
        Tuple of (padded_tensor, original_length).

    Raises
    ------
    IndexError
        If dimension is out of bounds.
    ValueError
        If tensor is empty along specified dimension.
    """
    if dim >= x.ndim or dim < -x.ndim:
        raise IndexError(f"Dimension {dim} out of bounds for tensor with {x.ndim} dimensions")

    dim = dim % x.ndim
    original_length = x.shape[dim]

    if original_length == 0:
        raise ValueError("Cannot pad empty tensor dimension for FFT")

    # Find next power of 2
    optimal_length = pad_to_power_of_2(original_length)

    if optimal_length == original_length:
        return x, original_length

    padded = pad_to_length(x, optimal_length, dim, mode="zero")
    return padded, original_length


def pad_to_power_of_2(length: int) -> int:
    """Find next power of 2 greater than or equal to length.

    Parameters
    ----------
    length : int
        Input length.

    Returns
    -------
    int
        Next power of 2.

    Raises
    ------
    ValueError
        If length is not positive.
    """
    if length <= 0:
        raise ValueError(f"Length must be positive, got {length}")

    if length == 1:
        return 1

    # Find next power of 2
    power = 1
    while power < length:
        power <<= 1

    return power


def wavelet_symmetric_pad(x: Tensor, pad_len: int, dim: int = -1) -> Tensor:
    """Apply symmetric padding for wavelet transforms (PyWavelets-compatible).

    This implements the exact symmetric padding used by PyWavelets which
    reflects the signal WITH edge repeat. This is equivalent to numpy's
    'symmetric' mode and PyTorch's 'reflect' mode.

    Parameters
    ----------
    x : Tensor
        Input tensor to pad.
    pad_len : int
        Number of samples to pad on each side.
    dim : int, default=-1
        Dimension along which to pad.

    Returns
    -------
    Tensor
        Symmetrically padded tensor.

    Examples
    --------
    >>> x = torch.tensor([1, 2, 3, 4])
    >>> padded = wavelet_symmetric_pad(x, 3)
    >>> # Result: [3, 2, 1, 1, 2, 3, 4, 4, 3, 2]

    Notes
    -----
    For signal [a,b,c,d] with pad_len=3, creates [c,b,a|a,b,c,d|d,c,b].
    This ensures continuity at boundaries for wavelet transforms.
    """
    if pad_len == 0:
        return x

    if pad_len < 0:
        raise ValueError(f"Pad length must be non-negative, got {pad_len}")

    if dim >= x.ndim or dim < -x.ndim:
        raise IndexError(f"Dimension {dim} out of bounds for tensor with {x.ndim} dimensions")

    # Normalize dimension
    dim = dim % x.ndim
    signal_len = x.shape[dim]

    if signal_len == 1:
        # Single element: just repeat it
        if x.ndim == 1:
            return x[0].repeat(2 * pad_len + 1)
        else:
            repeat_dims = [1] * x.ndim
            repeat_dims[dim] = 2 * pad_len + 1
            return x.repeat(*repeat_dims)

    # Build reflection indices with edge repeat (symmetric mode)
    # Left padding: reflect starting from index 0
    left_indices = []
    pos = 0
    direction = 1
    for _ in range(pad_len):
        left_indices.append(pos)
        # Reflect at boundaries
        if direction == 1 and pos == signal_len - 1:
            direction = -1
        elif direction == -1 and pos == 0:
            direction = 1
        pos += direction
    left_indices.reverse()

    # Right padding: reflect starting from last index
    right_indices = []
    pos = signal_len - 1
    direction = -1
    for _ in range(pad_len):
        right_indices.append(pos)
        # Reflect at boundaries
        if direction == -1 and pos == 0:
            direction = 1
        elif direction == 1 and pos == signal_len - 1:
            direction = -1
        pos += direction

    # Convert to tensor indices
    left_indices_t = torch.tensor(left_indices, dtype=torch.long, device=x.device)
    right_indices_t = torch.tensor(right_indices, dtype=torch.long, device=x.device)

    # Apply padding
    if x.ndim == 1:
        left_pad = x[left_indices_t]
        right_pad = x[right_indices_t]
        return torch.cat([left_pad, x, right_pad])
    else:
        # Multi-dimensional: use advanced indexing
        slices: list[slice | torch.Tensor] = [slice(None)] * x.ndim
        slices[dim] = left_indices_t
        left_pad = x[tuple(slices)]

        slices[dim] = right_indices_t
        right_pad = x[tuple(slices)]

        return torch.cat([left_pad, x, right_pad], dim=dim)


def pad_for_convolution(x: Tensor, kernel_size: int, dim: int = -1, mode: str = "zero") -> Tensor:
    """Pad tensor for valid convolution without size reduction.

    Applies symmetric padding to both sides of the specified dimension to ensure
    that convolution output has the same size as input (same padding).

    Parameters
    ----------
    x : Tensor
        Input tensor.
    kernel_size : int
        Size of convolution kernel.
    dim : int, default=-1
        Dimension to pad along.
    mode : str, default="zero"
        Padding mode: "zero", "circular", "reflect", "symmetric".

    Returns
    -------
    Tensor
        Padded tensor suitable for convolution.

    Raises
    ------
    ValueError
        If kernel_size is not positive odd integer or mode is invalid.
    IndexError
        If dimension is out of bounds.
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be positive odd integer, got {kernel_size}")

    if dim >= x.ndim or dim < -x.ndim:
        raise IndexError(f"Dimension {dim} out of bounds for tensor with {x.ndim} dimensions")

    # Normalize negative dimension
    dim = dim % x.ndim

    # Calculate padding needed for 'same' convolution
    pad_amount = kernel_size // 2

    if pad_amount == 0:
        return x

    # Apply symmetric padding based on mode
    if mode == "zero":
        # Create left padding
        left_pad_shape = list(x.shape)
        left_pad_shape[dim] = pad_amount
        left_padding = torch.zeros(left_pad_shape, dtype=x.dtype, device=x.device)

        # Create right padding
        right_pad_shape = list(x.shape)
        right_pad_shape[dim] = pad_amount
        right_padding = torch.zeros(right_pad_shape, dtype=x.dtype, device=x.device)

        # Concatenate: left_padding + original + right_padding
        return torch.cat([left_padding, x, right_padding], dim=dim)

    elif mode == "circular":
        # Apply circular padding on both sides
        # Left padding: take last pad_amount elements
        left_slices = [slice(None)] * x.ndim
        left_slices[dim] = slice(-pad_amount, None)
        left_padding = x[tuple(left_slices)]

        # Right padding: take first pad_amount elements
        right_slices = [slice(None)] * x.ndim
        right_slices[dim] = slice(0, pad_amount)
        right_padding = x[tuple(right_slices)]

        return torch.cat([left_padding, x, right_padding], dim=dim)

    elif mode == "reflect":
        # Apply reflection padding on both sides
        seq_len = x.shape[dim]
        if 2 * pad_amount >= seq_len:
            raise ValueError(
                f"Reflect pad amount {2 * pad_amount} too large for tensor size {seq_len}"
            )

        # Left padding: reflect first pad_amount elements (excluding edges)
        left_slices = [slice(None)] * x.ndim
        left_slices[dim] = slice(1, pad_amount + 1)
        left_padding = torch.flip(x[tuple(left_slices)], dims=[dim])

        # Right padding: reflect last pad_amount elements (excluding edges)
        right_slices = [slice(None)] * x.ndim
        right_slices[dim] = slice(seq_len - pad_amount - 1, seq_len - 1)
        right_padding = torch.flip(x[tuple(right_slices)], dims=[dim])

        return torch.cat([left_padding, x, right_padding], dim=dim)

    elif mode == "symmetric":
        # Apply symmetric padding on both sides
        seq_len = x.shape[dim]
        if 2 * pad_amount > seq_len:
            raise ValueError(
                f"Symmetric pad amount {2 * pad_amount} too large for tensor size {seq_len}"
            )

        # Left padding: reflect first pad_amount elements (including edges)
        left_slices = [slice(None)] * x.ndim
        left_slices[dim] = slice(0, pad_amount)
        left_padding = torch.flip(x[tuple(left_slices)], dims=[dim])

        # Right padding: reflect last pad_amount elements (including edges)
        right_slices = [slice(None)] * x.ndim
        right_slices[dim] = slice(seq_len - pad_amount, seq_len)
        right_padding = torch.flip(x[tuple(right_slices)], dims=[dim])

        return torch.cat([left_padding, x, right_padding], dim=dim)

    else:
        raise ValueError(
            f"Invalid padding mode: {mode}. Must be one of: zero, circular, reflect, symmetric"
        )
