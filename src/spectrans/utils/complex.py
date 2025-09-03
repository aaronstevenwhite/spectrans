"""Complex tensor operations for spectral transformations.

This module provides specialized complex number operations for PyTorch tensors,
designed for spectral transformer implementations. The functions provide consistent
error handling, mathematical safety for edge cases, and uniform interfaces for
complex tensor operations in spectral transforms.

All functions support batch operations and proper broadcasting.
"""

import torch

from ..core.types import Tensor

__all__ = [
    "complex_conjugate",
    "complex_divide",
    "complex_dropout",
    "complex_exp",
    "complex_log",
    "complex_modulus",
    "complex_multiply",
    "complex_phase",
    "complex_polar",
    "complex_relu",
    "make_complex",
    "split_complex",
]


def complex_multiply(a: Tensor, b: Tensor) -> Tensor:
    """Multiply two complex tensors element-wise.

    Performs (a_real + i*a_imag) * (b_real + i*b_imag) efficiently.
    Supports broadcasting according to PyTorch broadcasting rules.


    Parameters
    ----------
    a : Tensor
        First complex tensor.
    b : Tensor
        Second complex tensor.

    Returns
    -------
    Tensor
        Complex product tensor.

    Raises
    ------
    TypeError
        If inputs are not complex tensors.
    RuntimeError
        If tensors cannot be broadcast together.
    """
    if not a.is_complex():
        raise TypeError(f"First argument must be complex tensor, got {a.dtype}")
    if not b.is_complex():
        raise TypeError(f"Second argument must be complex tensor, got {b.dtype}")

    try:
        return torch.mul(a, b)
    except RuntimeError as e:
        raise RuntimeError(f"Cannot broadcast tensors with shapes {a.shape} and {b.shape}") from e


def complex_conjugate(x: Tensor) -> Tensor:
    """Compute complex conjugate of input tensor.

    Essential operation for spectral transforms, particularly for ensuring
    Hermitian symmetry in frequency domain operations.

    Parameters
    ----------
    x : Tensor
        Input complex tensor.

    Returns
    -------
    Tensor
        Complex conjugate tensor.

    Raises
    ------
    TypeError
        If input is not a complex tensor.
    """
    if not x.is_complex():
        raise TypeError(f"Input must be complex tensor, got {x.dtype}")

    return torch.conj(x)


def complex_modulus(x: Tensor) -> Tensor:
    """Compute magnitude (absolute value) of complex tensor.

    Critical for spectral analysis where magnitude represents signal energy.

    Parameters
    ----------
    x : Tensor
        Input complex tensor.

    Returns
    -------
    Tensor
        Real tensor containing magnitudes.

    Raises
    ------
    TypeError
        If input is not a complex tensor.
    """
    if not x.is_complex():
        raise TypeError(f"Input must be complex tensor, got {x.dtype}")

    return torch.abs(x)


def complex_phase(x: Tensor) -> Tensor:
    """Compute phase angle of complex tensor.

    Phase information is crucial for spectral transformations and filter design.

    Parameters
    ----------
    x : Tensor
        Input complex tensor.

    Returns
    -------
    Tensor
        Real tensor containing phase angles in radians [-π, π].

    Raises
    ------
    TypeError
        If input is not a complex tensor.
    """
    if not x.is_complex():
        raise TypeError(f"Input must be complex tensor, got {x.dtype}")

    return torch.angle(x)


def complex_polar(magnitude: Tensor, phase: Tensor) -> Tensor:
    """Construct complex tensor from magnitude and phase.

    Fundamental for spectral operations where separate magnitude and phase
    processing is required. Includes validation for non-negative magnitudes.

    Parameters
    ----------
    magnitude : Tensor
        Real tensor containing magnitudes (must be non-negative).
    phase : Tensor
        Real tensor containing phase angles in radians.

    Returns
    -------
    Tensor
        Complex tensor constructed from polar coordinates.

    Raises
    ------
    TypeError
        If inputs are not real tensors.
    ValueError
        If magnitude contains negative values.
    RuntimeError
        If tensors cannot be broadcast together.
    """
    if magnitude.is_complex():
        raise TypeError(f"Magnitude must be real tensor, got {magnitude.dtype}")
    if phase.is_complex():
        raise TypeError(f"Phase must be real tensor, got {phase.dtype}")

    if torch.any(magnitude < 0):
        raise ValueError("Magnitude must be non-negative")

    try:
        return torch.polar(magnitude, phase)
    except RuntimeError as e:
        raise RuntimeError(f"Cannot broadcast tensors with shapes {magnitude.shape} and {phase.shape}") from e


def complex_exp(x: Tensor) -> Tensor:
    """Compute complex exponential e^x.

    Core operation for Fourier transforms and oscillatory functions.
    Accepts both real and complex inputs for flexibility.

    Parameters
    ----------
    x : Tensor
        Input tensor (can be real or complex).

    Returns
    -------
    Tensor
        Complex exponential tensor.
    """
    return torch.exp(x)


def complex_log(x: Tensor) -> Tensor:
    """Compute complex natural logarithm.

    Used in spectral domain operations and inverse transforms.
    Includes safety check for zeros where logarithm is undefined.

    Parameters
    ----------
    x : Tensor
        Input complex tensor.

    Returns
    -------
    Tensor
        Complex logarithm tensor.

    Raises
    ------
    TypeError
        If input is not a complex tensor.
    ValueError
        If input contains zeros (logarithm undefined).
    """
    if not x.is_complex():
        raise TypeError(f"Input must be complex tensor, got {x.dtype}")

    # Check for zeros where log is undefined
    if torch.any(torch.abs(x) == 0):
        raise ValueError("Logarithm undefined for zero values")

    return torch.log(x)


def complex_divide(a: Tensor, b: Tensor) -> Tensor:
    """Divide two complex tensors element-wise.

    Essential for spectral filtering operations. Includes safety checks
    for division by zero, which can occur in spectral nulls.

    Parameters
    ----------
    a : Tensor
        Numerator complex tensor.
    b : Tensor
        Denominator complex tensor.

    Returns
    -------
    Tensor
        Complex division result.

    Raises
    ------
    TypeError
        If inputs are not complex tensors.
    ValueError
        If denominator contains zeros.
    RuntimeError
        If tensors cannot be broadcast together.
    """
    if not a.is_complex():
        raise TypeError(f"Numerator must be complex tensor, got {a.dtype}")
    if not b.is_complex():
        raise TypeError(f"Denominator must be complex tensor, got {b.dtype}")

    # Check for zeros in denominator
    if torch.any(torch.abs(b) == 0):
        raise ValueError("Division by zero in denominator")

    try:
        return torch.div(a, b)
    except RuntimeError as e:
        raise RuntimeError(f"Cannot broadcast tensors with shapes {a.shape} and {b.shape}") from e


def make_complex(real: Tensor, imag: Tensor) -> Tensor:
    """Construct complex tensor from real and imaginary parts.

    Fundamental constructor for complex tensors in spectral transforms.

    Parameters
    ----------
    real : Tensor
        Real part tensor.
    imag : Tensor
        Imaginary part tensor.

    Returns
    -------
    Tensor
        Complex tensor.

    Raises
    ------
    TypeError
        If inputs are not real tensors.
    RuntimeError
        If tensors cannot be broadcast together.
    """
    if real.is_complex():
        raise TypeError(f"Real part must be real tensor, got {real.dtype}")
    if imag.is_complex():
        raise TypeError(f"Imaginary part must be real tensor, got {imag.dtype}")

    try:
        return torch.complex(real, imag)
    except RuntimeError as e:
        raise RuntimeError(f"Cannot broadcast tensors with shapes {real.shape} and {imag.shape}") from e


def split_complex(x: Tensor) -> tuple[Tensor, Tensor]:
    """Split complex tensor into real and imaginary parts.

    Useful for separate processing of real and imaginary components
    in spectral neural networks and filter implementations.

    Parameters
    ----------
    x : Tensor
        Input complex tensor.

    Returns
    -------
    tuple[Tensor, Tensor]
        Tuple of (real_part, imaginary_part) tensors.

    Raises
    ------
    TypeError
        If input is not a complex tensor.
    """
    if not x.is_complex():
        raise TypeError(f"Input must be complex tensor, got {x.dtype}")

    return torch.real(x), torch.imag(x)


def complex_relu(x: Tensor) -> Tensor:
    """Apply ReLU activation to complex tensor.

    Applies ReLU to both real and imaginary parts independently.
    Note: This is not holomorphic but useful for some neural architectures.

    This specialized activation is designed for complex-valued neural networks
    in spectral transformers where non-linearity is needed in both components.

    Parameters
    ----------
    x : Tensor
        Input complex tensor.

    Returns
    -------
    Tensor
        Complex tensor with ReLU applied to each part.

    Raises
    ------
    TypeError
        If input is not a complex tensor.
    """
    if not x.is_complex():
        raise TypeError(f"Input must be complex tensor, got {x.dtype}")

    real_part = torch.real(x)
    imag_part = torch.imag(x)

    real_relu = torch.relu(real_part)
    imag_relu = torch.relu(imag_part)

    return torch.complex(real_relu, imag_relu)


def complex_dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Apply dropout to complex tensor.

    Applies dropout to magnitude while preserving phase relationships.
    This is superior to independent real/imaginary dropout for spectral data.

    This specialized dropout maintains the complex structure essential for
    spectral transformations while providing regularization.

    Parameters
    ----------
    x : Tensor
        Input complex tensor.
    p : float, default=0.5
        Dropout probability.
    training : bool, default=True
        Whether in training mode.

    Returns
    -------
    Tensor
        Complex tensor with dropout applied.

    Raises
    ------
    TypeError
        If input is not a complex tensor.
    ValueError
        If dropout probability is not in [0, 1].
    """
    if not x.is_complex():
        raise TypeError(f"Input must be complex tensor, got {x.dtype}")

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Dropout probability must be in [0, 1], got {p}")

    if not training or p == 0.0:
        return x

    # Create dropout mask for the magnitude
    # This preserves phase relationships better than independent dropout
    magnitude = torch.abs(x)
    phase = torch.angle(x)

    # Apply dropout to magnitude only
    dropped_magnitude = torch.nn.functional.dropout(magnitude, p=p, training=training)

    # Reconstruct complex tensor with same phases
    return torch.polar(dropped_magnitude, phase)
