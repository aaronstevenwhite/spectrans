"""Utility functions for spectral transformer implementations.

This module provides essential utility functions for spectral neural networks,
including specialized complex number operations, advanced initialization schemes
for spectral parameters, and padding utilities for signal processing operations.

These utilities are designed to support the mathematical rigor and numerical
stability required for spectral transformer architectures while providing
convenient abstractions for common operations.

Functions
---------
complex_conjugate(x)
    Compute complex conjugate with proper error handling.
complex_multiply(a, b)
    Element-wise complex multiplication with broadcasting.
complex_divide(a, b) 
    Complex division with zero-division safety checks.
complex_modulus(x)
    Compute magnitude of complex tensors.
complex_phase(x)
    Extract phase angles from complex tensors.
complex_polar(magnitude, phase)
    Construct complex tensors from polar coordinates.
complex_exp(x)
    Complex exponential function.
complex_log(x)
    Complex logarithm with numerical safety.
complex_relu(x)
    ReLU activation applied to both real and imaginary parts.
complex_dropout(x, p, training)
    Dropout preserving phase relationships.
make_complex(real, imag)
    Construct complex tensors from real/imaginary parts.
split_complex(x)
    Split complex tensors into real/imaginary components.
spectral_init(tensor, method)
    Initialize parameters for spectral neural networks.
frequency_init(tensor, freq_range)
    Initialize parameters with frequency-domain properties.
orthogonal_spectral_init(tensor)
    Initialize with orthogonality constraints.
complex_xavier_init(tensor)
    Xavier initialization for complex-valued parameters.
complex_kaiming_init(tensor)  
    Kaiming initialization for complex parameters.
pad_to_power_of_2(x, dim)
    Pad tensor to next power of 2 for efficient FFT.
pad_for_fft(x, target_length, dim)
    Pad tensor for FFT operations.
circular_pad(x, padding, dim)
    Apply circular (periodic) padding.
reflect_pad(x, padding, dim)
    Apply reflection padding for boundary handling.

Examples
--------
Complex number operations:

>>> import torch
>>> from spectrans.utils import complex_multiply, complex_polar, split_complex
>>> # Create complex tensors
>>> z1 = torch.complex(torch.randn(10), torch.randn(10))
>>> z2 = torch.complex(torch.randn(10), torch.randn(10))
>>> product = complex_multiply(z1, z2)
>>> 
>>> # Convert to polar form
>>> magnitude = torch.abs(z1)
>>> phase = torch.angle(z1)
>>> z1_reconstructed = complex_polar(magnitude, phase)

Spectral parameter initialization:

>>> from spectrans.utils import spectral_init, complex_xavier_init
>>> import torch.nn as nn
>>> # Initialize a linear layer for spectral transforms
>>> linear = nn.Linear(512, 512)
>>> spectral_init(linear.weight, method='frequency')
>>> 
>>> # Initialize complex-valued parameters
>>> complex_params = torch.empty(256, 256, dtype=torch.complex64)
>>> complex_xavier_init(complex_params)

Padding for spectral operations:

>>> from spectrans.utils import pad_to_power_of_2, pad_for_fft
>>> signal = torch.randn(32, 500)  # 500 is not power of 2
>>> padded = pad_to_power_of_2(signal, dim=-1)  # Pads to 512
>>> 
>>> # Pad to specific FFT length
>>> fft_ready = pad_for_fft(signal, target_length=1024, dim=-1)

Notes
-----
Design Philosophy:

The utility functions follow these principles:

1. **Mathematical Safety**: All operations include proper error checking
   and handle edge cases (zeros, infinities, etc.)

2. **Numerical Stability**: Implementations prioritize numerical stability
   over raw performance where trade-offs exist

3. **Type Safety**: Comprehensive type checking and clear error messages
   for incorrect usage patterns

4. **Gradient Compatibility**: All operations support automatic differentiation
   for end-to-end neural network training

5. **Broadcasting Support**: Operations follow PyTorch broadcasting conventions
   for flexible tensor manipulation

Complex Number Operations:
The complex utilities provide a consistent interface for complex tensor operations
with proper error handling and mathematical safety. While many wrap existing
PyTorch functions, they add domain-specific validation and optimization for
spectral neural networks.

Initialization Schemes:
Spectral neural networks often require specialized parameter initialization due to:
- Different scaling properties of spectral transforms
- Complex-valued parameters requiring magnitude/phase initialization
- Orthogonality constraints for certain spectral methods
- Frequency-domain parameter interpretation

Padding Utilities:
Signal processing operations often require specific padding strategies:
- Power-of-2 lengths for efficient FFT computation
- Circular padding for periodic signal assumptions  
- Reflection padding for boundary effect minimization
- Zero padding with proper unpadding for shape restoration

Performance Considerations:
- All utilities are optimized for batch operations
- GPU acceleration through native PyTorch operations
- Memory efficiency with in-place operations where safe
- Vectorized implementations for optimal throughput

See Also
--------
spectrans.utils.complex : Complex tensor operations
spectrans.utils.initialization : Parameter initialization schemes
spectrans.utils.padding : Padding utilities for signal processing
spectrans.transforms : Spectral transforms using these utilities
"""

from .complex import (
    complex_conjugate,
    complex_divide,
    complex_dropout,
    complex_exp,
    complex_log,
    complex_modulus,
    complex_multiply,
    complex_phase,
    complex_polar,
    complex_relu,
    make_complex,
    split_complex,
)
from .initialization import (
    complex_kaiming_init,
    complex_normal_init,
    complex_xavier_init,
    dct_init,
    frequency_init,
    hadamard_init,
    init_conv_spectral,
    init_linear_spectral,
    kaiming_spectral_init,
    orthogonal_spectral_init,
    spectral_init,
    wavelet_init,
    xavier_spectral_init,
)
from .padding import (
    circular_pad,
    pad_for_convolution,
    pad_for_fft,
    pad_sequence,
    pad_to_length,
    pad_to_power_of_2,
    reflect_pad,
    symmetric_pad,
    unpad_sequence,
    unpad_to_length,
    zero_pad,
)

__all__ = [
    # Complex operations
    "complex_conjugate",
    "complex_divide",
    "complex_dropout",
    "complex_exp",
    "complex_kaiming_init",
    "complex_log",
    "complex_modulus",
    "complex_multiply",
    "complex_normal_init",
    "complex_phase",
    "complex_polar",
    "complex_relu",
    "complex_xavier_init",
    # Padding and general utilities
    "circular_pad",
    "dct_init",
    "frequency_init",
    "hadamard_init",
    "init_conv_spectral",
    "init_linear_spectral",
    "kaiming_spectral_init",
    "make_complex",
    "orthogonal_spectral_init",
    "pad_for_convolution",
    "pad_for_fft",
    "pad_sequence",
    "pad_to_length",
    "pad_to_power_of_2",
    "reflect_pad",
    "spectral_init",
    "split_complex",
    "symmetric_pad",
    "unpad_sequence",
    "unpad_to_length",
    "wavelet_init",
    "xavier_spectral_init",
    "zero_pad",
]
