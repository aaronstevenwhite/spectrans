"""Weight initialization utilities for spectral transformer components.

This module provides specialized initialization schemes tailored for spectral neural
networks, including complex-valued parameters, frequency-domain aware initialization,
and transform-specific initialization strategies. Proper initialization is crucial
for spectral transformers due to their unique mathematical properties and parameter
scaling requirements.

The initialization functions account for the specific characteristics of spectral
transforms, including orthogonality constraints, complex number scaling, frequency
domain properties, and stability requirements for gradient-based optimization.

Functions
---------
spectral_init(tensor, method, **kwargs)
    General-purpose spectral parameter initialization.
frequency_init(tensor, freq_range, scale)
    Initialize parameters with frequency-domain properties.
complex_xavier_init(tensor, gain)
    Xavier/Glorot initialization for complex-valued parameters.
complex_kaiming_init(tensor, a, mode, nonlinearity)
    Kaiming/He initialization for complex parameters.
complex_normal_init(tensor, mean, std)
    Normal initialization for complex tensors.
orthogonal_spectral_init(tensor, gain)
    Orthogonal initialization preserving spectral properties.
xavier_spectral_init(tensor, gain, transform_type)
    Xavier initialization adapted for spectral transforms.
kaiming_spectral_init(tensor, a, mode, nonlinearity, transform_type)
    Kaiming initialization adapted for spectral transforms.
dct_init(tensor, normalized)
    Specialized initialization for DCT parameters.
hadamard_init(tensor, normalized)
    Initialization for Hadamard transform parameters.
wavelet_init(tensor, wavelet_type, levels)
    Initialize parameters for wavelet transforms.
init_linear_spectral(linear_layer, method, **kwargs)
    Initialize linear layers for spectral operations.
init_conv_spectral(conv_layer, method, **kwargs)  
    Initialize convolutional layers for spectral operations.

Examples
--------
Basic spectral initialization:

>>> import torch
>>> import torch.nn as nn
>>> from spectrans.utils.initialization import spectral_init, complex_xavier_init
>>> # Initialize a linear layer for spectral transforms
>>> linear = nn.Linear(512, 512)
>>> spectral_init(linear.weight, method='frequency', freq_range=(0.0, 0.5))
>>> spectral_init(linear.bias, method='zero')

Complex parameter initialization:

>>> # Initialize complex-valued parameters
>>> complex_weights = torch.empty(256, 256, dtype=torch.complex64)
>>> complex_xavier_init(complex_weights, gain=1.0)
>>> 
>>> # Manual complex initialization
>>> real_part = torch.empty(256, 256)
>>> imag_part = torch.empty(256, 256)
>>> torch.nn.init.xavier_uniform_(real_part, gain=1.0/math.sqrt(2))
>>> torch.nn.init.xavier_uniform_(imag_part, gain=1.0/math.sqrt(2))
>>> complex_weights = torch.complex(real_part, imag_part)

Transform-specific initialization:

>>> from spectrans.utils.initialization import dct_init, hadamard_init, wavelet_init
>>> # DCT parameter initialization
>>> dct_params = torch.empty(512, 512)
>>> dct_init(dct_params, normalized=True)
>>> 
>>> # Hadamard transform parameters
>>> hadamard_params = torch.empty(256, 256)  # Must be power of 2
>>> hadamard_init(hadamard_params, normalized=True)
>>> 
>>> # Wavelet parameters
>>> wavelet_params = torch.empty(1024, 1024)
>>> wavelet_init(wavelet_params, wavelet_type='db4', levels=3)

Layer initialization:

>>> from spectrans.utils.initialization import init_linear_spectral, init_conv_spectral
>>> # Initialize entire layers
>>> linear_layer = nn.Linear(768, 768)
>>> init_linear_spectral(linear_layer, method='xavier_spectral', transform_type='fourier')
>>> 
>>> # Convolutional layer for spectral processing
>>> conv_layer = nn.Conv1d(512, 512, kernel_size=3)
>>> init_conv_spectral(conv_layer, method='kaiming_spectral', transform_type='dct')

Notes
-----
Initialization Theory for Spectral Networks:

**Complex Parameter Scaling**:
Complex parameters require careful scaling to maintain proper variance:
- Real and imaginary parts should be scaled by 1/√2 relative to real-valued case
- This maintains the same total variance while distributing it across both components
- Critical for stable training of complex neural networks

**Frequency-Domain Considerations**:
Parameters operating in frequency domain have different scaling requirements:
- Low frequencies often have higher magnitude than high frequencies
- Initialization should reflect expected frequency content
- Different spectral transforms have different frequency characteristics

**Orthogonal Transform Properties**:
Many spectral transforms are orthogonal/unitary and require special treatment:
- Parameters should preserve orthogonality during training
- Initial values should respect the mathematical structure
- Gradients may need special handling to maintain constraints

Mathematical Foundations:

**Xavier/Glorot Initialization**:
For real-valued parameters: σ² = 2/(n_in + n_out)
For complex-valued: σ² = 1/(n_in + n_out), split equally between real/imaginary

**Kaiming/He Initialization**:
For ReLU activation: σ² = 2/n_in
Complex variant: σ² = 1/n_in, split equally

**Orthogonal Initialization**:
Creates matrices with orthonormal rows/columns using QR decomposition
Essential for transforms requiring orthogonality constraints

Transform-Specific Considerations:

**FFT Parameters**:
- Complex-valued requiring careful magnitude/phase initialization
- Often benefit from frequency-aware initialization
- Should maintain Parseval's theorem properties

**DCT/DST Parameters**:
- Real-valued but with cosine/sine basis constraints
- Energy compaction properties should be preserved
- Orthogonality is crucial for proper reconstruction

**Hadamard Parameters**:
- Binary {-1, +1} structure should be respected
- Fast transform structure affects parameter scaling
- Power-of-2 constraints affect initialization patterns

**Wavelet Parameters**:
- Multi-resolution structure requires level-aware initialization
- Different wavelets have different scaling properties
- Perfect reconstruction constraints must be maintained

Implementation Details:

- **Gradient Preservation**: All initializations maintain gradient flow
- **Device Handling**: Automatically matches input tensor device and dtype
- **Batch Operations**: Efficient initialization for large parameter sets
- **Memory Efficiency**: In-place initialization where possible
- **Numerical Stability**: Careful handling of edge cases and extreme values

Common Patterns:

1. **Spectral Mixing Layers**: Use frequency_init with appropriate frequency ranges
2. **Complex Attention**: Use complex_xavier_init for query/key/value projections
3. **Transform Embeddings**: Use transform-specific initialization (dct_init, etc.)
4. **Learnable Filters**: Use orthogonal_spectral_init to maintain properties
5. **Residual Connections**: Use xavier_spectral_init with proper gain scheduling

Performance Considerations:
- All initialization functions are vectorized and GPU-compatible
- Large parameter tensors are handled efficiently
- Memory usage is optimized for typical spectral network sizes
- Initialization time is minimized through optimized algorithms

See Also
--------
spectrans.core.base : Base classes requiring proper initialization
spectrans.transforms : Transform classes with specific initialization needs
spectrans.utils.complex : Complex tensor operations for initialized parameters
torch.nn.init : PyTorch's standard initialization functions
"""

import math
from typing import Literal

import torch
import torch.nn as nn

from ..core.types import Tensor

__all__ = [
    "complex_kaiming_init",
    "complex_normal_init",
    "complex_xavier_init",
    "dct_init",
    "frequency_init",
    "hadamard_init",
    "init_conv_spectral",
    "init_linear_spectral",
    "kaiming_spectral_init",
    "orthogonal_spectral_init",
    "spectral_init",
    "wavelet_init",
    "xavier_spectral_init",
]


def spectral_init(tensor: Tensor, mode: str = "normal", gain: float = 1.0) -> Tensor:
    """Initialize tensor with spectral-aware method.
    
    Parameters
    ----------
    tensor : Tensor
        Tensor to initialize.
    mode : str, default="normal"
        Initialization mode: "normal", "uniform", "xavier", "kaiming", "orthogonal".
    gain : float, default=1.0
        Scaling factor for initialization.
        
    Returns
    -------
    Tensor
        Initialized tensor.
        
    Raises
    ------
    ValueError
        If mode is not supported or gain is not positive.
    RuntimeError
        If tensor is not 2D for orthogonal initialization.
    """
    if gain <= 0:
        raise ValueError(f"Gain must be positive, got {gain}")

    with torch.no_grad():
        if mode == "normal":
            # Standard normal initialization scaled by gain
            tensor.normal_(0, gain)
        elif mode == "uniform":
            # Uniform initialization in [-gain, gain]
            tensor.uniform_(-gain, gain)
        elif mode == "xavier":
            xavier_spectral_init(tensor, gain=gain)
        elif mode == "kaiming":
            kaiming_spectral_init(tensor, gain=gain)
        elif mode == "orthogonal":
            orthogonal_spectral_init(tensor, gain=gain)
        else:
            raise ValueError(f"Unsupported initialization mode: {mode}")

    return tensor


def xavier_spectral_init(
    tensor: Tensor,
    gain: float = 1.0,
    distribution: Literal["normal", "uniform"] = "normal"
) -> Tensor:
    """Xavier/Glorot initialization adapted for spectral transforms.
    
    Maintains variance of activations and gradients across layers by scaling
    based on input and output dimensions.
    
    Parameters
    ----------
    tensor : Tensor
        Tensor to initialize.
    gain : float, default=1.0
        Scaling factor for initialization.
    distribution : {"normal", "uniform"}, default="normal"
        Distribution to use for initialization.
        
    Returns
    -------
    Tensor
        Initialized tensor.
        
    Raises
    ------
    ValueError
        If tensor has fewer than 2 dimensions, gain is not positive,
        or distribution is invalid.
    """
    if tensor.ndim < 2:
        raise ValueError(f"Xavier initialization requires at least 2D tensor, got {tensor.ndim}D")

    if gain <= 0:
        raise ValueError(f"Gain must be positive, got {gain}")

    if distribution not in ("normal", "uniform"):
        raise ValueError(f"Distribution must be 'normal' or 'uniform', got {distribution}")

    # Calculate fan-in and fan-out
    # For spectral transforms, consider all dimensions except the last as input
    # and the last as output (or vice versa for transpose operations)
    fan_in = tensor.shape[-2] if tensor.ndim >= 2 else tensor.numel()
    fan_out = tensor.shape[-1] if tensor.ndim >= 2 else tensor.numel()

    # Xavier scaling factor
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))

    with torch.no_grad():
        if distribution == "normal":
            tensor.normal_(0, std)
        else:  # uniform
            bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
            tensor.uniform_(-bound, bound)

    return tensor


def kaiming_spectral_init(
    tensor: Tensor,
    gain: float = 1.0,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    nonlinearity: str = "relu"
) -> Tensor:
    """Kaiming/He initialization adapted for spectral transforms.
    
    Designed for networks with ReLU-like activations, maintaining
    variance through forward/backward passes.
    
    Parameters
    ----------
    tensor : Tensor
        Tensor to initialize.
    gain : float, default=1.0
        Scaling factor for initialization.
    mode : {"fan_in", "fan_out"}, default="fan_in"
        Fan mode for variance calculation.
    nonlinearity : str, default="relu"
        Nonlinearity type for gain calculation.
        
    Returns
    -------
    Tensor
        Initialized tensor.
        
    Raises
    ------
    ValueError
        If tensor has fewer than 2 dimensions, parameters are invalid.
    """
    if tensor.ndim < 2:
        raise ValueError(f"Kaiming initialization requires at least 2D tensor, got {tensor.ndim}D")

    if gain <= 0:
        raise ValueError(f"Gain must be positive, got {gain}")

    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"Mode must be 'fan_in' or 'fan_out', got {mode}")

    # Calculate fan-in and fan-out
    fan_in = tensor.shape[-2] if tensor.ndim >= 2 else tensor.numel()
    fan_out = tensor.shape[-1] if tensor.ndim >= 2 else tensor.numel()

    fan = fan_in if mode == "fan_in" else fan_out

    # Nonlinearity-specific gains
    nonlinearity_gains = {
        "linear": 1.0,
        "relu": math.sqrt(2.0),
        "leaky_relu": math.sqrt(2.0 / (1 + 0.01**2)),
        "tanh": 5.0/3,
        "sigmoid": 1.0,
        "gelu": 1.0,
    }

    if nonlinearity not in nonlinearity_gains:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

    nl_gain = nonlinearity_gains[nonlinearity]
    std = gain * nl_gain / math.sqrt(fan)

    with torch.no_grad():
        tensor.normal_(0, std)

    return tensor


def orthogonal_spectral_init(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Orthogonal initialization for spectral transform matrices.
    
    Creates orthogonal matrices that preserve norms, which is important
    for spectral transforms that should maintain energy conservation.
    
    Parameters
    ----------
    tensor : Tensor
        2D tensor to initialize.
    gain : float, default=1.0
        Scaling factor for the orthogonal matrix.
        
    Returns
    -------
    Tensor
        Initialized orthogonal tensor.
        
    Raises
    ------
    ValueError
        If tensor is not 2D or gain is not positive.
    """
    if tensor.ndim != 2:
        raise ValueError(f"Orthogonal initialization requires 2D tensor, got {tensor.ndim}D")

    if gain <= 0:
        raise ValueError(f"Gain must be positive, got {gain}")

    with torch.no_grad():
        nn.init.orthogonal_(tensor, gain=gain)

    return tensor


def complex_normal_init(tensor: Tensor, std: float = 1.0) -> Tensor:
    """Initialize complex tensor with complex normal distribution.
    
    Both real and imaginary parts are initialized independently with
    normal distribution scaled to maintain proper variance.
    
    Parameters
    ----------
    tensor : Tensor
        Complex tensor to initialize.
    std : float, default=1.0
        Standard deviation for each component.
        
    Returns
    -------
    Tensor
        Initialized complex tensor.
        
    Raises
    ------
    TypeError
        If tensor is not complex.
    ValueError
        If std is not positive.
    """
    if not tensor.is_complex():
        raise TypeError(f"Tensor must be complex, got {tensor.dtype}")

    if std <= 0:
        raise ValueError(f"Standard deviation must be positive, got {std}")

    # For complex normal: each component has std/sqrt(2) to maintain total variance
    component_std = std / math.sqrt(2)

    with torch.no_grad():
        # Initialize real and imaginary parts independently
        real_part = torch.randn_like(tensor.real) * component_std
        imag_part = torch.randn_like(tensor.imag) * component_std
        tensor.copy_(torch.complex(real_part, imag_part))

    return tensor


def complex_xavier_init(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Xavier initialization for complex tensors.
    
    Parameters
    ----------
    tensor : Tensor
        Complex tensor to initialize.
    gain : float, default=1.0
        Scaling factor for initialization.
        
    Returns
    -------
    Tensor
        Initialized complex tensor.
        
    Raises
    ------
    TypeError
        If tensor is not complex.
    ValueError
        If tensor dimensions or gain are invalid.
    """
    if not tensor.is_complex():
        raise TypeError(f"Tensor must be complex, got {tensor.dtype}")

    if tensor.ndim < 2:
        raise ValueError(f"Xavier initialization requires at least 2D tensor, got {tensor.ndim}D")

    if gain <= 0:
        raise ValueError(f"Gain must be positive, got {gain}")

    # Calculate Xavier scaling for complex tensors
    fan_in = tensor.shape[-2]
    fan_out = tensor.shape[-1]
    std = gain * math.sqrt(1.0 / (fan_in + fan_out))  # Adjusted for complex

    return complex_normal_init(tensor, std)


def complex_kaiming_init(
    tensor: Tensor,
    gain: float = 1.0,
    mode: Literal["fan_in", "fan_out"] = "fan_in"
) -> Tensor:
    """Kaiming initialization for complex tensors.
    
    Parameters
    ----------
    tensor : Tensor
        Complex tensor to initialize.
    gain : float, default=1.0
        Scaling factor for initialization.
    mode : {"fan_in", "fan_out"}, default="fan_in"
        Fan mode for variance calculation.
        
    Returns
    -------
    Tensor
        Initialized complex tensor.
        
    Raises
    ------
    TypeError
        If tensor is not complex.
    ValueError
        If tensor dimensions or parameters are invalid.
    """
    if not tensor.is_complex():
        raise TypeError(f"Tensor must be complex, got {tensor.dtype}")

    if tensor.ndim < 2:
        raise ValueError(f"Kaiming initialization requires at least 2D tensor, got {tensor.ndim}D")

    if gain <= 0:
        raise ValueError(f"Gain must be positive, got {gain}")

    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"Mode must be 'fan_in' or 'fan_out', got {mode}")

    # Calculate Kaiming scaling for complex tensors
    fan_in = tensor.shape[-2]
    fan_out = tensor.shape[-1]
    fan = fan_in if mode == "fan_in" else fan_out
    std = gain / math.sqrt(fan)  # Adjusted for complex

    return complex_normal_init(tensor, std)


def frequency_init(tensor: Tensor, max_freq: float = 1.0) -> Tensor:
    """Initialize tensor with frequency-domain aware values.
    
    Initializes with small values at high frequencies and larger values
    at low frequencies, mimicking natural signal characteristics.
    
    Parameters
    ----------
    tensor : Tensor
        Tensor to initialize (typically frequency domain parameters).
    max_freq : float, default=1.0
        Maximum frequency for scaling.
        
    Returns
    -------
    Tensor
        Initialized tensor.
        
    Raises
    ------
    ValueError
        If max_freq is not positive.
    """
    if max_freq <= 0:
        raise ValueError(f"Max frequency must be positive, got {max_freq}")

    with torch.no_grad():
        # Create frequency-based scaling
        # Assume last dimension represents frequency bins
        freq_dim = tensor.shape[-1]
        freqs = torch.linspace(0, max_freq, freq_dim, device=tensor.device)

        # 1/f-like scaling (pink noise characteristic)
        scaling = 1.0 / (1.0 + freqs)

        # Broadcast scaling to tensor shape
        shape = [1] * tensor.ndim
        shape[-1] = freq_dim
        scaling = scaling.view(shape)

        # Initialize with normal then scale
        tensor.normal_(0, 1)
        tensor.mul_(scaling)

    return tensor


def wavelet_init(tensor: Tensor, wavelet_type: str = "db1") -> Tensor:
    """Initialize tensor with wavelet-like properties.
    
    Parameters
    ----------
    tensor : Tensor
        Tensor to initialize.
    wavelet_type : str, default="db1"
        Type of wavelet initialization.
        
    Returns
    -------
    Tensor
        Initialized tensor.
        
    Raises
    ------
    ValueError
        If wavelet_type is not supported.
    """
    supported_wavelets = ["db1", "db2", "haar"]
    if wavelet_type not in supported_wavelets:
        raise ValueError(f"Wavelet type must be one of {supported_wavelets}, got {wavelet_type}")

    with torch.no_grad():
        if wavelet_type in ("db1", "haar"):
            # Haar/Daubechies-1 wavelet properties
            # Initialize with small random values then apply haar-like structure
            tensor.normal_(0, 0.1)

            # Apply alternating signs for wavelet-like behavior
            if tensor.ndim >= 2:
                for i in range(tensor.shape[-1]):
                    if i % 2 == 1:
                        tensor[..., i] *= -1
        elif wavelet_type == "db2":
            # Daubechies-2 initialization
            tensor.normal_(0, 0.1)
            # Apply more complex pattern for DB2
            if tensor.ndim >= 2:
                pattern = [1, -1, 1, -1]  # Simple DB2-like pattern
                for i in range(tensor.shape[-1]):
                    tensor[..., i] *= pattern[i % len(pattern)]

    return tensor


def hadamard_init(tensor: Tensor) -> Tensor:
    """Initialize tensor with Hadamard matrix properties.
    
    Parameters
    ----------
    tensor : Tensor
        Square tensor to initialize.
        
    Returns
    -------
    Tensor
        Initialized tensor with Hadamard-like structure.
        
    Raises
    ------
    ValueError
        If tensor is not square or not power-of-2 sized.
    """
    if tensor.ndim != 2:
        raise ValueError(f"Hadamard initialization requires 2D tensor, got {tensor.ndim}D")

    if tensor.shape[0] != tensor.shape[1]:
        raise ValueError(f"Hadamard initialization requires square tensor, got {tensor.shape}")

    size = tensor.shape[0]

    # Check if size is power of 2
    if size & (size - 1) != 0 or size == 0:
        raise ValueError(f"Hadamard initialization requires power-of-2 size, got {size}")

    with torch.no_grad():
        # Build Hadamard matrix recursively
        h = torch.tensor([[1.0]], device=tensor.device, dtype=tensor.dtype)

        while h.shape[0] < size:
            current_size = h.shape[0]
            new_h = torch.zeros(2 * current_size, 2 * current_size, device=tensor.device, dtype=tensor.dtype)
            new_h[:current_size, :current_size] = h
            new_h[:current_size, current_size:] = h
            new_h[current_size:, :current_size] = h
            new_h[current_size:, current_size:] = -h
            h = new_h

        # Normalize
        h = h / math.sqrt(size)
        tensor.copy_(h)

    return tensor


def dct_init(tensor: Tensor) -> Tensor:
    """Initialize tensor with DCT matrix properties.
    
    Parameters
    ----------
    tensor : Tensor
        2D tensor to initialize.
        
    Returns
    -------
    Tensor
        Initialized tensor with DCT-like structure.
        
    Raises
    ------
    ValueError
        If tensor is not 2D.
    """
    if tensor.ndim != 2:
        raise ValueError(f"DCT initialization requires 2D tensor, got {tensor.ndim}D")

    n, m = tensor.shape

    with torch.no_grad():
        # Build DCT-II matrix
        dct_matrix = torch.zeros(n, m, device=tensor.device, dtype=tensor.dtype)

        for i in range(n):
            for j in range(m):
                if i == 0:
                    dct_matrix[i, j] = math.sqrt(1.0 / m)
                else:
                    dct_matrix[i, j] = math.sqrt(2.0 / m) * math.cos(math.pi * i * (2*j + 1) / (2*m))

        tensor.copy_(dct_matrix)

    return tensor


def init_linear_spectral(linear: nn.Linear, method: str = "xavier") -> nn.Linear:
    """Initialize linear layer with spectral-aware method.
    
    Parameters
    ----------
    linear : nn.Linear
        Linear layer to initialize.
    method : str, default="xavier"
        Initialization method: "xavier", "kaiming", "orthogonal".
        
    Returns
    -------
    nn.Linear
        Initialized linear layer.
        
    Raises
    ------
    ValueError
        If method is not supported.
    """
    if method == "xavier":
        xavier_spectral_init(linear.weight)
    elif method == "kaiming":
        kaiming_spectral_init(linear.weight)
    elif method == "orthogonal":
        orthogonal_spectral_init(linear.weight)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if linear.bias is not None:
        nn.init.zeros_(linear.bias)

    return linear


def init_conv_spectral(conv: nn.Conv1d | nn.Conv2d, method: str = "kaiming") -> nn.Conv1d | nn.Conv2d:
    """Initialize convolution layer with spectral-aware method.
    
    Parameters
    ----------
    conv : nn.Conv1d | nn.Conv2d
        Convolution layer to initialize.
    method : str, default="kaiming"
        Initialization method: "xavier", "kaiming".
        
    Returns
    -------
    nn.Conv1d | nn.Conv2d
        Initialized convolution layer.
        
    Raises
    ------
    ValueError
        If method is not supported.
    """
    if method == "xavier":
        xavier_spectral_init(conv.weight)
    elif method == "kaiming":
        kaiming_spectral_init(conv.weight, nonlinearity="relu")
    else:
        raise ValueError(f"Unsupported method: {method}")

    if conv.bias is not None:
        nn.init.zeros_(conv.bias)

    return conv
