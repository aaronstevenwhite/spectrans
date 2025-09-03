"""Fourier Neural Operator (FNO) implementation for spectral transformers.

This module provides the Fourier Neural Operator, which learns mappings between
infinite-dimensional function spaces by parameterizing integral kernels in the
Fourier domain. FNO is particularly effective for learning solution operators
for partial differential equations and other continuous transformations.

The FNO architecture combines spectral convolutions in the Fourier domain with
pointwise linear transformations, enabling efficient learning of global
dependencies while maintaining resolution invariance.

Classes
-------
FourierNeuralOperator
    Base FNO layer for learning operators in function spaces.
FNOBlock
    Complete FNO block with spectral convolution and feedforward network.
SpectralConv1d
    1D spectral convolution operator.
SpectralConv2d
    2D spectral convolution operator for image-like data.

Examples
--------
Basic FNO layer:

>>> import torch
>>> from spectrans.layers.operators.fno import FourierNeuralOperator
>>> fno = FourierNeuralOperator(hidden_dim=64, modes=16)
>>> x = torch.randn(32, 128, 64)  # (batch, sequence, channels)
>>> output = fno(x)
>>> assert output.shape == x.shape

FNO block with residual connection:

>>> from spectrans.layers.operators.fno import FNOBlock
>>> block = FNOBlock(hidden_dim=64, modes=16, mlp_ratio=2.0)
>>> x = torch.randn(32, 128, 64)
>>> output = block(x)

2D spectral convolution for images:

>>> from spectrans.layers.operators.fno import SpectralConv2d
>>> conv2d = SpectralConv2d(in_channels=3, out_channels=64, modes=(32, 32))
>>> image = torch.randn(32, 3, 256, 256)
>>> output = conv2d(image)

Notes
-----
The FNO learns the kernel K in the integral operator:
    (K*v)(x) = ∫ k(x,y)v(y)dy

By parameterizing k in the Fourier domain as R_θ, the convolution becomes:
    F[(K*v)] = R_θ · F[v]

This allows efficient computation via FFT and learnable complex weights R_θ
that are truncated to retain only the lowest frequency modes.

References
----------
.. [1] Li, Z., et al., "Fourier Neural Operator for Parametric Partial
       Differential Equations", ICLR 2021.
.. [2] Guibas, J., et al., "Adaptive Fourier Neural Operators", ICLR 2022.

See Also
--------
spectrans.layers.mixing.afno : Adaptive FNO mixing layer.
spectrans.transforms.fourier : Underlying FFT implementations.
"""

import torch
import torch.nn as nn

from spectrans.core.base import SpectralComponent
from spectrans.core.types import ActivationType, NormType


class SpectralConv1d(nn.Module):
    """1D Spectral convolution layer.

    Performs convolution in the Fourier domain by element-wise multiplication
    with learnable complex-valued weights on truncated modes.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    modes : int
        Number of Fourier modes to keep (frequency truncation).

    Attributes
    ----------
    in_channels : int
        Input channel count.
    out_channels : int
        Output channel count.
    modes : int
        Number of retained Fourier modes.
    weights : nn.Parameter
        Complex-valued learnable weights of shape (in_channels, out_channels, modes).

    Examples
    --------
    >>> conv = SpectralConv1d(in_channels=64, out_channels=64, modes=16)
    >>> x = torch.randn(32, 64, 128)  # (batch, channels, sequence)
    >>> output = conv(x)
    >>> assert output.shape == x.shape
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Complex weights for Fourier modes
        # Scale initialization for stability
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, 2) * scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, sequence_length).
        """
        batch_size, _, seq_len = x.shape

        # Compute FFT
        x_ft = torch.fft.rfft(x, dim=-1)

        # Truncate to retained modes
        x_ft_truncated = x_ft[..., :self.modes]

        # Prepare output in Fourier domain
        out_ft = torch.zeros(
            batch_size, self.out_channels, seq_len // 2 + 1,
            dtype=x_ft.dtype, device=x.device
        )

        # Apply spectral convolution via complex multiplication
        # Convert weights to complex and match input dtype
        weights_complex = torch.view_as_complex(self.weights.to(x.dtype))

        # Perform einsum for channel mixing with mode-wise multiplication
        # Shape: (batch, in_channels, modes) x (in_channels, out_channels, modes)
        # -> (batch, out_channels, modes)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bim,iom->bom",
            x_ft_truncated,
            weights_complex
        )

        # Inverse FFT to get back to spatial domain
        out = torch.fft.irfft(out_ft, n=seq_len, dim=-1)

        return out  # type: ignore[no-any-return]


class SpectralConv2d(nn.Module):
    """2D Spectral convolution layer.

    Performs 2D convolution in the Fourier domain for image-like data.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    modes : tuple[int, int]
        Number of Fourier modes to keep in each dimension (height, width).

    Attributes
    ----------
    in_channels : int
        Input channel count.
    out_channels : int
        Output channel count.
    modes1 : int
        Number of retained modes in first spatial dimension.
    modes2 : int
        Number of retained modes in second spatial dimension.
    weights : nn.Parameter
        Complex weights of shape (in_channels, out_channels, modes1, modes2).

    Examples
    --------
    >>> conv2d = SpectralConv2d(in_channels=3, out_channels=64, modes=(32, 32))
    >>> x = torch.randn(8, 3, 256, 256)
    >>> output = conv2d(x)
    >>> assert output.shape == (8, 64, 256, 256)
    """

    def __init__(self, in_channels: int, out_channels: int, modes: tuple[int, int]):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes[0]
        self.modes2 = modes[1]

        # Complex weights for 2D Fourier modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2) * scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D spectral convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        batch_size, _, h, w = x.shape

        # Compute 2D FFT
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        # Prepare output
        out_ft = torch.zeros(
            batch_size, self.out_channels, h, w // 2 + 1,
            dtype=x_ft.dtype, device=x.device
        )

        # Truncate and apply convolution
        weights_complex = torch.view_as_complex(self.weights.to(x.dtype))

        # Apply convolution on truncated modes
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bihw,iohw->bohw",
            x_ft[:, :, :self.modes1, :self.modes2],
            weights_complex
        )

        # Inverse FFT
        out = torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))

        return out  # type: ignore[no-any-return]


class FourierNeuralOperator(SpectralComponent):
    """Fourier Neural Operator layer for learning operators in function spaces.

    This layer combines spectral convolution with pointwise linear transformations
    to learn mappings between function spaces efficiently.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension (number of channels).
    modes : int | tuple[int, ...], optional
        Number of Fourier modes to retain. Can be an integer for 1D or tuple
        for higher dimensions. Default is 16.
    activation : str, optional
        Activation function. Options: 'gelu', 'relu', 'tanh'. Default is 'gelu'.
    use_spectral_conv : bool, optional
        Whether to use spectral convolution. Default is True.
    use_linear : bool, optional
        Whether to use pointwise linear transformation. Default is True.

    Attributes
    ----------
    hidden_dim : int
        Hidden dimension size.
    modes : int | tuple[int, ...]
        Number of retained Fourier modes.
    spectral_conv : SpectralConv1d | SpectralConv2d | None
        Spectral convolution layer if enabled.
    linear : nn.Conv1d | nn.Conv2d | None
        Pointwise convolution layer if enabled.
    activation : nn.Module
        Activation function.

    Examples
    --------
    >>> fno = FourierNeuralOperator(hidden_dim=64, modes=16)
    >>> x = torch.randn(32, 128, 64)  # (batch, sequence, channels)
    >>> output = fno(x)
    >>> assert output.shape == x.shape
    """

    spectral_conv: SpectralConv1d | SpectralConv2d | None
    linear: nn.Conv1d | nn.Conv2d | None
    activation: nn.Module

    def __init__(
        self,
        hidden_dim: int,
        modes: int | tuple[int, ...] = 16,
        activation: ActivationType = 'gelu',
        use_spectral_conv: bool = True,
        use_linear: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.modes = modes
        self.use_spectral_conv = use_spectral_conv
        self.use_linear = use_linear

        if not use_spectral_conv and not use_linear:
            raise ValueError("At least one of spectral_conv or linear must be enabled")

        # Determine dimensionality
        if isinstance(modes, int):
            # 1D case
            if use_spectral_conv:
                self.spectral_conv = SpectralConv1d(hidden_dim, hidden_dim, modes)
            else:
                self.spectral_conv = None

            if use_linear:
                self.linear = nn.Conv1d(hidden_dim, hidden_dim, 1)
            else:
                self.linear = None

            self.dim = 1
        elif len(modes) == 2:
            # 2D case
            if use_spectral_conv:
                self.spectral_conv = SpectralConv2d(hidden_dim, hidden_dim, modes)
            else:
                self.spectral_conv = None

            if use_linear:
                self.linear = nn.Conv2d(hidden_dim, hidden_dim, 1)
            else:
                self.linear = None

            self.dim = 2
        else:
            raise ValueError(f"Unsupported modes shape: {modes}")

        # Activation function
        activation_fn: nn.Module
        if activation == 'gelu':
            activation_fn = nn.GELU()
        elif activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'silu' or activation == 'swish':
            activation_fn = nn.SiLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        elif activation == 'sigmoid':
            activation_fn = nn.Sigmoid()
        elif activation == 'identity':
            activation_fn = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.activation = activation_fn

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        if self.linear is not None:
            nn.init.xavier_uniform_(self.linear.weight)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier Neural Operator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape depends on dimensionality:
            - 1D: (batch_size, sequence_length, hidden_dim)
            - 2D: (batch_size, height, width, hidden_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of same shape as input.
        """
        # Ensure all layers match input dtype for proper dtype preservation
        input_dtype = x.dtype
        if self.linear is not None and self.linear.weight.dtype != input_dtype:
            self.linear = self.linear.to(input_dtype)

        if self.dim == 1:
            # For 1D, expect (batch, sequence, channels)
            # Transpose to (batch, channels, sequence) for convolution
            x = x.transpose(-1, -2)

            # Apply spectral convolution and/or linear transformation
            out = torch.zeros_like(x)
            if self.spectral_conv is not None:
                out = out + self.spectral_conv(x)
            if self.linear is not None:
                out = out + self.linear(x)

            # Apply activation
            out = self.activation(out)

            # Transpose back
            out = out.transpose(-1, -2)

        elif self.dim == 2:
            # For 2D, expect (batch, height, width, channels)
            # Permute to (batch, channels, height, width)
            x = x.permute(0, 3, 1, 2)

            # Apply spectral convolution and/or linear transformation
            out = torch.zeros_like(x)
            if self.spectral_conv is not None:
                out = out + self.spectral_conv(x)
            if self.linear is not None:
                out = out + self.linear(x)

            # Apply activation
            out = self.activation(out)

            # Permute back
            out = out.permute(0, 2, 3, 1)

        return out

    @property
    def complexity(self) -> dict[str, str]:
        """Computational complexity of FNO.

        Returns
        -------
        dict[str, str]
            Time and space complexity strings.
        """
        if isinstance(self.modes, int):
            # 1D case
            return {
                'time': 'O(n*d*log(n) + k*d^2)',
                'space': 'O(k*d^2)'
            }
        else:
            # 2D case
            return {
                'time': 'O(n*m*d*log(n*m) + k1*k2*d^2)',
                'space': 'O(k1*k2*d^2)'
            }


class FNOBlock(SpectralComponent):
    """Complete FNO block with spectral convolution and feedforward network.

    This block combines the FNO layer with layer normalization, residual
    connections, and an optional feedforward network for a complete
    transformer-like block.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension size.
    modes : int | tuple[int, ...], optional
        Number of Fourier modes to retain. Default is 16.
    mlp_ratio : float, optional
        Expansion ratio for feedforward network. Default is 2.0.
    activation : str, optional
        Activation function. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    norm_type : str, optional
        Normalization type: 'layer' or 'batch'. Default is 'layer'.

    Attributes
    ----------
    hidden_dim : int
        Hidden dimension size.
    fno : FourierNeuralOperator
        FNO layer for spectral convolution.
    norm1 : nn.Module
        First normalization layer.
    norm2 : nn.Module | None
        Second normalization layer (if FFN is used).
    ffn : nn.Sequential | None
        Feedforward network.
    dropout : nn.Dropout
        Dropout layer.

    Examples
    --------
    >>> block = FNOBlock(hidden_dim=64, modes=16, mlp_ratio=2.0)
    >>> x = torch.randn(32, 128, 64)
    >>> output = block(x)
    >>> assert output.shape == x.shape
    """

    def __init__(
        self,
        hidden_dim: int,
        modes: int | tuple[int, ...] = 16,
        mlp_ratio: float = 2.0,
        activation: ActivationType = 'gelu',
        dropout: float = 0.0,
        norm_type: NormType = 'layernorm'
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # FNO layer
        self.fno = FourierNeuralOperator(
            hidden_dim=hidden_dim,
            modes=modes,
            activation=activation
        )

        # Normalization
        self.norm1: nn.Module | None
        self.norm2: nn.Module | None
        self.ffn: nn.Sequential | None
        if norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim) if mlp_ratio > 0 else None
        elif norm_type == 'batchnorm':
            self.norm1 = nn.BatchNorm1d(hidden_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim) if mlp_ratio > 0 else None
        elif norm_type == 'none':
            self.norm1 = None
            self.norm2 = None
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # Feedforward network
        if mlp_ratio > 0:
            mlp_hidden = int(hidden_dim * mlp_ratio)
            activation_fn: nn.Module
            if activation == 'gelu':
                activation_fn = nn.GELU()
            elif activation == 'relu':
                activation_fn = nn.ReLU()
            elif activation == 'silu' or activation == 'swish':
                activation_fn = nn.SiLU()
            elif activation == 'tanh':
                activation_fn = nn.Tanh()
            elif activation == 'sigmoid':
                activation_fn = nn.Sigmoid()
            elif activation == 'identity':
                activation_fn = nn.Identity()
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden),
                activation_fn,
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, hidden_dim),
                nn.Dropout(dropout)
            )
        else:
            self.ffn = None
            self.norm2 = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FNO block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of same shape as input.
        """
        # FNO with residual connection
        if self.norm1 is not None:
            if isinstance(self.norm1, nn.BatchNorm1d):
                # BatchNorm expects (batch, channels, length)
                x_norm = x.transpose(1, 2)
                x_norm = self.norm1(x_norm)
                x_norm = x_norm.transpose(1, 2)
            else:
                x_norm = self.norm1(x)
        else:
            x_norm = x

        x = x + self.dropout(self.fno(x_norm))

        # Feedforward network with residual connection
        if self.ffn is not None:
            if self.norm2 is not None:
                if isinstance(self.norm2, nn.BatchNorm1d):
                    x_norm = x.transpose(1, 2)
                    x_norm = self.norm2(x_norm)
                    x_norm = x_norm.transpose(1, 2)
                else:
                    x_norm = self.norm2(x)
            else:
                x_norm = x

            x = x + self.ffn(x_norm)

        return x

    @property
    def complexity(self) -> dict[str, str]:
        """Computational complexity of FNO block.

        Returns
        -------
        dict[str, str]
            Time and space complexity strings.
        """
        return self.fno.complexity
