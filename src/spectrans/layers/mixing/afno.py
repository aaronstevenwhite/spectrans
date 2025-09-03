"""Adaptive Fourier Neural Operator (AFNO) mixing layer implementation.

This module provides the AFNO mixing layer, which performs token mixing in the Fourier
domain with adaptive mode truncation and learnable spectral filters. AFNO efficiently
processes sequence data by operating on truncated Fourier modes, significantly reducing
computational complexity while maintaining expressive power.

The AFNO architecture leverages the sparsity of signals in the frequency domain,
applying learnable transformations to the most significant Fourier modes while
discarding higher-frequency components that often contain noise.

Classes
-------
AFNOMixing
    Adaptive Fourier Neural Operator mixing layer with mode truncation.

Examples
--------
Basic AFNO mixing layer:

>>> import torch
>>> from spectrans.layers.mixing.afno import AFNOMixing
>>> layer = AFNOMixing(hidden_dim=768, max_sequence_length=512)
>>> x = torch.randn(32, 512, 768)
>>> output = layer(x)
>>> assert output.shape == x.shape

With custom mode truncation:

>>> # Keep only 25% of Fourier modes
>>> layer = AFNOMixing(
...     hidden_dim=768,
...     max_sequence_length=512,
...     modes_seq=128,  # Keep 128 modes in sequence dimension
...     modes_hidden=384  # Keep 384 modes in hidden dimension
... )

Notes
-----
The AFNO mixing operation follows the mathematical formulation:
1. Apply 2D FFT to input tensor (treating sequence and hidden dims as spatial dims)
2. Truncate to keep only low-frequency modes
3. Apply learnable MLP to truncated modes
4. Zero-pad back to original size and apply inverse FFT
5. Add residual connection

The mode truncation significantly reduces memory and computation requirements,
making AFNO efficient for long sequences.

References
----------
.. [1] Guibas, J., et al., "Adaptive Fourier Neural Operators: Efficient Token
       Mixers for Transformers", ICLR 2022.

See Also
--------
spectrans.layers.mixing.fourier : Standard Fourier mixing without mode truncation.
spectrans.layers.operators.fno : Fourier Neural Operator implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectrans.core.types import ActivationType
from spectrans.layers.mixing.base import MixingLayer


class AFNOMixing(MixingLayer):
    """Adaptive Fourier Neural Operator mixing layer.

    This layer performs efficient token mixing by applying learnable transformations
    in the truncated Fourier domain, significantly reducing computational cost
    while maintaining model expressiveness.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the input/output tensors.
    max_sequence_length : int
        Maximum sequence length the model will process.
    modes_seq : int | None, optional
        Number of Fourier modes to keep in sequence dimension.
        If None, defaults to max_sequence_length // 2.
    modes_hidden : int | None, optional
        Number of Fourier modes to keep in hidden dimension.
        If None, defaults to hidden_dim // 2.
    mlp_ratio : float, optional
        Expansion ratio for the MLP in Fourier domain. Default is 2.0.
    activation : str, optional
        Activation function for MLP. Default is 'gelu'.
    dropout : float, optional
        Dropout probability for MLP. Default is 0.0.

    Attributes
    ----------
    hidden_dim : int
        Hidden dimension size.
    max_sequence_length : int
        Maximum supported sequence length.
    modes_seq : int
        Number of retained Fourier modes in sequence dimension.
    modes_hidden : int
        Number of retained Fourier modes in hidden dimension.
    mlp_ratio : float
        MLP expansion ratio.
    fourier_weight : nn.Parameter
        Complex-valued learnable weights for Fourier modes.
    mlp : nn.Sequential
        MLP applied in Fourier domain.

    Examples
    --------
    >>> import torch
    >>> layer = AFNOMixing(hidden_dim=768, max_sequence_length=512, modes_seq=128)
    >>> x = torch.randn(32, 512, 768)
    >>> output = layer(x)
    >>> print(output.shape)
    torch.Size([32, 512, 768])
    """

    def __init__(
        self,
        hidden_dim: int,
        max_sequence_length: int,
        modes_seq: int | None = None,
        modes_hidden: int | None = None,
        mlp_ratio: float = 2.0,
        activation: ActivationType = 'gelu',
        dropout: float = 0.0
    ):
        super().__init__(hidden_dim=hidden_dim, dropout=dropout)

        self.max_sequence_length = max_sequence_length

        # Set default mode truncation if not specified
        self.modes_seq = modes_seq if modes_seq is not None else max_sequence_length // 2
        self.modes_hidden = modes_hidden if modes_hidden is not None else hidden_dim // 2

        # Ensure modes don't exceed actual dimensions (for rfft)
        # For rfft, the last dimension has size n//2 + 1
        self.modes_seq = min(self.modes_seq, max_sequence_length)
        self.modes_hidden = min(self.modes_hidden, hidden_dim // 2 + 1)

        self.mlp_ratio = mlp_ratio

        # Complex-valued learnable weights for Fourier modes
        # We use real FFT, so last dimension is reduced
        scale = 1 / (self.modes_seq * self.modes_hidden)
        self.fourier_weight = nn.Parameter(
            torch.randn(self.modes_seq, self.modes_hidden, 2) * scale
        )

        # MLP in Fourier domain
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)

        # Activation function
        activation_fn: nn.Module
        if activation == 'gelu':
            activation_fn = nn.GELU()
        elif activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'silu':
            activation_fn = nn.SiLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        elif activation == 'sigmoid':
            activation_fn = nn.Sigmoid()
        elif activation == 'identity':
            activation_fn = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # MLP operates on real and imaginary parts concatenated
        self.mlp = nn.Sequential(
            nn.Linear(self.modes_seq * self.modes_hidden * 2, mlp_hidden_dim),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, self.modes_seq * self.modes_hidden * 2),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with appropriate scaling."""
        # MLP initialization
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply AFNO mixing to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of same shape as input.
        """
        batch_size, seq_len, hidden_dim = x.shape
        residual = x
        input_dtype = x.dtype

        # Convert to float32 for processing if needed
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            residual = residual.to(torch.float32)

        # Apply layer norm
        x = self.norm(x)

        # Pad if necessary to match max_sequence_length
        if seq_len < self.max_sequence_length:
            padding = self.max_sequence_length - seq_len
            x = F.pad(x, (0, 0, 0, padding))

        # Step 1: Transform to Fourier space using 2D FFT
        # Treat (sequence, hidden) as 2D spatial dimensions
        x_ft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # Step 2: Mode truncation - keep only low-frequency modes
        x_ft_truncated = x_ft[:, :self.modes_seq, :self.modes_hidden]

        # Step 3: Apply learnable transformation in Fourier domain
        # First apply pointwise multiplication with learnable weights
        weight_complex = torch.view_as_complex(self.fourier_weight)
        x_ft_truncated = x_ft_truncated * weight_complex

        # Flatten for MLP processing
        batch_size_ft = x_ft_truncated.shape[0]
        x_ft_flat = torch.view_as_real(x_ft_truncated).reshape(batch_size_ft, -1)

        # Apply MLP
        x_ft_flat = self.mlp(x_ft_flat)

        # Reshape back to complex truncated form
        x_ft_truncated = x_ft_flat.reshape(
            batch_size_ft, self.modes_seq, self.modes_hidden, 2
        )
        x_ft_truncated = torch.view_as_complex(x_ft_truncated)

        # Step 4: Zero-pad back to original size
        x_ft_padded = torch.zeros(
            (batch_size, self.max_sequence_length, hidden_dim // 2 + 1),
            dtype=x_ft.dtype,
            device=x_ft.device
        )
        x_ft_padded[:, :self.modes_seq, :self.modes_hidden] = x_ft_truncated

        # Step 5: Inverse FFT to get back to spatial domain
        x_spatial = torch.fft.irfft2(
            x_ft_padded,
            s=(self.max_sequence_length, hidden_dim),
            dim=(1, 2),
            norm='ortho'
        )

        # Remove padding if it was added
        if seq_len < self.max_sequence_length:
            x_spatial = x_spatial[:, :seq_len, :]

        # Step 6: Add residual connection
        output = residual + x_spatial

        # Convert back to original dtype if needed
        if output.dtype != input_dtype:
            output = output.to(input_dtype)

        return output  # type: ignore[no-any-return]

    @property
    def complexity(self) -> dict[str, str]:
        """Computational complexity of AFNO mixing.

        Returns
        -------
        dict[str, str]
            Time and space complexity strings.
        """
        k_n = self.modes_seq
        k_d = self.modes_hidden

        return {
            'time': f'O({k_n}*{k_d}*d + n*d*log(n*d))',
            'space': f'O({k_n}*{k_d} + n*d)'
        }

    def get_spectral_properties(self) -> dict[str, bool]:
        """Get mathematical properties of AFNO operation.

        Returns
        -------
        dict[str, bool]
            Mathematical properties of the transform.
        """
        return {
            'unitary': False,  # Not unitary due to mode truncation and MLP
            'real_output': True,  # Output is real-valued
            'frequency_domain': True,  # Operations in Fourier domain
            'energy_preserving': False,  # Energy not preserved due to truncation
            'learnable_parameters': True,  # Has learnable weights and MLP
            'translation_equivariant': False,  # Not equivariant due to MLP
            'mode_truncation': True,  # Uses Fourier mode truncation
            'adaptive': True  # Adaptive filtering based on learned parameters
        }
