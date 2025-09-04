"""Fourier-based mixing layers for spectral transformers.

This module implements Fourier-based token mixing mechanisms, including the
FNet architecture that replaces attention with two-dimensional Fourier transforms.
The implementations follow the mathematical formulations from the original papers
while providing efficient PyTorch implementations optimized for modern hardware.

The core idea is to perform mixing in the frequency domain using Fast Fourier
Transforms, which provides O(n log n) complexity compared to O(n²) for attention
while maintaining competitive performance on many sequence modeling tasks.

Classes
-------
FourierMixing
    Basic FNet-style Fourier mixing with 2D FFT operations.
FourierMixing1D
    1D Fourier mixing along sequence dimension only.
RealFourierMixing
    Memory-efficient variant using real FFT for real-valued inputs.

Examples
--------
Basic FNet-style mixing:

>>> import torch
>>> from spectrans.layers.mixing.fourier import FourierMixing
>>> mixer = FourierMixing(hidden_dim=768)
>>> input_seq = torch.randn(32, 512, 768)  # (batch, seq_len, hidden)
>>> output = mixer(input_seq)
>>> assert output.shape == input_seq.shape

Memory-efficient real variant:

>>> from spectrans.layers.mixing.fourier import RealFourierMixing
>>> real_mixer = RealFourierMixing(hidden_dim=768, use_real_fft=True)
>>> output_real = real_mixer(input_seq)

1D sequence mixing only:

>>> from spectrans.layers.mixing.fourier import FourierMixing1D
>>> seq_mixer = FourierMixing1D(hidden_dim=768)
>>> output_1d = seq_mixer(input_seq)

Notes
-----
Mathematical Foundation:

The FNet mixing operation is defined as:
    FourierMix(X) = Re(F_d^(-1)(F_n(X)))

Where:
- F_n: 1D DFT along sequence dimension (n)
- F_d^(-1): Inverse 1D DFT along feature dimension (d)
- Re(·): Real part extraction

This can be efficiently implemented using PyTorch's 2D FFT:
    FourierMix(X) = Re(fft2d(X, dim=(-2, -1)))

Complexity Analysis:
- Time: O(nd log n + nd log d) ≈ O(nd log(nd))
- Space: O(nd) for storing frequency domain representations

The real FFT variant exploits Hermitian symmetry for ~2x memory and
computational savings when inputs are real-valued.

Advantages:
- Linear complexity in sequence length (vs quadratic for attention)
- No learnable parameters (reduces overfitting risk)
- Translation equivariance in both sequence and feature dimensions
- Excellent parallelization properties

Limitations:
- No content-based interactions (purely positional mixing)
- May struggle with tasks requiring precise positional reasoning
- Real part extraction can lose information

See Also
--------
spectrans.layers.mixing.base : Base classes for mixing operations
spectrans.transforms.fourier : Underlying FFT transform implementations
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ...config.layers.mixing import FourierMixingConfig

from ...core.registry import register_component
from ...core.types import FFTNorm
from ...transforms.fourier import FFT1D, FFT2D, RFFT, RFFT2D
from .base import UnitaryMixingLayer


@register_component("mixing", "fourier")
class FourierMixing(UnitaryMixingLayer):
    """FNet-style Fourier mixing layer.

    Implements the core FNet mixing operation using 2D Fourier transforms
    along both sequence and feature dimensions. This provides an efficient
    alternative to attention with O(n log n) complexity.

    The operation performs:
    1. 2D FFT across sequence and feature dimensions
    2. Real part extraction for final output

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the input tensors.
    dropout : float, default=0.0
        Dropout probability applied after the mixing operation.
    norm_eps : float, default=1e-5
        Epsilon for numerical stability.
    energy_tolerance : float, default=1e-4
        Tolerance for energy preservation verification.
    fft_norm : str, default="ortho"
        Normalization mode for FFT operations ("forward", "backward", "ortho").

    Attributes
    ----------
    fft2d : FFT2D
        2D Fourier transform module.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        energy_tolerance: float = 1e-4,
        fft_norm: FFTNorm = "ortho",
    ):
        super().__init__(hidden_dim, dropout, norm_eps, energy_tolerance)
        # Store transform as non-module attribute to avoid PyTorch module registration
        self.fft2d: FFT2D  # Type annotation for mypy
        object.__setattr__(self, 'fft2d', FFT2D(norm=fft_norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier mixing to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        torch.Tensor
            Mixed tensor of same shape, with real values only.
        """
        # Apply 2D FFT along last two dimensions (sequence and feature)
        x_freq = self.fft2d.transform(x, dim=(-2, -1))

        # Take real part as in original FNet
        x_mixed = torch.real(x_freq)

        # Apply dropout
        x_mixed = self.dropout(x_mixed)

        return x_mixed  # type: ignore[no-any-return]

    @property
    def complexity(self) -> dict[str, str]:
        """Return computational complexity.

        Returns
        -------
        dict[str, str]
            Time and space complexity in Big O notation.
        """
        return {
            'time': 'O(nd log n + nd log d)',
            'space': 'O(nd)'
        }

    def get_spectral_properties(self) -> dict[str, str | bool]:
        """Get spectral properties of Fourier mixing.

        Returns
        -------
        dict[str, str | bool]
            Properties including energy preservation and domain information.
        """
        return {
            'unitary': False,  # Real part extraction breaks unitarity
            'real_output': True,
            'frequency_domain': True,
            'energy_preserving': False,  # Real part extraction changes energy
            'translation_equivariant': True,
            'learnable_parameters': False,
        }

    @classmethod
    def from_config(cls, config: "FourierMixingConfig") -> "FourierMixing":
        """Create FourierMixing layer from configuration.

        Parameters
        ----------
        config : FourierMixingConfig
            Configuration object with layer parameters.

        Returns
        -------
        FourierMixing
            Configured Fourier mixing layer.
        """
        return cls(
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            norm_eps=config.norm_eps,
            energy_tolerance=config.energy_tolerance,
            fft_norm=config.fft_norm,
        )


@register_component("mixing", "fourier_1d")
class FourierMixing1D(UnitaryMixingLayer):
    """1D Fourier mixing along sequence dimension only.

    Applies Fourier transform only along the sequence dimension,
    preserving feature dimension locality while mixing tokens.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the input tensors.
    dropout : float, default=0.0
        Dropout probability applied after the mixing operation.
    norm_eps : float, default=1e-5
        Epsilon for numerical stability.
    energy_tolerance : float, default=1e-4
        Tolerance for energy preservation verification.
    fft_norm : str, default="ortho"
        Normalization mode for FFT operations.

    Attributes
    ----------
    fft1d : FFT1D
        1D Fourier transform module.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        energy_tolerance: float = 1e-4,
        fft_norm: FFTNorm = "ortho",
    ):
        super().__init__(hidden_dim, dropout, norm_eps, energy_tolerance)
        # Store transform as non-module attribute to avoid PyTorch module registration
        self.fft1d: FFT1D  # Type annotation for mypy
        object.__setattr__(self, 'fft1d', FFT1D(norm=fft_norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1D Fourier mixing to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        torch.Tensor
            Mixed tensor with Fourier transform applied along sequence dimension.
        """
        # Apply 1D FFT along sequence dimension only
        x_freq = self.fft1d.transform(x, dim=1)  # sequence dimension

        # Take real part
        x_mixed = torch.real(x_freq)

        # Apply dropout
        x_mixed = self.dropout(x_mixed)

        return x_mixed  # type: ignore[no-any-return]

    @property
    def complexity(self) -> dict[str, str]:
        """Return computational complexity.

        Returns
        -------
        dict[str, str]
            Time and space complexity in Big O notation.
        """
        return {
            'time': 'O(nd log n)',
            'space': 'O(nd)'
        }

    def get_spectral_properties(self) -> dict[str, str | bool]:
        """Get spectral properties of 1D Fourier mixing.

        Returns
        -------
        dict[str, str | bool]
            Properties specific to 1D sequence mixing.
        """
        return {
            'unitary': False,  # Real part extraction breaks unitarity
            'real_output': True,
            'frequency_domain': True,
            'energy_preserving': False,
            'sequence_mixing_only': True,
            'feature_preserving': True,
            'learnable_parameters': False,
        }


@register_component("mixing", "real_fourier")
class RealFourierMixing(UnitaryMixingLayer):
    """Memory-efficient real Fourier mixing.

    Uses real FFT operations to exploit Hermitian symmetry,
    providing ~2x memory and computational savings for real inputs.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the input tensors.
    use_real_fft : bool, default=True
        Whether to use real FFT for efficiency.
    dropout : float, default=0.0
        Dropout probability applied after mixing.
    norm_eps : float, default=1e-5
        Epsilon for numerical stability.
    energy_tolerance : float, default=1e-4
        Tolerance for energy preservation verification.
    fft_norm : str, default="ortho"
        Normalization mode for FFT operations.

    Attributes
    ----------
    use_real_fft : bool
        Whether real FFT is enabled.
    rfft : RFFT
        Real FFT transform for sequence dimension.
    rfft2d : RFFT2D
        Real 2D FFT transform for both dimensions.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_real_fft: bool = True,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        energy_tolerance: float = 1e-4,
        fft_norm: FFTNorm = "ortho",
    ):
        super().__init__(hidden_dim, dropout, norm_eps, energy_tolerance)
        self.use_real_fft = use_real_fft

        if use_real_fft:
            # Type annotations for mypy
            self.rfft: RFFT
            self.rfft2d: RFFT2D
            # Store transforms as non-module attributes
            object.__setattr__(self, 'rfft', RFFT(norm=fft_norm))
            object.__setattr__(self, 'rfft2d', RFFT2D(norm=fft_norm))
        else:
            # Type annotation for mypy
            self.fft2d: FFT2D
            # Fallback to complex FFT
            object.__setattr__(self, 'fft2d', FFT2D(norm=fft_norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply real Fourier mixing to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).
            Should be real-valued for optimal efficiency.

        Returns
        -------
        torch.Tensor
            Mixed tensor, guaranteed to be real-valued.
        """
        if self.use_real_fft and torch.is_floating_point(x):
            # Use real FFT for efficiency
            x_freq = self.rfft2d.transform(x, dim=(-2, -1))
            # Inverse RFFT automatically returns real values
            x_mixed = self.rfft2d.inverse_transform(x_freq, dim=(-2, -1))
        else:
            # Fallback to complex FFT with real part extraction
            x_freq = self.fft2d.transform(x, dim=(-2, -1))
            x_mixed = torch.real(x_freq)

        # Apply dropout
        x_mixed = self.dropout(x_mixed)

        return x_mixed  # type: ignore[no-any-return]

    @property
    def complexity(self) -> dict[str, str]:
        """Return computational complexity.

        Returns
        -------
        dict[str, str]
            Time and space complexity, with efficiency gains noted.
        """
        if self.use_real_fft:
            return {
                'time': 'O(nd log n + nd log d) [~2x faster for real inputs]',
                'space': 'O(nd) [~2x less memory for real inputs]'
            }
        else:
            return {
                'time': 'O(nd log n + nd log d)',
                'space': 'O(nd)'
            }

    def get_spectral_properties(self) -> dict[str, str | bool]:
        """Get spectral properties of real Fourier mixing.

        Returns
        -------
        dict[str, str | bool]
            Properties including efficiency characteristics.
        """
        return {
            'unitary': self.use_real_fft,  # Real FFT preserves unitarity
            'real_output': True,
            'frequency_domain': True,
            'energy_preserving': self.use_real_fft,
            'memory_efficient': self.use_real_fft,
            'hermitian_symmetry': self.use_real_fft,
            'learnable_parameters': False,
        }


@register_component("mixing", "fourier_separable")
class SeparableFourierMixing(UnitaryMixingLayer):
    """Separable Fourier mixing with sequence and feature transforms.

    Applies separate 1D Fourier transforms along sequence and feature
    dimensions, which can be more efficient than 2D FFT for certain
    tensor shapes and provides different mixing characteristics.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the input tensors.
    mix_features : bool, default=True
        Whether to apply FFT along feature dimension.
    mix_sequence : bool, default=True
        Whether to apply FFT along sequence dimension.
    dropout : float, default=0.0
        Dropout probability.
    norm_eps : float, default=1e-5
        Epsilon for numerical stability.
    energy_tolerance : float, default=1e-4
        Tolerance for energy preservation verification.
    fft_norm : str, default="ortho"
        FFT normalization mode.

    Attributes
    ----------
    mix_features : bool
        Whether feature mixing is enabled.
    mix_sequence : bool
        Whether sequence mixing is enabled.
    fft1d : FFT1D
        1D FFT transform module.
    """

    def __init__(
        self,
        hidden_dim: int,
        mix_features: bool = True,
        mix_sequence: bool = True,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        energy_tolerance: float = 1e-4,
        fft_norm: FFTNorm = "ortho",
    ):
        super().__init__(hidden_dim, dropout, norm_eps, energy_tolerance)
        self.mix_features = mix_features
        self.mix_sequence = mix_sequence
        # Store transform as non-module attribute
        self.fft1d: FFT1D  # Type annotation for mypy
        object.__setattr__(self, 'fft1d', FFT1D(norm=fft_norm))

        if not mix_features and not mix_sequence:
            raise ValueError("At least one of mix_features or mix_sequence must be True")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply separable Fourier mixing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        torch.Tensor
            Mixed tensor after applying selected transforms.
        """
        # Apply sequence mixing (along dim=1)
        if self.mix_sequence:
            x_freq_seq = self.fft1d.transform(x, dim=1)
            x = torch.real(x_freq_seq)

        # Apply feature mixing (along dim=2)
        if self.mix_features:
            x_freq_feat = self.fft1d.transform(x, dim=2)
            x = torch.real(x_freq_feat)

        # Apply dropout
        x = self.dropout(x)

        return x

    @property
    def complexity(self) -> dict[str, str]:
        """Return computational complexity.

        Returns
        -------
        dict[str, str]
            Complexity depends on which dimensions are mixed.
        """
        time_components = []
        if self.mix_sequence:
            time_components.append("nd log n")
        if self.mix_features:
            time_components.append("nd log d")

        time_complexity = "O(" + " + ".join(time_components) + ")"

        return {
            'time': time_complexity,
            'space': 'O(nd)'
        }

    def get_spectral_properties(self) -> dict[str, str | bool]:
        """Get properties of separable mixing.

        Returns
        -------
        dict[str, str | bool]
            Properties reflecting the separable nature.
        """
        return {
            'unitary': False,  # Real part extraction
            'real_output': True,
            'frequency_domain': True,
            'energy_preserving': False,
            'separable': True,
            'sequence_mixing': self.mix_sequence,
            'feature_mixing': self.mix_features,
            'learnable_parameters': False,
        }


__all__: list[str] = [
    "FourierMixing",
    "FourierMixing1D",
    "RealFourierMixing",
    "SeparableFourierMixing",
]
