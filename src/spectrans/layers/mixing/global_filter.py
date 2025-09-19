r"""Global Filter Networks (GFNet) mixing layers.

Implements Global Filter Network mixing layers that apply learnable complex-valued
filters in the frequency domain. GFNet provides an alternative to attention by
performing element-wise filtering operations in the Fourier domain, maintaining
$O(n \log n)$ complexity while introducing learnable parameters.

Learnable complex filters can selectively emphasize or suppress different frequency
components, providing more modeling flexibility than parameter-free Fourier mixing
while maintaining computational complexity.

Classes
-------
GlobalFilterMixing
    Basic GFNet global filter with learnable complex filters.
GlobalFilterMixing2D
    2D variant applying filters to both sequence and feature dimensions.
AdaptiveGlobalFilter
    Advanced variant with adaptive filter initialization and regularization.

Examples
--------
Basic global filter usage:

>>> import torch
>>> from spectrans.layers.mixing.global_filter import GlobalFilterMixing
>>> filter_layer = GlobalFilterMixing(hidden_dim=768, sequence_length=512)
>>> input_seq = torch.randn(32, 512, 768)
>>> output = filter_layer(input_seq)
>>> assert output.shape == input_seq.shape

2D global filtering:

>>> from spectrans.layers.mixing.global_filter import GlobalFilterMixing2D
>>> filter_2d = GlobalFilterMixing2D(hidden_dim=768, sequence_length=512)
>>> output_2d = filter_2d(input_seq)

Adaptive filtering with regularization:

>>> from spectrans.layers.mixing.global_filter import AdaptiveGlobalFilter
>>> adaptive_filter = AdaptiveGlobalFilter(
...     hidden_dim=768, sequence_length=512,
...     filter_regularization=0.01, adaptive_initialization=True
... )
>>> output_adaptive = adaptive_filter(input_seq)

Notes
-----
Mathematical Foundation:

The Global Filter operation is defined as:
$$
\text{GF}(\mathbf{X}) = \mathcal{F}^{-1}(\mathbf{H} \odot \mathcal{F}(\mathbf{X}))
$$

Where $\mathcal{F}$ is FFT along sequence dimension, $\mathcal{F}^{-1}$ is inverse FFT,
$\mathbf{H} \in \mathbb{C}^{n \times d}$ is a learnable complex filter, and $\odot$ denotes
element-wise (Hadamard) multiplication.

The complex filter is parameterized as:
$$
\mathbf{H} = \sigma(\mathbf{W}_r + i\mathbf{W}_i)
$$

Where $\mathbf{W}_r, \mathbf{W}_i \in \mathbb{R}^{n \times d}$ are real-valued learnable parameters,
$\sigma$ is an activation function (typically sigmoid or tanh), and $i$ is the imaginary unit.

Sigmoid activation provides soft gating with values in $(0,1)$. Tanh provides symmetric
activation with values in $(-1,1)$. Identity activation has no transformation but may be
unstable.

Time complexity is $O(nd \log n)$ for FFT operations. Space complexity is $O(nd)$ for filter
parameters and frequency representations. The model uses $2nd$ real parameters ($\mathbf{W}_r$ and
$\mathbf{W}_i$).

Learnable parameters allow task-specific adaptation compared to FNet. Filters can emphasize
important frequencies and suppress noise while maintaining linear complexity with added
expressiveness. Filter initialization affects training stability. Regularization prevents
overfitting to specific frequencies. Activation choice impacts gradient flow and expressiveness.

References
----------
Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou. 2021.
Global filter networks for image classification. In Advances in Neural
Information Processing Systems 34 (NeurIPS 2021), pages 980-993.

See Also
--------
spectrans.layers.mixing.base : Base classes for mixing layers
spectrans.layers.mixing.fourier : Parameter-free Fourier mixing layers
spectrans.transforms.fourier : Underlying FFT implementations
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ...config.layers.mixing import GlobalFilterMixingConfig
import torch.nn as nn

from ...core.registry import register_component
from ...core.types import ActivationType, FFTNorm, Tensor
from ...transforms.fourier import FFT1D, FFT2D
from ...utils.complex import complex_multiply, make_complex
from .base import FilterMixingLayer


@register_component("mixing", "global_filter")
class GlobalFilterMixing(FilterMixingLayer):
    """Global Filter Network mixing layer.

    Implements the core GFNet mixing operation with learnable complex filters
    applied in the frequency domain along the sequence dimension.

    The layer uses interpolation to adapt filters to different sequence lengths,
    processing variable-length inputs while preserving learned frequency patterns.
    This provides resolution independence compared to fixed-size filtering.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of input tensors.
    sequence_length : int
        Base sequence length for filter parameter initialization. The filters
        will be interpolated to match actual input sequence lengths.
    activation : ActivationType, default="sigmoid"
        Activation function applied to filter parameters ("sigmoid", "tanh", "identity").
    dropout : float, default=0.0
        Dropout probability applied after filtering.
    norm_eps : float, default=1e-5
        Epsilon for numerical stability.
    learnable_filters : bool, default=True
        Whether filter parameters are learnable (always True for this class).
    fft_norm : str, default="ortho"
        FFT normalization mode.
    filter_init_std : float, default=0.02
        Standard deviation for filter parameter initialization.

    Attributes
    ----------
    activation : str
        Activation function name.
    filter_real : nn.Parameter
        Real part of complex filter parameters.
    filter_imag : nn.Parameter
        Imaginary part of complex filter parameters.
    fft1d : FFT1D
        1D FFT transform for sequence dimension.
    activation_fn : nn.Module
        Activation function module (Sigmoid, Tanh, or Identity).
    """

    def __init__(
        self,
        hidden_dim: int,
        sequence_length: int,
        activation: ActivationType = "sigmoid",
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        learnable_filters: bool = True,
        fft_norm: FFTNorm = "ortho",
        filter_init_std: float = 0.02,
    ):
        super().__init__(hidden_dim, sequence_length, dropout, norm_eps, learnable_filters)
        self.activation = activation

        # Initialize complex filter parameters
        self.filter_real = nn.Parameter(torch.randn(sequence_length, hidden_dim) * filter_init_std)
        self.filter_imag = nn.Parameter(torch.randn(sequence_length, hidden_dim) * filter_init_std)

        # Store FFT transform as non-module attribute
        self.fft1d: FFT1D  # Type annotation for mypy
        object.__setattr__(self, "fft1d", FFT1D(norm=fft_norm))

        # Activation function
        if activation == "sigmoid":
            self.activation_fn: Callable[[Tensor], Tensor] = nn.Sigmoid()
        elif activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation == "identity":
            self.activation_fn = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        """Apply global filtering to input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        Tensor
            Filtered tensor of same shape as input.
        """
        # Transform to frequency domain
        x_freq = self.fft1d.transform(x, dim=1)  # Along sequence dimension

        # Get actual sequence length
        seq_len = x_freq.shape[1]

        # Adapt filter to actual sequence length using interpolation
        if seq_len != self.sequence_length:
            # Use interpolation to adapt filters to the actual sequence length
            # This preserves the learned frequency patterns at different resolutions
            filter_real = (
                nn.functional.interpolate(
                    self.filter_real.T.unsqueeze(0),  # (1, hidden_dim, sequence_length)
                    size=seq_len,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .T
            )  # (seq_len, hidden_dim)

            filter_imag = (
                nn.functional.interpolate(
                    self.filter_imag.T.unsqueeze(0),  # (1, hidden_dim, sequence_length)
                    size=seq_len,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .T
            )  # (seq_len, hidden_dim)
        else:
            filter_real = self.filter_real
            filter_imag = self.filter_imag

        # Create complex filter
        filter_complex = make_complex(
            self.activation_fn(filter_real), self.activation_fn(filter_imag)
        )

        # Apply filter in frequency domain (element-wise multiplication)
        filtered_freq = complex_multiply(x_freq, filter_complex)

        # Transform back to time domain
        filtered_time = self.fft1d.inverse_transform(filtered_freq, dim=1)

        # Take real part (assuming real-valued output is desired)
        output = torch.real(filtered_time)

        # Apply dropout
        output = self.dropout(output)

        return output  # type: ignore[no-any-return]

    def get_filter_response(self) -> Tensor:
        """Get the current frequency response of the filters.

        Returns
        -------
        Tensor
            Complex-valued frequency response of shape (sequence_length, hidden_dim).
        """
        return make_complex(
            self.activation_fn(self.filter_real), self.activation_fn(self.filter_imag)
        )

    def get_spectral_properties(self) -> dict[str, str | bool | int]:
        """Get spectral properties of global filtering.

        Returns
        -------
        dict[str, str | bool | int]
            Properties including filter characteristics.
        """
        return {
            "frequency_domain": True,
            "learnable_filters": True,
            "complex_valued": True,
            "selective_filtering": True,
            "energy_preserving": False,  # Filtering can change energy
            "activation": self.activation,
            "parameter_count": 2 * self.sequence_length * self.hidden_dim,
        }

    @classmethod
    def from_config(cls, config: "GlobalFilterMixingConfig") -> "GlobalFilterMixing":
        """Create GlobalFilterMixing layer from configuration.

        Parameters
        ----------
        config : GlobalFilterMixingConfig
            Configuration object with layer parameters.

        Returns
        -------
        GlobalFilterMixing
            Configured global filter mixing layer.
        """
        return cls(
            hidden_dim=config.hidden_dim,
            sequence_length=config.sequence_length,
            activation=config.activation,
            dropout=config.dropout,
            learnable_filters=config.learnable_filters,
            fft_norm=config.fft_norm,
            filter_init_std=config.filter_init_std,
        )


@register_component("mixing", "global_filter_2d")
class GlobalFilterMixing2D(FilterMixingLayer):
    """2D Global Filter mixing with filtering along both dimensions.

    Extends global filtering to both sequence and feature dimensions,
    similar to FNet's 2D FFT but with learnable filters.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of input tensors.
    sequence_length : int
        Expected sequence length.
    activation : ActivationType, default="sigmoid"
        Activation function for filter parameters.
    dropout : float, default=0.0
        Dropout probability.
    norm_eps : float, default=1e-5
        Epsilon for numerical stability.
    learnable_filters : bool, default=True
        Whether filters are learnable.
    fft_norm : str, default="ortho"
        FFT normalization mode.
    filter_init_std : float, default=0.02
        Filter parameter initialization standard deviation.

    Attributes
    ----------
    filter_real : nn.Parameter
        Real part of 2D complex filters.
    filter_imag : nn.Parameter
        Imaginary part of 2D complex filters.
    fft2d : FFT2D
        2D FFT transform module.
    activation_fn : nn.Module
        Activation function.
    """

    def __init__(
        self,
        hidden_dim: int,
        sequence_length: int,
        activation: ActivationType = "sigmoid",
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        learnable_filters: bool = True,
        fft_norm: FFTNorm = "ortho",
        filter_init_std: float = 0.02,
    ):
        super().__init__(hidden_dim, sequence_length, dropout, norm_eps, learnable_filters)
        self.activation = activation

        # Initialize 2D complex filter parameters
        self.filter_real = nn.Parameter(torch.randn(sequence_length, hidden_dim) * filter_init_std)
        self.filter_imag = nn.Parameter(torch.randn(sequence_length, hidden_dim) * filter_init_std)

        # Store 2D FFT transform as non-module attribute
        self.fft2d: FFT2D  # Type annotation for mypy
        object.__setattr__(self, "fft2d", FFT2D(norm=fft_norm))

        # Activation function
        if activation == "sigmoid":
            self.activation_fn: Callable[[Tensor], Tensor] = nn.Sigmoid()
        elif activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation == "identity":
            self.activation_fn = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        """Apply 2D global filtering.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        Tensor
            Filtered tensor of same shape.
        """
        # Transform to 2D frequency domain
        x_freq = self.fft2d.transform(x, dim=(-2, -1))

        # Get actual dimensions
        seq_len = x_freq.shape[-2]
        hidden = x_freq.shape[-1]

        # Adapt filter to actual dimensions using bilinear interpolation
        if seq_len != self.sequence_length or hidden != self.hidden_dim:
            # Reshape for 2D interpolation
            filter_real = (
                nn.functional.interpolate(
                    self.filter_real.unsqueeze(0).unsqueeze(0),  # (1, 1, seq_length, hidden_dim)
                    size=(seq_len, hidden),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )  # (seq_len, hidden)

            filter_imag = (
                nn.functional.interpolate(
                    self.filter_imag.unsqueeze(0).unsqueeze(0),  # (1, 1, seq_length, hidden_dim)
                    size=(seq_len, hidden),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )  # (seq_len, hidden)
        else:
            filter_real = self.filter_real
            filter_imag = self.filter_imag

        # Create complex filter
        filter_complex = make_complex(
            self.activation_fn(filter_real), self.activation_fn(filter_imag)
        )

        # Apply 2D filter
        filtered_freq = complex_multiply(x_freq, filter_complex)

        # Transform back to spatial domain
        filtered_spatial = self.fft2d.inverse_transform(filtered_freq, dim=(-2, -1))

        # Take real part
        output = torch.real(filtered_spatial)

        # Apply dropout
        output = self.dropout(output)

        return output  # type: ignore[no-any-return]

    def get_filter_response(self) -> Tensor:
        """Get 2D frequency response.

        Returns
        -------
        Tensor
            Complex 2D frequency response.
        """
        return make_complex(
            self.activation_fn(self.filter_real), self.activation_fn(self.filter_imag)
        )

    def get_spectral_properties(self) -> dict[str, str | bool | int]:
        """Get 2D filter properties.

        Returns
        -------
        dict[str, str | bool | int]
            2D filtering characteristics.
        """
        return {
            "frequency_domain": True,
            "learnable_filters": True,
            "complex_valued": True,
            "selective_filtering": True,
            "energy_preserving": False,
            "two_dimensional": True,
            "activation": self.activation,
            "parameter_count": 2 * self.sequence_length * self.hidden_dim,
        }


@register_component("mixing", "adaptive_global_filter")
class AdaptiveGlobalFilter(FilterMixingLayer):
    """Adaptive Global Filter with regularization and smart initialization.

    Enhanced version of global filtering with adaptive initialization
    strategies, regularization options, and improved training stability.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of input tensors.
    sequence_length : int
        Expected sequence length.
    activation : ActivationType, default="sigmoid"
        Filter activation function.
    dropout : float, default=0.0
        Dropout probability.
    norm_eps : float, default=1e-5
        Numerical stability epsilon.
    learnable_filters : bool, default=True
        Whether filters are learnable.
    fft_norm : str, default="ortho"
        FFT normalization.
    filter_init_std : float, default=0.02
        Filter initialization standard deviation.
    filter_regularization : float, default=0.0
        L2 regularization strength for filter parameters.
    adaptive_initialization : bool, default=True
        Whether to use frequency-aware initialization.
    spectral_dropout_p : float, default=0.0
        Spectral dropout probability in frequency domain.

    Attributes
    ----------
    filter_regularization : float
        Regularization strength.
    adaptive_initialization : bool
        Whether adaptive initialization is used.
    spectral_dropout_p : float
        Spectral dropout probability.
    spectral_dropout : nn.Module
        Spectral dropout layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        sequence_length: int,
        activation: ActivationType = "sigmoid",
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        learnable_filters: bool = True,
        fft_norm: FFTNorm = "ortho",
        filter_init_std: float = 0.02,
        filter_regularization: float = 0.0,
        adaptive_initialization: bool = True,
        spectral_dropout_p: float = 0.0,
    ):
        super().__init__(hidden_dim, sequence_length, dropout, norm_eps, learnable_filters)
        self.activation = activation
        self.filter_regularization = filter_regularization
        self.adaptive_initialization = adaptive_initialization
        self.spectral_dropout_p = spectral_dropout_p

        # Initialize filter parameters
        if adaptive_initialization:
            # Frequency-aware initialization: smaller values for high frequencies
            frequencies = torch.fft.fftfreq(sequence_length)
            # Weight by inverse frequency (avoiding DC component)
            freq_weights = 1.0 / (torch.abs(frequencies) + 0.1)
            freq_weights = freq_weights / freq_weights.mean()

            self.filter_real = nn.Parameter(
                torch.randn(sequence_length, hidden_dim)
                * filter_init_std
                * freq_weights.unsqueeze(-1)
            )
            self.filter_imag = nn.Parameter(
                torch.randn(sequence_length, hidden_dim)
                * filter_init_std
                * freq_weights.unsqueeze(-1)
            )
        else:
            # Standard initialization
            self.filter_real = nn.Parameter(
                torch.randn(sequence_length, hidden_dim) * filter_init_std
            )
            self.filter_imag = nn.Parameter(
                torch.randn(sequence_length, hidden_dim) * filter_init_std
            )

        # Store FFT transform as non-module attribute
        self.fft1d: FFT1D  # Type annotation for mypy
        object.__setattr__(self, "fft1d", FFT1D(norm=fft_norm))

        self.activation_fn: Callable[[Tensor], Tensor]
        if activation == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation == "identity":
            self.activation_fn = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Spectral dropout for regularization
        if spectral_dropout_p > 0:
            self.spectral_dropout: nn.Module = nn.Dropout2d(spectral_dropout_p)
        else:
            self.spectral_dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply adaptive global filtering.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        Tensor
            Adaptively filtered tensor.
        """
        # Transform to frequency domain
        x_freq = self.fft1d.transform(x, dim=1)

        # Get actual sequence length
        seq_len = x_freq.shape[1]

        # Adapt filter to actual sequence length using interpolation
        if seq_len != self.sequence_length:
            # Use interpolation to adapt filters to the actual sequence length
            filter_real = (
                nn.functional.interpolate(
                    self.filter_real.T.unsqueeze(0),  # (1, hidden_dim, sequence_length)
                    size=seq_len,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .T
            )  # (seq_len, hidden_dim)

            filter_imag = (
                nn.functional.interpolate(
                    self.filter_imag.T.unsqueeze(0),  # (1, hidden_dim, sequence_length)
                    size=seq_len,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .T
            )  # (seq_len, hidden_dim)
        else:
            filter_real = self.filter_real
            filter_imag = self.filter_imag

        # Create complex filter with activation
        filter_complex = make_complex(
            self.activation_fn(filter_real), self.activation_fn(filter_imag)
        )

        # Apply spectral dropout to filter (not input)
        if self.training and self.spectral_dropout_p > 0:
            filter_complex = self.spectral_dropout(filter_complex)

        # Apply filtering in frequency domain
        filtered_freq = complex_multiply(x_freq, filter_complex)

        # Transform back to time domain
        filtered_time = self.fft1d.inverse_transform(filtered_freq, dim=1)

        # Take real part
        output = torch.real(filtered_time)

        # Apply standard dropout
        output = self.dropout(output)

        return output  # type: ignore[no-any-return]

    def get_filter_response(self) -> Tensor:
        """Get adaptive frequency response.

        Returns
        -------
        Tensor
            Complex frequency response with current parameters.
        """
        return make_complex(
            self.activation_fn(self.filter_real), self.activation_fn(self.filter_imag)
        )

    def get_regularization_loss(self) -> Tensor:
        """Compute L2 regularization loss for filter parameters.

        Returns
        -------
        Tensor
            Scalar regularization loss.
        """
        if self.filter_regularization <= 0:
            return torch.tensor(0.0, device=self.filter_real.device)

        real_loss = torch.norm(self.filter_real, p=2) ** 2
        imag_loss = torch.norm(self.filter_imag, p=2) ** 2

        return self.filter_regularization * (real_loss + imag_loss)  # type: ignore[no-any-return]

    def get_spectral_properties(self) -> dict[str, str | bool | int]:
        """Get adaptive filter properties.

        Returns
        -------
        dict[str, str | bool | int]
            Comprehensive properties including adaptive features.
        """
        return {
            "frequency_domain": True,
            "learnable_filters": True,
            "complex_valued": True,
            "selective_filtering": True,
            "energy_preserving": False,
            "adaptive_initialization": self.adaptive_initialization,
            "regularization": self.filter_regularization > 0,
            "spectral_dropout": self.spectral_dropout_p > 0,
            "activation": self.activation,
            "parameter_count": 2 * self.sequence_length * self.hidden_dim,
        }


__all__: list[str] = [
    "AdaptiveGlobalFilter",
    "GlobalFilterMixing",
    "GlobalFilterMixing2D",
]
