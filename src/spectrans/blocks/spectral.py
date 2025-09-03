"""Spectral transformer blocks.

This module provides pre-configured transformer blocks for different spectral
architectures, combining various mixing and attention layers with feedforward
networks.

Classes
-------
FNetBlock
    Transformer block using Fourier mixing (FNet architecture).
GFNetBlock
    Transformer block using global filter mixing.
AFNOBlock
    Transformer block using adaptive Fourier neural operator.
SpectralAttentionBlock
    Transformer block using spectral attention with RFF.
LSTBlock
    Transformer block using linear spectral transform attention.
WaveletBlock
    Transformer block using wavelet mixing.
FNOBlock
    Transformer block using Fourier neural operator layers.

Examples
--------
Creating different spectral blocks:

>>> from spectrans.blocks.spectral import FNetBlock, GFNetBlock
>>> fnet_block = FNetBlock(hidden_dim=768, dropout=0.1)
>>> gfnet_block = GFNetBlock(hidden_dim=768, sequence_length=512)

Notes
-----
Each spectral block is optimized for specific use cases:
- FNetBlock: Efficient mixing using FFT, good for long sequences
- GFNetBlock: Learnable frequency filters, good for structured patterns
- AFNOBlock: Adaptive mode selection, good for multiscale features
- SpectralAttentionBlock: Kernel approximation, good when attention patterns matter
- LSTBlock: Orthogonal transforms, good for preserving energy
- WaveletBlock: Multiscale analysis, good for hierarchical features
- FNOBlock: Neural operators, good for function approximation

References
----------
.. [1] Lee-Thorp et al., "FNet: Mixing Tokens with Fourier Transforms", 2022.
.. [2] Rao et al., "Global Filter Networks for Image Classification", 2021.
.. [3] Guibas et al., "Adaptive Fourier Neural Operators", 2022.
"""

from typing import cast

import torch.nn as nn

from ..core.registry import register_component
from ..core.types import ActivationType, KernelType, TransformLSTType, WaveletType
from ..layers.attention.lst import LSTAttention
from ..layers.attention.spectral import SpectralAttention
from ..layers.mixing.afno import AFNOMixing
from ..layers.mixing.fourier import FourierMixing
from ..layers.mixing.global_filter import GlobalFilterMixing
from ..layers.mixing.wavelet import WaveletMixing
from ..layers.operators.fno import FourierNeuralOperator
from .base import PreNormBlock


@register_component("block", "fnet")
class FNetBlock(PreNormBlock):
    """FNet transformer block with Fourier mixing.

    This block uses Fourier transforms for token mixing, providing an efficient
    alternative to attention with O(n log n) complexity.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the model.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the FFN. Default is 4 * hidden_dim.
    activation : str, optional
        Activation function. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        mixing_layer = FourierMixing(hidden_dim=hidden_dim)
        super().__init__(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=cast(ActivationType, activation),
            dropout=dropout,
            norm_eps=norm_eps,
        )


@register_component("block", "gfnet")
class GFNetBlock(PreNormBlock):
    """GFNet transformer block with global filter mixing.

    This block uses learnable frequency-domain filters for token mixing,
    allowing the model to learn which frequency components to emphasize.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the model.
    sequence_length : int
        Maximum sequence length.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the FFN. Default is 4 * hidden_dim.
    filter_activation : str, optional
        Activation for filters ('sigmoid', 'tanh', or 'identity'). Default is 'sigmoid'.
    filter_init_std : float, optional
        Initialization std for filters. Default is 0.02.
    activation : str, optional
        Activation function. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.
    """

    def __init__(
        self,
        hidden_dim: int,
        sequence_length: int,
        ffn_hidden_dim: int | None = None,
        filter_activation: str = "sigmoid",
        filter_init_std: float = 0.02,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        mixing_layer = GlobalFilterMixing(
            hidden_dim=hidden_dim,
            sequence_length=sequence_length,
            activation=cast(ActivationType, filter_activation),
            filter_init_std=filter_init_std,
        )
        super().__init__(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=cast(ActivationType, activation),
            dropout=dropout,
            norm_eps=norm_eps,
        )


@register_component("block", "afno")
class AFNOBlock(PreNormBlock):
    """AFNO transformer block with adaptive Fourier neural operator.

    This block uses adaptive Fourier mode selection with MLPs in the frequency
    domain for efficient token mixing.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the model.
    sequence_length : int
        Maximum sequence length.
    modes : int | None, optional
        Number of Fourier modes to retain. Default is sequence_length // 2.
    mlp_hidden_dim : int | None, optional
        Hidden dimension of the frequency-domain MLP. Default is hidden_dim.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the FFN. Default is 4 * hidden_dim.
    activation : str, optional
        Activation function. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.
    """

    def __init__(
        self,
        hidden_dim: int,
        sequence_length: int,
        modes: int | None = None,
        mlp_hidden_dim: int | None = None,
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        # Determine mlp_ratio from mlp_hidden_dim if provided
        mlp_ratio = mlp_hidden_dim / hidden_dim if mlp_hidden_dim is not None else 2.0

        mixing_layer = AFNOMixing(
            hidden_dim=hidden_dim,
            max_sequence_length=sequence_length,
            modes_seq=modes,
            modes_hidden=modes,
            mlp_ratio=mlp_ratio,
            activation=cast(ActivationType, activation),
            dropout=dropout,
        )
        super().__init__(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=cast(ActivationType, activation),
            dropout=dropout,
            norm_eps=norm_eps,
        )


@register_component("block", "spectral_attention")
class SpectralAttentionBlock(PreNormBlock):
    """Spectral attention transformer block.

    This block uses spectral attention with random Fourier features for
    efficient kernel approximation.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the model.
    num_heads : int, optional
        Number of attention heads. Default is 8.
    num_features : int | None, optional
        Number of random features. Default is 256.
    kernel_type : str, optional
        Type of kernel ('gaussian' or 'laplacian'). Default is 'gaussian'.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the FFN. Default is 4 * hidden_dim.
    activation : str, optional
        Activation function. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_features: int | None = None,
        kernel_type: str = "gaussian",
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        mixing_layer = SpectralAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_features=num_features,
            kernel_type=cast(KernelType, kernel_type),
            dropout=dropout,
        )
        super().__init__(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=cast(ActivationType, activation),
            dropout=dropout,
            norm_eps=norm_eps,
        )


@register_component("block", "lst")
class LSTBlock(PreNormBlock):
    """LST transformer block with linear spectral transform attention.

    This block uses orthogonal transforms (DCT, DST, or Hadamard) for
    efficient attention computation.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the model.
    num_heads : int, optional
        Number of attention heads. Default is 8.
    transform_type : str, optional
        Type of transform ('dct', 'dst', or 'hadamard'). Default is 'dct'.
    use_scaling : bool, optional
        Whether to use learnable scaling. Default is True.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the FFN. Default is 4 * hidden_dim.
    activation : str, optional
        Activation function. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        transform_type: str = "dct",
        use_scaling: bool = True,
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        mixing_layer = LSTAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            transform_type=cast(TransformLSTType, transform_type),
            learnable_scale=use_scaling,
            dropout=dropout,
        )
        super().__init__(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=cast(ActivationType, activation),
            dropout=dropout,
            norm_eps=norm_eps,
        )


@register_component("block", "wavelet")
class WaveletBlock(PreNormBlock):
    """Wavelet transformer block with wavelet mixing.

    This block uses discrete wavelet transforms for multiscale token mixing,
    capturing both local and global patterns.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the model.
    wavelet : str, optional
        Type of wavelet. Default is 'db4'.
    levels : int, optional
        Number of decomposition levels. Default is 3.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the FFN. Default is 4 * hidden_dim.
    activation : str, optional
        Activation function. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.
    """

    def __init__(
        self,
        hidden_dim: int,
        wavelet: str = "db4",
        levels: int = 3,
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        mixing_layer = WaveletMixing(
            hidden_dim=hidden_dim,
            wavelet=cast(WaveletType, wavelet),
            levels=levels,
            dropout=dropout,
        )
        super().__init__(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=cast(ActivationType, activation),
            dropout=dropout,
            norm_eps=norm_eps,
        )


@register_component("block", "fno")
class FNOBlock(PreNormBlock):
    """FNO transformer block with Fourier neural operator.

    This block uses Fourier neural operators for learning mappings between
    function spaces, particularly useful for PDE-related tasks.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the model.
    modes : int | None, optional
        Number of Fourier modes. Default is 16.
    num_layers : int, optional
        Number of FNO layers. Default is 1.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the FFN. Default is 4 * hidden_dim.
    activation : str, optional
        Activation function. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.
    """

    def __init__(
        self,
        hidden_dim: int,
        modes: int | None = None,
        num_layers: int = 1,
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        if modes is None:
            modes = 16

        # Use FourierNeuralOperator for mixing
        if num_layers == 1:
            mixing_layer = FourierNeuralOperator(
                hidden_dim=hidden_dim,
                modes=modes,
                activation=cast(ActivationType, activation),
            )
        else:
            # Stack multiple FNO layers
            layers = []
            for _ in range(num_layers):
                layers.append(
                    FourierNeuralOperator(
                        hidden_dim=hidden_dim,
                        modes=modes,
                        activation=cast(ActivationType, activation),
                    )
                )
            mixing_layer = nn.Sequential(*layers)

        super().__init__(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=cast(ActivationType, activation),
            dropout=dropout,
            norm_eps=norm_eps,
        )


@register_component("block", "fno2d")
class FNO2DBlock(PreNormBlock):
    """2D FNO transformer block for image or grid data.

    This block uses 2D Fourier neural operators for spatial data processing.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension (number of channels).
    modes_h : int | None, optional
        Number of Fourier modes for height. Default is 16.
    modes_w : int | None, optional
        Number of Fourier modes for width. Default is 16.
    num_layers : int, optional
        Number of FNO layers. Default is 1.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the FFN. Default is 4 * hidden_dim.
    activation : str, optional
        Activation function. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.
    """

    def __init__(
        self,
        hidden_dim: int,
        modes_h: int | None = None,
        modes_w: int | None = None,
        num_layers: int = 1,
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        if modes_h is None:
            modes_h = 16
        if modes_w is None:
            modes_w = 16

        # For 2D, we use FourierNeuralOperator with 2D mode specification
        modes_2d = (modes_h, modes_w)
        if num_layers == 1:
            mixing_layer = FourierNeuralOperator(
                hidden_dim=hidden_dim,
                modes=modes_2d,  # Use 2D mode tuple
                activation=cast(ActivationType, activation),
            )
        else:
            # Stack multiple FNO layers
            layers = []
            for _ in range(num_layers):
                layers.append(
                    FourierNeuralOperator(
                        hidden_dim=hidden_dim,
                        modes=modes_2d,  # Use 2D mode tuple
                        activation=cast(ActivationType, activation),
                    )
                )
            mixing_layer = nn.Sequential(*layers)

        super().__init__(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=cast(ActivationType, activation),
            dropout=dropout,
            norm_eps=norm_eps,
        )
