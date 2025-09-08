r"""Wavelet transformer models using discrete wavelet transforms.

This module implements transformer architectures that replace standard attention
mechanisms with discrete wavelet transforms (DWT) for sequence mixing. The DWT
provides multi-resolution analysis, decomposing sequences into approximation and
detail coefficients at different scales while maintaining perfect reconstruction.

Classes
-------
WaveletTransformer
    Full wavelet transformer with DWT-based sequence mixing.
WaveletEncoder
    Encoder-only variant for representation learning tasks.
WaveletDecoder
    Decoder variant for sequence generation with causal processing.

Examples
--------
Basic wavelet transformer for classification:

>>> import torch
>>> from spectrans.models.wavenet_transformer import WaveletTransformer
>>> model = WaveletTransformer(
...     vocab_size=30000,
...     hidden_dim=768,
...     num_layers=12,
...     wavelet='db4',
...     levels=3,
...     max_sequence_length=512,
...     num_classes=10
... )
>>> input_ids = torch.randint(0, 30000, (32, 512))
>>> logits = model(input_ids)
>>> assert logits.shape == (32, 10)

Using different wavelet families:

>>> model_db = WaveletTransformer(
...     hidden_dim=512,
...     wavelet='db8',
...     levels=4
... )
>>> model_sym = WaveletTransformer(
...     hidden_dim=512,
...     wavelet='sym6',
...     levels=3
... )

Encoder for feature extraction:

>>> from spectrans.models.wavenet_transformer import WaveletEncoder
>>> encoder = WaveletEncoder(
...     hidden_dim=768,
...     num_layers=6,
...     wavelet='coif3',
...     levels=2
... )
>>> embeddings = torch.randn(32, 100, 768)
>>> features = encoder(inputs_embeds=embeddings)

Notes
-----
Mathematical Foundation:

The discrete wavelet transform decomposes a signal $\mathbf{x} \in \mathbb{R}^n$
into a multi-scale representation. For $J$ decomposition levels, the DWT produces:

$$
\text{DWT}_J(\mathbf{x}) = \{\mathbf{c}_{A_J}, \{\mathbf{c}_{D_j}\}_{j=1}^J\}
$$

where $\mathbf{c}_{A_J} \in \mathbb{R}^{\frac{n}{2^J}}$ are approximation coefficients
at the coarsest level and $\mathbf{c}_{D_j} \in \mathbb{R}^{\frac{n}{2^j}}$ are detail
coefficients at level $j$.

The decomposition employs convolution with filter banks:

$$
\mathbf{c}_{A_{j+1}}[k] = \sum_m h[m-2k] \mathbf{c}_{A_j}[m]
$$

$$
\mathbf{c}_{D_{j+1}}[k] = \sum_m g[m-2k] \mathbf{c}_{A_j}[m]
$$

where $h$ and $g$ are the low-pass and high-pass analysis filters.
Perfect reconstruction is guaranteed by the synthesis filters satisfying:

$$
\mathbf{x} = \sum_{k} \mathbf{c}_{A_J}[k] \phi_{J,k}(t) + \sum_{j=1}^J \sum_k \mathbf{c}_{D_j}[k] \psi_{j,k}(t)
$$

where $\phi_{J,k}$ and $\psi_{j,k}$ are scaling and wavelet functions.

Transformer Block Structure:

Each wavelet transformer block applies the DWT mixing with residual connections:

$$
\mathbf{Z}_l = \mathbf{X}_l + \text{WaveletMix}(\text{LayerNorm}(\mathbf{X}_l))
$$

$$
\mathbf{X}_{l+1} = \mathbf{Z}_l + \text{FFN}(\text{LayerNorm}(\mathbf{Z}_l))
$$

where the wavelet mixing operation processes each channel of the hidden representation
independently through the DWT/IDWT pipeline.

Complexity Analysis:

For a sequence of length $n$ with hidden dimension $d$ and $L$ layers:
- Time complexity: $O(L \cdot n \cdot d \cdot J)$ where $J$ is decomposition levels
- Space complexity: $O(L \cdot n \cdot d)$
- Single DWT operation: $O(n)$ per channel due to fast convolution algorithms

The linear complexity per channel makes wavelet mixing more efficient than quadratic
attention mechanisms for long sequences.

References
----------
Stéphane Mallat. 1999. A Wavelet Tour of Signal Processing, 2nd edition.
Academic Press, San Diego.

Ingrid Daubechies. 1992. Ten Lectures on Wavelets. CBMS-NSF Regional Conference
Series in Applied Mathematics, Vol. 61. SIAM, Philadelphia.

Martin Vetterli and Jelena Kovačević. 1995. Wavelets and Subband Coding.
Prentice Hall, Englewood Cliffs.

See Also
--------
spectrans.layers.mixing.wavelet : Wavelet mixing layer implementation.
spectrans.transforms.wavelet : DWT transform implementations.
"""

from typing import TYPE_CHECKING

import torch.nn as nn

from ..blocks.base import PreNormBlock
from ..core.registry import register_component
from ..core.types import OutputHeadType, PositionalEncodingType, WaveletType
from ..layers.mixing.wavelet import WaveletMixing
from .base import BaseModel

if TYPE_CHECKING:
    from ..config.models import WaveletTransformerConfig


@register_component("model", "wavelet_transformer")
class WaveletTransformer(BaseModel):
    r"""Wavelet transformer with DWT-based sequence mixing.

    This model replaces attention mechanisms with discrete wavelet transforms,
    providing multi-resolution analysis of sequences with $O(n)$ complexity per channel.
    The DWT decomposes input sequences into approximation and detail coefficients at
    multiple scales, naturally capturing both local transients and global structure.

    The wavelet mixing operation applies the DWT along the sequence dimension for each
    channel independently, processes the coefficients through learnable transformations,
    and reconstructs the sequence via the inverse DWT (IDWT). Perfect reconstruction
    is maintained when no coefficient modification occurs.

    For input :math:`\mathbf{X} \in \mathbb{R}^{n \times d}`, each channel undergoes:

    .. math::
        \mathbf{c} = \text{DWT}_J(\mathbf{X}_{:,i}) \quad \text{for } i \in [1,d]

    .. math::
        \tilde{\mathbf{c}} = f_{\theta}(\mathbf{c})

    .. math::
        \mathbf{Y}_{:,i} = \text{IDWT}_J(\tilde{\mathbf{c}})

    where :math:`f_{\theta}` represents learnable coefficient transformations and :math:`J`
    is the number of decomposition levels.

    Parameters
    ----------
    vocab_size : int | None, optional
        Size of the vocabulary for token embeddings. If None, expects
        pre-embedded inputs.
    hidden_dim : int, default=768
        Hidden dimension size.
    num_layers : int, default=12
        Number of wavelet transformer blocks.
    max_sequence_length : int, default=512
        Maximum sequence length the model can process.
    wavelet : WaveletType, default='db4'
        Type of wavelet to use (e.g., 'db4', 'sym6', 'coif3').
    levels : int, default=3
        Number of wavelet decomposition levels.
    mixing_mode : str, default='pointwise'
        How to mix wavelet coefficients: 'pointwise', 'channel', or 'level'.
    num_classes : int | None, optional
        Number of output classes for classification.
    use_positional_encoding : bool, default=True
        Whether to use positional encoding.
    positional_encoding_type : PositionalEncodingType, default='sinusoidal'
        Type of positional encoding.
    dropout : float, default=0.1
        Dropout probability.
    ffn_hidden_dim : int | None, optional
        Hidden dimension for FFN. If None, defaults to 4 * hidden_dim.
    norm_eps : float, default=1e-12
        Epsilon for layer normalization.
    output_type : OutputHeadType, default='classification'
        Type of output head.
    gradient_checkpointing : bool, default=False
        Whether to use gradient checkpointing for memory efficiency.

    Attributes
    ----------
    wavelet : WaveletType
        The wavelet family being used.
    levels : int
        Number of decomposition levels.
    mixing_mode : str
        Coefficient mixing strategy.
    blocks : nn.ModuleList
        List of wavelet transformer blocks.
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_dim: int = 768,
        num_layers: int = 12,
        max_sequence_length: int = 512,
        wavelet: WaveletType = "db4",
        levels: int = 3,
        mixing_mode: str = "pointwise",
        num_classes: int | None = None,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        dropout: float = 0.1,
        ffn_hidden_dim: int | None = None,
        norm_eps: float = 1e-12,
        output_type: OutputHeadType = "classification",
        gradient_checkpointing: bool = False,
    ):
        self.wavelet = wavelet
        self.levels = levels
        self.mixing_mode = mixing_mode
        self._dropout_rate = dropout  # Store for build_blocks

        super().__init__(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            num_classes=num_classes,
            use_positional_encoding=use_positional_encoding,
            positional_encoding_type=positional_encoding_type,
            dropout=dropout,
            ffn_hidden_dim=ffn_hidden_dim,
            norm_eps=norm_eps,
            output_type=output_type,
            gradient_checkpointing=gradient_checkpointing,
        )

    def build_blocks(self) -> nn.ModuleList:
        """Build wavelet transformer blocks.

        Returns
        -------
        nn.ModuleList
            List of wavelet transformer blocks with DWT mixing layers.
        """
        blocks = []
        for _ in range(self.num_layers):
            # Create wavelet mixing layer
            mixing_layer = WaveletMixing(
                hidden_dim=self.hidden_dim,
                wavelet=self.wavelet,
                levels=self.levels,
                mixing_mode=self.mixing_mode,
                dropout=self._dropout_rate,
            )

            # Create block with pre-normalization
            block = PreNormBlock(
                mixing_layer=mixing_layer,
                hidden_dim=self.hidden_dim,
                ffn_hidden_dim=self.ffn_hidden_dim,
                activation="gelu",
                dropout=self._dropout_rate,
                norm_eps=self.norm_eps,
            )
            blocks.append(block)

        return nn.ModuleList(blocks)


    @classmethod
    def from_config(cls, config: "WaveletTransformerConfig") -> "WaveletTransformer":  # type: ignore[override]
        """Create wavelet transformer from configuration.

        Parameters
        ----------
        config : WaveletTransformerConfig
            Configuration object with model parameters.

        Returns
        -------
        WaveletTransformer
            Configured wavelet transformer model.
        """
        return cls(
            vocab_size=getattr(config, "vocab_size", None),
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            max_sequence_length=config.sequence_length,
            wavelet=getattr(config, "wavelet", "db4"),
            levels=getattr(config, "levels", 3),
            mixing_mode=getattr(config, "mixing_mode", "pointwise"),
            num_classes=getattr(config, "num_classes", None),
            use_positional_encoding=getattr(config, "use_positional_encoding", True),
            positional_encoding_type=getattr(config, "positional_encoding_type", "sinusoidal"),
            dropout=config.dropout,
            ffn_hidden_dim=getattr(config, "ffn_hidden_dim", None),
            norm_eps=getattr(config, "norm_eps", 1e-12),
            output_type=getattr(config, "output_type", "classification"),
            gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
        )


@register_component("model", "wavelet_encoder")
class WaveletEncoder(WaveletTransformer):
    """Encoder-only wavelet transformer for representation learning.

    This variant is designed for extracting representations from sequences
    using wavelet-based mixing, without any task-specific output head.

    Parameters
    ----------
    hidden_dim : int, default=768
        Hidden dimension size.
    num_layers : int, default=12
        Number of wavelet transformer blocks.
    max_sequence_length : int, default=512
        Maximum sequence length.
    wavelet : WaveletType, default='db4'
        Type of wavelet to use.
    levels : int, default=3
        Number of decomposition levels.
    mixing_mode : str, default='pointwise'
        Coefficient mixing strategy.
    use_positional_encoding : bool, default=True
        Whether to use positional encoding.
    positional_encoding_type : PositionalEncodingType, default='sinusoidal'
        Type of positional encoding.
    dropout : float, default=0.1
        Dropout probability.
    ffn_hidden_dim : int | None, optional
        Hidden dimension for FFN.
    norm_eps : float, default=1e-12
        Layer normalization epsilon.
    gradient_checkpointing : bool, default=False
        Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        max_sequence_length: int = 512,
        wavelet: WaveletType = "db4",
        levels: int = 3,
        mixing_mode: str = "pointwise",
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        dropout: float = 0.1,
        ffn_hidden_dim: int | None = None,
        norm_eps: float = 1e-12,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(
            vocab_size=None,  # No token embeddings for encoder
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            wavelet=wavelet,
            levels=levels,
            mixing_mode=mixing_mode,
            num_classes=None,  # No classification head
            use_positional_encoding=use_positional_encoding,
            positional_encoding_type=positional_encoding_type,
            dropout=dropout,
            ffn_hidden_dim=ffn_hidden_dim,
            norm_eps=norm_eps,
            output_type="none",  # Return hidden states
            gradient_checkpointing=gradient_checkpointing,
        )


@register_component("model", "wavelet_decoder")
class WaveletDecoder(WaveletTransformer):
    """Decoder wavelet transformer for sequence generation.

    This variant uses causal wavelet processing suitable for autoregressive
    generation tasks. The wavelet decomposition is modified to respect
    causality constraints.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary for token generation.
    hidden_dim : int, default=768
        Hidden dimension size.
    num_layers : int, default=12
        Number of wavelet transformer blocks.
    max_sequence_length : int, default=512
        Maximum sequence length.
    wavelet : WaveletType, default='db4'
        Type of wavelet to use.
    levels : int, default=2
        Number of decomposition levels (typically lower for causality).
    mixing_mode : str, default='pointwise'
        Coefficient mixing strategy.
    use_positional_encoding : bool, default=True
        Whether to use positional encoding.
    positional_encoding_type : PositionalEncodingType, default='sinusoidal'
        Type of positional encoding.
    dropout : float, default=0.1
        Dropout probability.
    ffn_hidden_dim : int | None, optional
        Hidden dimension for FFN.
    norm_eps : float, default=1e-12
        Layer normalization epsilon.
    gradient_checkpointing : bool, default=False
        Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 12,
        max_sequence_length: int = 512,
        wavelet: WaveletType = "db4",
        levels: int = 2,  # Lower default for causality
        mixing_mode: str = "pointwise",
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        dropout: float = 0.1,
        ffn_hidden_dim: int | None = None,
        norm_eps: float = 1e-12,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            wavelet=wavelet,
            levels=levels,
            mixing_mode=mixing_mode,
            num_classes=vocab_size,  # Output vocabulary size
            use_positional_encoding=use_positional_encoding,
            positional_encoding_type=positional_encoding_type,
            dropout=dropout,
            ffn_hidden_dim=ffn_hidden_dim,
            norm_eps=norm_eps,
            output_type="lm",  # Language modeling head
            gradient_checkpointing=gradient_checkpointing,
        )

