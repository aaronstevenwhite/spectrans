r"""Hybrid transformer models combining spectral and spatial mixing strategies.

This module implements transformer architectures that alternate between different
mixing mechanisms across layers. Spectral mixing layers (Fourier, Wavelet, AFNO, GFNet)
provide efficient global pattern capture, while spatial mixing layers (attention variants)
model local dependencies. This design balances computational efficiency with modeling
capacity.

Classes
-------
HybridTransformer
    Configurable hybrid model with alternating spectral-spatial layers.
HybridEncoder
    Encoder-only variant for representation learning tasks.
AlternatingTransformer
    Simplified model alternating between exactly two mixing types.

Examples
--------
Basic hybrid transformer with Fourier-Attention alternation:

>>> import torch
>>> from spectrans.models.hybrid import HybridTransformer
>>> model = HybridTransformer(
...     vocab_size=30000,
...     hidden_dim=768,
...     num_layers=12,
...     spectral_type='fourier',
...     spatial_type='attention',
...     num_classes=10
... )
>>> input_ids = torch.randint(0, 30000, (32, 512))
>>> logits = model(input_ids)
>>> assert logits.shape == (32, 10)

Wavelet-SpectralAttention hybrid:

>>> model = HybridTransformer(
...     hidden_dim=512,
...     num_layers=8,
...     spectral_type='wavelet',
...     spatial_type='spectral_attention',
...     spectral_config={'wavelet': 'db8', 'levels': 3},
...     spatial_config={'num_features': 256}
... )

Alternating transformer with two specific layer types:

>>> model = AlternatingTransformer(
...     hidden_dim=768,
...     num_layers=12,
...     layer1_type='fourier',
...     layer2_type='attention',
...     layer1_config={'use_real_fft': True},
...     layer2_config={'num_heads': 8}
... )

Notes
-----
Mathematical Foundation:

Hybrid transformers alternate between spectral and spatial mixing strategies across
layers. For the default "even_spectral" pattern with :math:`L` layers:

Even-indexed layers (spectral mixing):

.. math::
    \mathbf{X}_{2i+1} = \mathbf{X}_{2i} + \text{SpectralMix}(\text{LayerNorm}(\mathbf{X}_{2i}))

Odd-indexed layers (spatial mixing):

.. math::
    \mathbf{X}_{2i+2} = \mathbf{X}_{2i+1} + \text{SpatialMix}(\text{LayerNorm}(\mathbf{X}_{2i+1}))

where each is followed by a feedforward network:

.. math::
    \mathbf{X}_{l+1} = \mathbf{X}_{l} + \text{FFN}(\text{LayerNorm}(\mathbf{X}_{l}))

Spectral Mixing Operations:

Different spectral mixing types provide varying computational complexities:

- **Fourier**: :math:`\text{FFT}` and :math:`\text{IFFT}` with :math:`O(n \log n)` complexity
- **Wavelet**: Multi-scale DWT decomposition with :math:`O(n)` complexity
- **AFNO**: Mode-truncated spectral convolution with :math:`O(k_n k_d)` complexity
- **GFNet**: Learnable global filters with :math:`O(n \log n)` complexity

Spatial Mixing Operations:

Spatial mixing layers model position-dependent interactions:

- **Standard Attention**: :math:`\text{softmax}(\mathbf{Q}\mathbf{K}^T/\sqrt{d})\mathbf{V}` with :math:`O(n^2 d)` complexity
- **Spectral Attention**: RFF-approximated attention with :math:`O(n D d)` complexity where :math:`D` is feature dimension
- **LST**: Linear spectral transform attention with :math:`O(n \log n \cdot d)` complexity

Complexity Analysis:

For a hybrid model with :math:`L` layers, :math:`n` sequence length, :math:`d` hidden dimension:

Total complexity depends on the dominant mixing operation. With :math:`L_s` spectral
and :math:`L_{sp}` spatial layers:

.. math::
    T_{\text{total}} = L_s \cdot T_{\text{spectral}} + L_{sp} \cdot T_{\text{spatial}} + L \cdot T_{\text{FFN}}

where :math:`T_{\text{FFN}} = O(n d^2)` for feedforward networks.

The hybrid approach reduces the overall complexity compared to pure attention models
while maintaining modeling capacity through the complementary mixing strategies.

References
----------
.. [1] Tay, Y., et al. "Efficient Transformers: A Survey." ACM Computing Surveys 55.6 (2022): 1-28.
.. [2] Lee-Thorp, J., et al. "FNet: Mixing Tokens with Fourier Transforms." NAACL 2022.
.. [3] Guibas, J., et al. "Adaptive Fourier Neural Operators." ICLR 2022.

See Also
--------
spectrans.blocks.hybrid : Hybrid block implementations.
spectrans.models.fnet : Pure Fourier transformer implementation.
spectrans.models.spectral_attention : Pure spectral attention transformer.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..blocks.base import PreNormBlock
from ..core.registry import register_component
from ..core.types import OutputHeadType, PositionalEncodingType
from ..layers.attention.lst import LSTAttention
from ..layers.attention.spectral import SpectralAttention
from ..layers.mixing.afno import AFNOMixing
from ..layers.mixing.fourier import FourierMixing, RealFourierMixing
from ..layers.mixing.global_filter import GlobalFilterMixing
from ..layers.mixing.wavelet import WaveletMixing
from .base import BaseModel

if TYPE_CHECKING:
    from ..config.models import HybridModelConfig, ModelConfig


class StandardAttention(nn.Module):
    """Standard multi-head self-attention wrapper.

    Wraps PyTorch's MultiheadAttention for use as a mixing layer.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension size.
    num_heads : int, default=8
        Number of attention heads.
    dropout : float, default=0.0
        Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of same shape.
        """
        # Self-attention: queries, keys, and values are all x
        attn_output: torch.Tensor
        attn_output, _ = self.attention(x, x, x, need_weights=False)
        return attn_output


@register_component("model", "hybrid")
class HybridTransformer(BaseModel):
    r"""Hybrid Spectral-Spatial Transformer model.

    Combines spectral and spatial mixing strategies across layers to balance
    computational efficiency with modeling expressiveness. The model alternates
    between spectral layers (efficient global mixing) and spatial layers
    (expressive local modeling) according to configurable patterns.

    For a sequence :math:`\mathbf{X}_0 \in \mathbb{R}^{n \times d}`, the hybrid
    transformer applies alternating transformations:

    **Spectral layers** (:math:`\ell` even for "even_spectral" pattern):

    .. math::
        \mathbf{Z}_\ell = \mathbf{X}_\ell + \text{SpectralMix}(\text{LN}(\mathbf{X}_\ell))

    **Spatial layers** (:math:`\ell` odd for "even_spectral" pattern):

    .. math::
        \mathbf{Z}_\ell = \mathbf{X}_\ell + \text{SpatialMix}(\text{LN}(\mathbf{X}_\ell))

    where :math:`\text{LN}(\cdot)` denotes LayerNorm and each block concludes with:

    .. math::
        \mathbf{X}_{\ell+1} = \mathbf{Z}_\ell + \text{FFN}(\text{LN}(\mathbf{Z}_\ell))

    The spectral mixing operations provide different complexity-accuracy tradeoffs:
    - Fourier: :math:`O(n \log n)` via FFT/IFFT
    - Wavelet: :math:`O(n)` via fast DWT algorithms
    - AFNO: :math:`O(k_n k_d d)` with mode truncation parameters :math:`k_n, k_d`
    - GFNet: :math:`O(n \log n)` with learnable spectral filters

    Parameters
    ----------
    vocab_size : int | None, optional
        Size of the vocabulary for token embeddings.
    hidden_dim : int, default=768
        Hidden dimension size.
    num_layers : int, default=12
        Number of transformer blocks.
    max_sequence_length : int, default=512
        Maximum sequence length.
    spectral_type : str, default='fourier'
        Type of spectral mixing: 'fourier', 'wavelet', 'afno', 'gfnet'.
    spatial_type : str, default='attention'
        Type of spatial mixing: 'attention', 'spectral_attention', 'lst'.
    alternation_pattern : str, default='even_spectral'
        How to alternate: 'even_spectral', 'alternate', 'custom'.
    num_heads : int, default=8
        Number of attention heads for spatial layers.
    spectral_config : dict | None, optional
        Additional configuration for spectral layers.
    spatial_config : dict | None, optional
        Additional configuration for spatial layers.
    num_classes : int | None, optional
        Number of output classes for classification.
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
    output_type : OutputHeadType, default='classification'
        Type of output head.
    gradient_checkpointing : bool, default=False
        Whether to use gradient checkpointing.

    Attributes
    ----------
    spectral_type : str
        Type of spectral mixing being used.
    spatial_type : str
        Type of spatial mixing being used.
    alternation_pattern : str
        The alternation pattern.
    blocks : nn.ModuleList
        List of hybrid transformer blocks.
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_dim: int = 768,
        num_layers: int = 12,
        max_sequence_length: int = 512,
        spectral_type: str = "fourier",
        spatial_type: str = "attention",
        alternation_pattern: str = "even_spectral",
        num_heads: int = 8,
        spectral_config: dict | None = None,
        spatial_config: dict | None = None,
        num_classes: int | None = None,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        dropout: float = 0.1,
        ffn_hidden_dim: int | None = None,
        norm_eps: float = 1e-12,
        output_type: OutputHeadType = "classification",
        gradient_checkpointing: bool = False,
    ):
        self.spectral_type = spectral_type
        self.spatial_type = spatial_type
        self.alternation_pattern = alternation_pattern
        self.num_heads = num_heads
        self.spectral_config = spectral_config or {}
        self.spatial_config = spatial_config or {}
        self._dropout_rate = dropout

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

    def _create_spectral_layer(self) -> nn.Module:
        """Create a spectral mixing layer based on configuration.

        Returns
        -------
        nn.Module
            Configured spectral mixing layer.
        """
        if self.spectral_type == "fourier":
            use_real_fft = self.spectral_config.get("use_real_fft", True)
            if use_real_fft:
                return RealFourierMixing(
                    hidden_dim=self.hidden_dim,
                    dropout=self._dropout_rate,
                )
            else:
                return FourierMixing(
                    hidden_dim=self.hidden_dim,
                    dropout=self._dropout_rate,
                )

        elif self.spectral_type == "wavelet":
            return WaveletMixing(
                hidden_dim=self.hidden_dim,
                wavelet=self.spectral_config.get("wavelet", "db4"),
                levels=self.spectral_config.get("levels", 3),
                mixing_mode=self.spectral_config.get("mixing_mode", "pointwise"),
                dropout=self._dropout_rate,
            )

        elif self.spectral_type == "afno":
            # Handle n_modes shorthand for modes_seq
            modes_seq = self.spectral_config.get("modes_seq") or self.spectral_config.get("n_modes")
            return AFNOMixing(
                hidden_dim=self.hidden_dim,
                max_sequence_length=self.max_sequence_length,
                modes_seq=modes_seq,
                modes_hidden=self.spectral_config.get("modes_hidden"),
                mlp_ratio=self.spectral_config.get("mlp_ratio", 2.0),
                dropout=self._dropout_rate,
            )

        elif self.spectral_type == "gfnet":
            return GlobalFilterMixing(
                sequence_length=self.max_sequence_length,
                hidden_dim=self.hidden_dim,
                activation=self.spectral_config.get("activation", "sigmoid"),
                dropout=self._dropout_rate,
            )

        else:
            raise ValueError(f"Unknown spectral type: {self.spectral_type}")

    def _create_spatial_layer(self) -> nn.Module:
        """Create a spatial mixing layer based on configuration.

        Returns
        -------
        nn.Module
            Configured spatial mixing layer.
        """
        if self.spatial_type == "attention":
            return StandardAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self._dropout_rate,
            )

        elif self.spatial_type == "spectral_attention":
            return SpectralAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_features=self.spatial_config.get("num_features"),
                kernel_type=self.spatial_config.get("kernel_type", "softmax"),
                use_orthogonal=self.spatial_config.get("use_orthogonal", False),
                dropout=self._dropout_rate,
            )

        elif self.spatial_type == "lst":
            return LSTAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                transform_type=self.spatial_config.get("transform_type", "dct"),
                use_bias=self.spatial_config.get("use_bias", True),
                dropout=self._dropout_rate,
            )

        else:
            raise ValueError(f"Unknown spatial type: {self.spatial_type}")

    def build_blocks(self) -> nn.ModuleList:
        """Build hybrid transformer blocks.

        Returns
        -------
        nn.ModuleList
            List of transformer blocks with alternating mixing strategies.
        """
        blocks = []

        for layer_idx in range(self.num_layers):
            # Determine which mixing layer to use based on pattern
            if self.alternation_pattern == "even_spectral":
                # Even layers use spectral, odd use spatial
                use_spectral = (layer_idx % 2 == 0)
            elif self.alternation_pattern == "alternate":
                # Strictly alternate starting with spectral
                use_spectral = (layer_idx % 2 == 0)
            else:  # custom or other patterns
                # Default to alternating
                use_spectral = (layer_idx % 2 == 0)

            # Create appropriate mixing layer
            if use_spectral:
                mixing_layer = self._create_spectral_layer()
            else:
                mixing_layer = self._create_spatial_layer()

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

    @property
    def complexity(self) -> dict[str, str]:
        """Computational complexity of the hybrid transformer.

        Returns
        -------
        dict[str, str]
            Dictionary with 'time' and 'space' complexity.
        """
        n = self.max_sequence_length
        d = self.hidden_dim
        L = self.num_layers

        # Complexity depends on the mix of layer types
        spectral_layers = L // 2 + (L % 2 if self.alternation_pattern == "even_spectral" else 0)
        spatial_layers = L - spectral_layers

        # Spectral complexity
        if self.spectral_type in ["fourier", "gfnet"]:
            spectral_time = f"{spectral_layers} * {n} * log({n}) * {d}"
        else:  # wavelet, afno
            spectral_time = f"{spectral_layers} * {n} * {d}"

        # Spatial complexity
        if self.spatial_type == "attention":
            spatial_time = f"{spatial_layers} * {n}^2 * {d}"
        else:  # spectral_attention, lst
            spatial_time = f"{spatial_layers} * {n} * {d}"

        return {
            "time": f"O({spectral_time} + {spatial_time})",
            "space": f"O({L} * {n} * {d})",
        }

    @classmethod
    def from_config(cls, config: "HybridModelConfig | ModelConfig") -> "HybridTransformer":
        """Create hybrid transformer from configuration.

        Parameters
        ----------
        config : HybridModelConfig | ModelConfig
            Configuration object with model parameters.

        Returns
        -------
        HybridTransformer
            Configured hybrid transformer model.
        """
        return cls(
            vocab_size=getattr(config, "vocab_size", None),
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            max_sequence_length=config.sequence_length,
            spectral_type=getattr(config, "spectral_type", "fourier"),
            spatial_type=getattr(config, "spatial_type", "attention"),
            alternation_pattern=getattr(config, "alternation_pattern", "even_spectral"),
            num_heads=getattr(config, "num_heads", 8),
            spectral_config=getattr(config, "spectral_config", None),
            spatial_config=getattr(config, "spatial_config", None),
            num_classes=getattr(config, "num_classes", None),
            use_positional_encoding=getattr(config, "use_positional_encoding", True),
            positional_encoding_type=getattr(config, "positional_encoding_type", "sinusoidal"),
            dropout=config.dropout,
            ffn_hidden_dim=getattr(config, "ffn_hidden_dim", None),
            norm_eps=getattr(config, "norm_eps", 1e-12),
            output_type=getattr(config, "output_type", "classification"),
            gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
        )


@register_component("model", "hybrid_encoder")
class HybridEncoder(HybridTransformer):
    """Encoder-only hybrid transformer for representation learning.

    This variant returns hidden states without any task-specific head,
    suitable for feature extraction and representation learning.

    Parameters
    ----------
    hidden_dim : int, default=768
        Hidden dimension size.
    num_layers : int, default=12
        Number of transformer blocks.
    max_sequence_length : int, default=512
        Maximum sequence length.
    spectral_type : str, default='fourier'
        Type of spectral mixing.
    spatial_type : str, default='attention'
        Type of spatial mixing.
    alternation_pattern : str, default='even_spectral'
        Layer alternation pattern.
    num_heads : int, default=8
        Number of attention heads.
    spectral_config : dict | None, optional
        Spectral layer configuration.
    spatial_config : dict | None, optional
        Spatial layer configuration.
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
        spectral_type: str = "fourier",
        spatial_type: str = "attention",
        alternation_pattern: str = "even_spectral",
        num_heads: int = 8,
        spectral_config: dict | None = None,
        spatial_config: dict | None = None,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        dropout: float = 0.1,
        ffn_hidden_dim: int | None = None,
        norm_eps: float = 1e-12,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(
            vocab_size=None,  # No token embeddings
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            spectral_type=spectral_type,
            spatial_type=spatial_type,
            alternation_pattern=alternation_pattern,
            num_heads=num_heads,
            spectral_config=spectral_config,
            spatial_config=spatial_config,
            num_classes=None,  # No classification head
            use_positional_encoding=use_positional_encoding,
            positional_encoding_type=positional_encoding_type,
            dropout=dropout,
            ffn_hidden_dim=ffn_hidden_dim,
            norm_eps=norm_eps,
            output_type="none",  # Return hidden states
            gradient_checkpointing=gradient_checkpointing,
        )


@register_component("model", "alternating_transformer")
class AlternatingTransformer(BaseModel):
    r"""Transformer that strictly alternates between two mixing strategies.

    A simplified hybrid model that alternates between exactly two types
    of mixing layers following a strict pattern: layer1_type for even-indexed
    layers, layer2_type for odd-indexed layers. This design enables controlled
    comparisons between different mixing strategies.

    For :math:`L` layers, the alternation follows:

    .. math::
        \text{Layer}(\ell) = \begin{cases}
        \text{layer1_type} & \text{if } \ell \bmod 2 = 0 \\
        \text{layer2_type} & \text{if } \ell \bmod 2 = 1
        \end{cases}

    Each layer applies the mixing operation with residual connection:

    .. math::
        \mathbf{X}_{\ell+1} = \mathbf{X}_\ell + \text{MixingLayer}_\ell(\text{LayerNorm}(\mathbf{X}_\ell))

    followed by the standard feedforward block with another residual connection.

    Parameters
    ----------
    vocab_size : int | None, optional
        Size of the vocabulary for token embeddings.
    hidden_dim : int, default=768
        Hidden dimension size.
    num_layers : int, default=12
        Number of transformer blocks.
    max_sequence_length : int, default=512
        Maximum sequence length.
    layer1_type : str, default='fourier'
        Type of first mixing layer.
    layer2_type : str, default='attention'
        Type of second mixing layer.
    layer1_config : dict | None, optional
        Configuration for first layer type.
    layer2_config : dict | None, optional
        Configuration for second layer type.
    num_classes : int | None, optional
        Number of output classes.
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
    output_type : OutputHeadType, default='classification'
        Type of output head.
    gradient_checkpointing : bool, default=False
        Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_dim: int = 768,
        num_layers: int = 12,
        max_sequence_length: int = 512,
        layer1_type: str = "fourier",
        layer2_type: str = "attention",
        layer1_config: dict | None = None,
        layer2_config: dict | None = None,
        num_classes: int | None = None,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        dropout: float = 0.1,
        ffn_hidden_dim: int | None = None,
        norm_eps: float = 1e-12,
        output_type: OutputHeadType = "classification",
        gradient_checkpointing: bool = False,
    ):
        # Store configuration for alternating layers before super().__init__
        # These need to be available when build_blocks is called
        self.layer1_type = layer1_type
        self.layer2_type = layer2_type
        self.layer1_config = layer1_config or {}
        self.layer2_config = layer2_config or {}
        self._dropout_rate = dropout
        self._layer1_is_spectral = layer1_type in ["fourier", "wavelet", "afno", "gfnet"]
        self._layer2_is_spectral = layer2_type in ["fourier", "wavelet", "afno", "gfnet"]

        # Initialize BaseModel
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
        """Build alternating transformer blocks.

        Returns
        -------
        nn.ModuleList
            List of alternating transformer blocks.
        """
        blocks = []

        for layer_idx in range(self.num_layers):
            # Alternate between layer types
            use_layer1 = (layer_idx % 2 == 0)
            mixing_layer: RealFourierMixing | FourierMixing | StandardAttention

            if use_layer1:
                # Create layer1 type
                if self.layer1_type == "fourier":
                    use_real_fft = self.layer1_config.get("use_real_fft", True)
                    if use_real_fft:
                        mixing_layer = RealFourierMixing(hidden_dim=self.hidden_dim, dropout=self._dropout_rate)
                    else:
                        mixing_layer = FourierMixing(hidden_dim=self.hidden_dim, dropout=self._dropout_rate)
                elif self.layer1_type == "attention":
                    mixing_layer = StandardAttention(
                        hidden_dim=self.hidden_dim,
                        num_heads=self.layer1_config.get("num_heads", 8),
                        dropout=self._dropout_rate,
                    )
                else:
                    raise ValueError(
                        f"Invalid layer1_type '{self.layer1_type}'. "
                        f"Supported types: ['fourier', 'attention']"
                    )
            else:
                # Create layer2 type
                if self.layer2_type == "attention":
                    mixing_layer = StandardAttention(
                        hidden_dim=self.hidden_dim,
                        num_heads=self.layer2_config.get("num_heads", 8),
                        dropout=self._dropout_rate,
                    )
                elif self.layer2_type == "fourier":
                    use_real_fft = self.layer2_config.get("use_real_fft", True)
                    if use_real_fft:
                        mixing_layer = RealFourierMixing(hidden_dim=self.hidden_dim, dropout=self._dropout_rate)
                    else:
                        mixing_layer = FourierMixing(hidden_dim=self.hidden_dim, dropout=self._dropout_rate)
                else:
                    raise ValueError(
                        f"Invalid layer2_type '{self.layer2_type}'. "
                        f"Supported types: ['attention', 'fourier']"
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

    @property
    def complexity(self) -> dict[str, str]:
        """Computational complexity of the alternating transformer.

        Returns
        -------
        dict[str, str]
            Dictionary with 'time' and 'space' complexity.
        """
        n = self.max_sequence_length
        L = self.num_layers

        # Complexity depends on the mix of layer types
        time_components = []

        if self.layer1_type == "fourier" or self.layer2_type == "fourier":
            time_components.append(f"{n} * log({n})")
        if self.layer1_type == "attention" or self.layer2_type == "attention":
            time_components.append(f"{n}^2")

        time_complexity = f"O({L} * (" + " + ".join(time_components) + "))"
        space_complexity = f"O({L} * {n} * {self.hidden_dim})"

        return {
            "time": time_complexity,
            "space": space_complexity,
        }

