r"""Adaptive Fourier Neural Operator (AFNO) transformer models.

This module implements the AFNO architecture, which performs efficient token mixing
by applying learnable transformations in the truncated Fourier domain. AFNO leverages
the sparsity of signals in frequency space to achieve computational efficiency
while maintaining performance.

The architecture uses adaptive mode truncation - keeping only the most significant
Fourier modes and applying MLPs in the frequency domain, reducing computational
requirements for long sequences.

Classes
-------
AFNOModel
    Complete AFNO model with adaptive Fourier mixing layers.
AFNOEncoder
    Encoder-only AFNO for representation learning.

Examples
--------
Basic AFNO usage for classification:

>>> from spectrans.models.afno import AFNOModel
>>> model = AFNOModel(
...     vocab_size=30000,
...     hidden_dim=768,
...     num_layers=12,
...     max_sequence_length=1024,
...     num_classes=2,
...     modes_seq=256,  # Keep 256 modes in sequence dimension
...     modes_hidden=384  # Keep 384 modes in hidden dimension
... )
>>> input_ids = torch.randint(0, 30000, (8, 1024))
>>> logits = model(input_ids=input_ids)
>>> assert logits.shape == (8, 2)

Using AFNO encoder for long sequences:

>>> from spectrans.models.afno import AFNOEncoder
>>> encoder = AFNOEncoder(
...     hidden_dim=768,
...     num_layers=12,
...     max_sequence_length=4096,
...     modes_seq=512,  # Truncation for efficiency
...     modes_hidden=384
... )
>>> inputs = torch.randn(8, 4096, 768)
>>> features = encoder(inputs_embeds=inputs)
>>> assert features.shape == (8, 4096, 768)

Creating from configuration:

>>> from spectrans.config.models import AFNOModelConfig
>>> config = AFNOModelConfig(
...     hidden_dim=768,
...     num_layers=12,
...     sequence_length=1024,
...     n_modes=256,
...     compression_ratio=0.5
... )
>>> model = AFNOModel.from_config(config)

Notes
-----
Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

Given input tensor :math:`\mathbf{X} \in \mathbb{R}^{n \times d}` where :math:`n`
is sequence length and :math:`d` is hidden dimension, AFNO applies mode-truncated
Fourier operations with learnable transformations.

**Adaptive Fourier Operation:**

The core AFNO operation consists of four steps:

1. **2D Fourier Transform:**

   .. math::
       \mathbf{X}_{\text{freq}} = \mathcal{F}_{2D}(\mathbf{X})

   where :math:`\mathbf{X}_{\text{freq}} \in \mathbb{C}^{n \times d}` is the frequency representation.

2. **Mode Truncation:**

   .. math::
       \mathbf{X}_{\text{trunc}} = \mathbf{X}_{\text{freq}}[0:k_n, 0:k_d]

   where :math:`k_n \ll n` and :math:`k_d \ll d` are the number of retained modes,
   resulting in :math:`\mathbf{X}_{\text{trunc}} \in \mathbb{C}^{k_n \times k_d}`.

3. **Frequency Domain MLP:**

   .. math::
       \mathbf{Y}_{\text{freq}} = \text{MLP}(\mathbf{X}_{\text{trunc}}) \odot \mathbf{X}_{\text{trunc}}

   where :math:`\odot` denotes element-wise (Hadamard) multiplication and the MLP
   operates on complex values with expansion ratio :math:`r`:

   .. math::
       \text{MLP}(\mathbf{z}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2

   with :math:`\mathbf{W}_1 \in \mathbb{C}^{rk_d \times k_d}`, :math:`\mathbf{W}_2 \in \mathbb{C}^{k_d \times rk_d}`.

4. **Zero-padding and Inverse Transform:**

   .. math::
       \mathbf{Y} = \Re\left(\mathcal{F}_{2D}^{-1}(\text{pad}(\mathbf{Y}_{\text{freq}}))\right)

   where :math:`\text{pad}` zero-pads to original dimensions :math:`n \times d` and
   :math:`\Re(\cdot)` takes the real part.

**Complete Layer Operations:**

For each AFNO layer :math:`l`, the computation proceeds as:

.. math::
    \mathbf{Z}_l = \mathbf{X}_l + \text{AFNO}(\text{LayerNorm}(\mathbf{X}_l))

.. math::
    \mathbf{X}_{l+1} = \mathbf{Z}_l + \text{FFN}(\text{LayerNorm}(\mathbf{Z}_l))

where FFN follows the same structure as in standard transformers.

**Complexity Analysis:**

- Time Complexity: :math:`O(L \cdot (nd \log(nd) + k_n k_d d))` where :math:`L` is the number of layers
- Space Complexity: :math:`O(L \cdot k_n \cdot k_d \cdot d)`
- Memory reduction from :math:`O(nd)` to :math:`O(k_n k_d)` per layer through mode truncation

The mode truncation significantly reduces memory usage, with typical settings using
:math:`k_n = n/4` and :math:`k_d = d/2` achieving 8x memory reduction while maintaining
performance.

References
----------
.. [1] Guibas et al., "Adaptive Fourier Neural Operators: Efficient Token Mixers
       for Transformers", ICLR 2022.

See Also
--------
spectrans.layers.mixing.afno : AFNO mixing layer implementation.
spectrans.layers.operators.fno : Related Fourier Neural Operator implementation.
spectrans.models.base : Base model classes.
"""

from typing import TYPE_CHECKING

import torch.nn as nn

from ..blocks.base import PreNormBlock
from ..core.registry import register_component
from ..core.types import OutputHeadType, PositionalEncodingType
from ..layers.mixing.afno import AFNOMixing
from ..models.base import BaseModel

if TYPE_CHECKING:
    from ..config.models import AFNOModelConfig


@register_component("model", "afno")
class AFNOModel(BaseModel):
    """Adaptive Fourier Neural Operator transformer model.

    AFNO performs efficient token mixing using truncated Fourier modes
    and learnable MLPs in the frequency domain, achieving excellent
    performance on long sequences with reduced computational cost.

    Parameters
    ----------
    vocab_size : int | None, optional
        Size of the vocabulary for token embeddings. If None, expects
        pre-embedded inputs.
    hidden_dim : int, optional
        Hidden dimension size. Default is 768.
    num_layers : int, optional
        Number of AFNO layers. Default is 12.
    max_sequence_length : int, optional
        Maximum sequence length. Default is 1024.
    modes_seq : int | None, optional
        Number of Fourier modes to keep in sequence dimension.
        If None, defaults to max_sequence_length // 2.
    modes_hidden : int | None, optional
        Number of Fourier modes to keep in hidden dimension.
        If None, defaults to hidden_dim // 2.
    mlp_ratio : float, optional
        Expansion ratio for MLP in Fourier domain. Default is 2.0.
    num_classes : int | None, optional
        Number of output classes for classification. Default is None.
    use_positional_encoding : bool, optional
        Whether to use positional encoding. Default is True.
    positional_encoding_type : str, optional
        Type of positional encoding: 'sinusoidal' or 'learned'. Default is 'sinusoidal'.
    dropout : float, optional
        Dropout probability. Default is 0.1.
    ffn_hidden_dim : int | None, optional
        Hidden dimension for FFN. If None, defaults to 4 * hidden_dim.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.
    output_type : str, optional
        Type of output head: 'classification', 'regression', 'sequence', or 'none'.
        Default is 'classification'.
    gradient_checkpointing : bool, optional
        Whether to use gradient checkpointing. Default is False.

    Attributes
    ----------
    modes_seq : int
        Number of Fourier modes in sequence dimension.
    modes_hidden : int
        Number of Fourier modes in hidden dimension.
    mlp_ratio : float
        MLP expansion ratio in frequency domain.
    blocks : nn.ModuleList
        List of AFNO transformer blocks.
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_dim: int = 768,
        num_layers: int = 12,
        max_sequence_length: int = 1024,
        modes_seq: int | None = None,
        modes_hidden: int | None = None,
        mlp_ratio: float = 2.0,
        num_classes: int | None = None,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        dropout: float = 0.1,
        ffn_hidden_dim: int | None = None,
        norm_eps: float = 1e-12,
        output_type: OutputHeadType = "classification",
        gradient_checkpointing: bool = False,
    ):
        self.modes_seq = modes_seq or (max_sequence_length // 2)
        self.modes_hidden = modes_hidden or (hidden_dim // 2)
        self.mlp_ratio = mlp_ratio
        self._dropout_rate = dropout  # Store dropout rate for build_blocks

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
        """Build AFNO transformer blocks with adaptive Fourier mixing.

        Returns
        -------
        nn.ModuleList
            List of AFNO transformer blocks.
        """
        blocks = []
        for _ in range(self.num_layers):
            # Create AFNO mixing layer
            mixing_layer = AFNOMixing(
                hidden_dim=self.hidden_dim,
                max_sequence_length=self.max_sequence_length,
                modes_seq=self.modes_seq,
                modes_hidden=self.modes_hidden,
                mlp_ratio=self.mlp_ratio,
                dropout=self._dropout_rate,
            )

            # Create AFNO block with pre-normalization
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
        """Computational complexity of AFNO model.

        Returns
        -------
        dict[str, str]
            Dictionary with 'time' and 'space' complexity.
        """
        n = self.max_sequence_length
        d = self.hidden_dim
        k_n = self.modes_seq
        k_d = self.modes_hidden
        L = self.num_layers

        # AFNO has reduced complexity due to mode truncation
        # FFT: O(nd log(nd)), Mode operations: O(k_n * k_d * d)
        return {
            "time": f"O({L} * ({n}*{d}*log({n}*{d}) + {k_n}*{k_d}*{d}))",
            "space": f"O({L} * {k_n} * {k_d} * {d})",
        }

    @classmethod
    def from_config(cls, config: "AFNOModelConfig") -> "AFNOModel":  # type: ignore[override]
        """Create AFNO model from configuration.

        Parameters
        ----------
        config : AFNOModelConfig
            Configuration object with model parameters.

        Returns
        -------
        AFNOModel
            Configured AFNO model.
        """
        # Handle AFNO-specific mode configuration
        n_modes = getattr(config, "n_modes", None)
        modes_seq = getattr(config, "modes_seq", None)
        modes_hidden = getattr(config, "modes_hidden", None)

        # If n_modes is provided but not modes_seq/modes_hidden, compute them
        if n_modes is not None and modes_seq is None:
            modes_seq = n_modes
        if n_modes is not None and modes_hidden is None:
            compression_ratio = getattr(config, "compression_ratio", 0.5)
            modes_hidden = int(n_modes * compression_ratio)

        return cls(
            vocab_size=getattr(config, "vocab_size", None),
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            max_sequence_length=config.sequence_length,
            modes_seq=modes_seq,
            modes_hidden=modes_hidden,
            mlp_ratio=getattr(config, "mlp_ratio", 2.0),
            num_classes=getattr(config, "num_classes", None),
            use_positional_encoding=getattr(config, "use_positional_encoding", True),
            positional_encoding_type=getattr(config, "positional_encoding_type", "sinusoidal"),
            dropout=config.dropout,
            ffn_hidden_dim=getattr(config, "ffn_hidden_dim", None),
            norm_eps=getattr(config, "norm_eps", 1e-12),
            output_type=getattr(config, "output_type", "classification"),
            gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
        )


@register_component("model", "afno_encoder")
class AFNOEncoder(AFNOModel):
    """Encoder-only AFNO model for representation learning.

    This variant of AFNO is designed for tasks that require extracting
    representations rather than making predictions. It's particularly
    efficient for processing very long sequences due to the mode truncation.

    Parameters
    ----------
    hidden_dim : int, optional
        Hidden dimension size. Default is 768.
    num_layers : int, optional
        Number of AFNO layers. Default is 12.
    max_sequence_length : int, optional
        Maximum sequence length. Default is 1024.
    modes_seq : int | None, optional
        Number of Fourier modes in sequence dimension.
    modes_hidden : int | None, optional
        Number of Fourier modes in hidden dimension.
    mlp_ratio : float, optional
        MLP expansion ratio. Default is 2.0.
    use_positional_encoding : bool, optional
        Whether to use positional encoding. Default is True.
    positional_encoding_type : str, optional
        Type of positional encoding. Default is 'sinusoidal'.
    dropout : float, optional
        Dropout probability. Default is 0.1.
    ffn_hidden_dim : int | None, optional
        Hidden dimension for FFN. If None, defaults to 4 * hidden_dim.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.
    gradient_checkpointing : bool, optional
        Whether to use gradient checkpointing. Default is False.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        max_sequence_length: int = 1024,
        modes_seq: int | None = None,
        modes_hidden: int | None = None,
        mlp_ratio: float = 2.0,
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
            modes_seq=modes_seq,
            modes_hidden=modes_hidden,
            mlp_ratio=mlp_ratio,
            num_classes=None,  # No classification head
            use_positional_encoding=use_positional_encoding,
            positional_encoding_type=positional_encoding_type,
            dropout=dropout,
            ffn_hidden_dim=ffn_hidden_dim,
            norm_eps=norm_eps,
            output_type="none",  # Return hidden states
            gradient_checkpointing=gradient_checkpointing,
        )

    # Inherit forward from BaseModel - no need to override
