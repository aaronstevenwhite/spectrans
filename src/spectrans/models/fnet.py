r"""FNet: Mixing Tokens with Fourier Transforms.

This module implements the FNet architecture, which replaces the self-attention
mechanism in transformers with Fourier transform-based token mixing. FNet
maintains $O(n \log n)$ computational complexity compared to $O(n^2)$ for
standard attention mechanisms.

The architecture uses 2D Discrete Fourier Transforms (DFT) to mix tokens,
enabling global information mixing across the sequence with reduced computational
cost compared to attention-based models.

Classes
-------
FNet
    Complete FNet model with Fourier mixing layers.
FNetEncoder
    Encoder-only FNet for representation learning.

Examples
--------
Basic FNet usage for classification:

>>> from spectrans.models.fnet import FNet
>>> model = FNet(
...     vocab_size=30000,
...     hidden_dim=768,
...     num_layers=12,
...     max_sequence_length=512,
...     num_classes=2
... )
>>> input_ids = torch.randint(0, 30000, (8, 512))
>>> logits = model(input_ids=input_ids)
>>> assert logits.shape == (8, 2)

Using FNet encoder for feature extraction:

>>> from spectrans.models.fnet import FNetEncoder
>>> encoder = FNetEncoder(hidden_dim=768, num_layers=12)
>>> inputs = torch.randn(8, 512, 768)
>>> features = encoder(inputs_embeds=inputs)
>>> assert features.shape == (8, 512, 768)

Creating from configuration:

>>> from spectrans.config.models import FNetModelConfig
>>> config = FNetModelConfig(
...     hidden_dim=768,
...     num_layers=12,
...     sequence_length=512
... )
>>> model = FNet.from_config(config)

Notes
-----
Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

Given input tensor $\mathbf{X} \in \mathbb{R}^{n \times d}$ where $n$
is sequence length and $d$ is hidden dimension, FNet applies the following
operations in each layer $l$:

**Fourier Mixing Operation:**

The core mixing operation is defined as:

$$
\text{FourierMix}(\mathbf{X}) = \Re\left(\mathcal{F}_d^{-1}\left(\mathcal{F}_n(\mathbf{X})\right)\right)
$$

where:

- $\mathcal{F}_n$ denotes 1D DFT along the sequence dimension
- $\mathcal{F}_d^{-1}$ denotes inverse 1D DFT along the feature dimension
- $\Re(\cdot)$ takes the real part of complex values

**Complete Layer Operations:**

For each FNet layer $l$, the computation proceeds as:

$$
\mathbf{H}_l = \text{LayerNorm}(\mathbf{X}_l + \text{FourierMix}(\mathbf{X}_l))
$$

$$
\mathbf{X}_{l+1} = \text{LayerNorm}(\mathbf{H}_l + \text{FFN}(\mathbf{H}_l))
$$

where the feedforward network (FFN) is:

$$
\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

with $\mathbf{W}_1 \in \mathbb{R}^{4d \times d}$, $\mathbf{b}_1 \in \mathbb{R}^{4d}$,
$\mathbf{W}_2 \in \mathbb{R}^{d \times 4d}$, $\mathbf{b}_2 \in \mathbb{R}^d$.

**Complexity Analysis:**

- Time Complexity: $O(L \cdot n \log n \cdot d)$ where $L$ is the number of layers
- Space Complexity: $O(L \cdot n \cdot d)$
- No learned parameters in the mixing operation (only in FFN and embeddings)

References
----------
James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon. 2022.
FNet: Mixing tokens with Fourier transforms. In Proceedings of the 2022 Conference
of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (NAACL-HLT), pages 4296-4313, Seattle.

See Also
--------
spectrans.layers.mixing.fourier : Fourier mixing layer implementation.
spectrans.models.base : Base model classes.
"""

from typing import TYPE_CHECKING

import torch.nn as nn

from ..blocks.base import PreNormBlock
from ..core.registry import register_component
from ..core.types import OutputHeadType, PositionalEncodingType
from ..layers.mixing.fourier import FourierMixing, RealFourierMixing
from ..models.base import BaseModel

if TYPE_CHECKING:
    from ..config.models import FNetModelConfig


@register_component("model", "fnet")
class FNet(BaseModel):
    r"""FNet model with Fourier transform-based token mixing.

    FNet replaces the self-attention mechanism with Fourier
    transforms, achieving $O(n \log n)$ complexity.

    Parameters
    ----------
    vocab_size : int | None, optional
        Size of the vocabulary for token embeddings. If None, expects
        pre-embedded inputs.
    hidden_dim : int, optional
        Hidden dimension size. Default is 768.
    num_layers : int, optional
        Number of FNet layers. Default is 12.
    max_sequence_length : int, optional
        Maximum sequence length. Default is 512.
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
    use_real_fft : bool, optional
        Whether to use real FFT for efficiency. Default is True.
    gradient_checkpointing : bool, optional
        Whether to use gradient checkpointing. Default is False.

    Attributes
    ----------
    use_real_fft : bool
        Whether real FFT is used for efficiency.
    blocks : nn.ModuleList
        List of FNet transformer blocks.
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_dim: int = 768,
        num_layers: int = 12,
        max_sequence_length: int = 512,
        num_classes: int | None = None,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        dropout: float = 0.1,
        ffn_hidden_dim: int | None = None,
        norm_eps: float = 1e-12,
        output_type: OutputHeadType = "classification",
        use_real_fft: bool = True,
        gradient_checkpointing: bool = False,
    ):
        self.use_real_fft = use_real_fft
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
        """Build FNet transformer blocks with Fourier mixing.

        Returns
        -------
        nn.ModuleList
            List of FNet transformer blocks.
        """
        blocks = []
        for _ in range(self.num_layers):
            # Choose mixing layer based on use_real_fft flag
            mixing_layer: FourierMixing | RealFourierMixing
            if self.use_real_fft:
                mixing_layer = RealFourierMixing(
                    hidden_dim=self.hidden_dim,
                    dropout=self._dropout_rate,
                )
            else:
                mixing_layer = FourierMixing(
                    hidden_dim=self.hidden_dim,
                    dropout=self._dropout_rate,
                )

            # Create FNet block with pre-normalization
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
    def from_config(cls, config: "FNetModelConfig") -> "FNet":  # type: ignore[override]
        """Create FNet model from configuration.

        Parameters
        ----------
        config : FNetModelConfig
            Configuration object with model parameters.

        Returns
        -------
        FNet
            Configured FNet model.
        """
        return cls(
            vocab_size=getattr(config, "vocab_size", None),
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            max_sequence_length=config.sequence_length,
            num_classes=getattr(config, "num_classes", None),
            use_positional_encoding=getattr(config, "use_positional_encoding", True),
            positional_encoding_type=getattr(config, "positional_encoding_type", "sinusoidal"),
            dropout=config.dropout,
            ffn_hidden_dim=getattr(config, "ffn_hidden_dim", None),
            norm_eps=getattr(config, "norm_eps", 1e-12),
            output_type=getattr(config, "output_type", "classification"),
            use_real_fft=getattr(config, "use_real_fft", True),
            gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
        )


@register_component("model", "fnet_encoder")
class FNetEncoder(FNet):
    """Encoder-only FNet model for representation learning.

    This variant of FNet is designed for tasks that require extracting
    representations rather than making predictions. It returns the
    hidden states from the final layer without any task-specific head.

    Parameters
    ----------
    hidden_dim : int, optional
        Hidden dimension size. Default is 768.
    num_layers : int, optional
        Number of FNet layers. Default is 12.
    max_sequence_length : int, optional
        Maximum sequence length. Default is 512.
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
    use_real_fft : bool, optional
        Whether to use real FFT. Default is True.
    gradient_checkpointing : bool, optional
        Whether to use gradient checkpointing. Default is False.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        max_sequence_length: int = 512,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        dropout: float = 0.1,
        ffn_hidden_dim: int | None = None,
        norm_eps: float = 1e-12,
        use_real_fft: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(
            vocab_size=None,  # No token embeddings for encoder
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            num_classes=None,  # No classification head
            use_positional_encoding=use_positional_encoding,
            positional_encoding_type=positional_encoding_type,
            dropout=dropout,
            ffn_hidden_dim=ffn_hidden_dim,
            norm_eps=norm_eps,
            output_type="none",  # Return hidden states
            use_real_fft=use_real_fft,
            gradient_checkpointing=gradient_checkpointing,
        )

    # Inherit forward from BaseModel - no need to override
