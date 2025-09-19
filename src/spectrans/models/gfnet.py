r"""Global Filter Networks (GFNet) for efficient spectral transformers.

This module implements the Global Filter Network architecture, which uses learnable
complex-valued filters in the frequency domain for token mixing. GFNet provides
a learnable alternative to FNet while maintaining $O(n \log n)$ complexity.

The architecture applies learnable filters in the Fourier domain, enabling
the model to selectively emphasize or suppress different frequency components
while maintaining computational efficiency compared to attention mechanisms.

Classes
-------
GFNet
    Complete GFNet model with global filter mixing layers.
GFNetEncoder
    Encoder-only GFNet for representation learning.

Examples
--------
Basic GFNet usage for classification:

>>> from spectrans.models.gfnet import GFNet
>>> model = GFNet(
...     vocab_size=30000,
...     hidden_dim=768,
...     num_layers=12,
...     max_sequence_length=512,
...     num_classes=2
... )
>>> input_ids = torch.randint(0, 30000, (8, 512))
>>> logits = model(input_ids=input_ids)
>>> assert logits.shape == (8, 2)

Using GFNet encoder:

>>> from spectrans.models.gfnet import GFNetEncoder
>>> encoder = GFNetEncoder(
...     hidden_dim=768,
...     num_layers=12,
...     max_sequence_length=512
... )
>>> inputs = torch.randn(8, 512, 768)
>>> features = encoder(inputs_embeds=inputs)
>>> assert features.shape == (8, 512, 768)

Creating from configuration:

>>> from spectrans.config.models import GFNetModelConfig
>>> config = GFNetModelConfig(
...     hidden_dim=768,
...     num_layers=12,
...     sequence_length=512
... )
>>> model = GFNet.from_config(config)

Notes
-----
Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

Given input tensor $\mathbf{X} \in \mathbb{R}^{n \times d}$ where $n$
is sequence length and $d$ is hidden dimension, GFNet applies learnable
complex filters in the frequency domain.

**Global Filter Operation:**

The core filtering operation is defined as:

$$
\text{GF}(\mathbf{X}) = \mathcal{F}^{-1}(\mathbf{H} \odot \mathcal{F}(\mathbf{X}))
$$

where:

- $\mathbf{H} \in \mathbb{C}^{n \times d}$ is a learnable complex-valued filter
- $\odot$ denotes element-wise (Hadamard) multiplication
- $\mathcal{F}$ and $\mathcal{F}^{-1}$ are FFT and IFFT along sequence dimension

**Filter Parameterization:**

The learnable filter $\mathbf{H}$ is parameterized as:

$$
\mathbf{H} = \sigma(\mathbf{W}_r + i\mathbf{W}_i)
$$

where:

- $\mathbf{W}_r, \mathbf{W}_i \in \mathbb{R}^{n \times d}$ are real-valued learnable parameters
- $\sigma$ is an activation function (typically sigmoid or tanh)
- $i$ is the imaginary unit

**Complete Layer Operations:**

For each GFNet layer $l$, the computation proceeds as:

$$
\mathbf{Z}_l = \mathbf{X}_l + \text{GF}(\text{LayerNorm}(\mathbf{X}_l))
$$

$$
\mathbf{X}_{l+1} = \mathbf{Z}_l + \text{FFN}(\text{LayerNorm}(\mathbf{Z}_l))
$$

where FFN follows the same structure as in FNet.

**Complexity Analysis:**

- Time Complexity: $O(L \cdot n \log n \cdot d)$ where $L$ is the number of layers
- Space Complexity: $O(L \cdot n \cdot d)$
- Learnable Parameters: $O(2nd)$ for the complex filter per layer

References
----------
Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou. 2021.
Global filter networks for image classification. In Advances in Neural
Information Processing Systems 34 (NeurIPS 2021), pages 980-993.

See Also
--------
spectrans.layers.mixing.global_filter : Global filter mixing layer implementation.
spectrans.models.base : Base model classes.
"""

from typing import TYPE_CHECKING

import torch.nn as nn

from ..blocks.base import PreNormBlock
from ..core.registry import register_component
from ..core.types import FilterActivationType, OutputHeadType, PositionalEncodingType
from ..layers.mixing.global_filter import GlobalFilterMixing
from ..models.base import BaseModel

if TYPE_CHECKING:
    from ..config.models import GFNetModelConfig


@register_component("model", "gfnet")
class GFNet(BaseModel):
    r"""Global Filter Network model with learnable frequency domain filters.

    GFNet uses learnable complex filters in the Fourier domain for
    token mixing, maintaining $O(n \log n)$ complexity.

    Parameters
    ----------
    vocab_size : int | None, optional
        Size of the vocabulary for token embeddings. If None, expects
        pre-embedded inputs.
    hidden_dim : int, optional
        Hidden dimension size. Default is 768.
    num_layers : int, optional
        Number of GFNet layers. Default is 12.
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
    filter_activation : str, optional
        Activation function for filters: 'sigmoid' or 'tanh'. Default is 'sigmoid'.
    gradient_checkpointing : bool, optional
        Whether to use gradient checkpointing. Default is False.

    Attributes
    ----------
    filter_activation : str
        Activation function used for filters.
    blocks : nn.ModuleList
        List of GFNet transformer blocks.
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
        filter_activation: FilterActivationType = "sigmoid",
        gradient_checkpointing: bool = False,
    ):
        self.filter_activation = filter_activation
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
        """Build GFNet transformer blocks with global filter mixing.

        Returns
        -------
        nn.ModuleList
            List of GFNet transformer blocks.
        """
        blocks = []
        for _ in range(self.num_layers):
            # Create global filter mixing layer
            mixing_layer = GlobalFilterMixing(
                hidden_dim=self.hidden_dim,
                sequence_length=self.max_sequence_length,
                activation=self.filter_activation,  # Note: parameter is 'activation' not 'filter_activation'
                dropout=self._dropout_rate,
            )

            # Create GFNet block with pre-normalization
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
    def from_config(cls, config: "GFNetModelConfig") -> "GFNet":  # type: ignore[override]
        """Create GFNet model from configuration.

        Parameters
        ----------
        config : GFNetModelConfig
            Configuration object with model parameters.

        Returns
        -------
        GFNet
            Configured GFNet model.
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
            filter_activation=getattr(config, "filter_activation", "sigmoid"),
            gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
        )


@register_component("model", "gfnet_encoder")
class GFNetEncoder(GFNet):
    """Encoder-only GFNet model for representation learning.

    This variant of GFNet is designed for tasks that require extracting
    representations rather than making predictions. It returns the
    hidden states from the final layer without any task-specific head.

    Parameters
    ----------
    hidden_dim : int, optional
        Hidden dimension size. Default is 768.
    num_layers : int, optional
        Number of GFNet layers. Default is 12.
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
    filter_activation : str, optional
        Activation function for filters. Default is 'sigmoid'.
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
        filter_activation: FilterActivationType = "sigmoid",
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
            filter_activation=filter_activation,
            gradient_checkpointing=gradient_checkpointing,
        )

    # Inherit forward from BaseModel - no need to override
