r"""Spectral Attention transformer models using kernel approximations.

This module implements transformer models based on spectral attention mechanisms
that use Random Fourier Features (RFF) to linearize attention computation. These
models achieve :math:`O(n)` complexity instead of the quadratic :math:`O(n^2)`
complexity of standard transformers, making them efficient for long sequences.

The spectral attention mechanism approximates the softmax kernel using random
feature maps, maintaining the expressive power of attention while dramatically
reducing computational cost.

Classes
-------
SpectralAttentionTransformer
    Complete transformer model using spectral attention layers.
SpectralAttentionEncoder
    Encoder-only model for representation learning.
PerformerTransformer
    Performer-style model with positive orthogonal random features.

Examples
--------
Basic spectral attention transformer:

>>> import torch
>>> from spectrans.models.spectral_attention import SpectralAttentionTransformer
>>> model = SpectralAttentionTransformer(
...     hidden_dim=512,
...     num_layers=6,
...     num_heads=8,
...     num_features=256,
...     max_sequence_length=1024
... )
>>> x = torch.randn(32, 100, 512)  # (batch, seq_len, dim)
>>> output = model(inputs_embeds=x)
>>> assert output.shape == x.shape

Using with token inputs and classification head:

>>> model = SpectralAttentionTransformer(
...     vocab_size=10000,
...     hidden_dim=512,
...     num_layers=6,
...     num_heads=8,
...     num_classes=10,
...     max_sequence_length=512
... )
>>> input_ids = torch.randint(0, 10000, (32, 100))
>>> logits = model(input_ids)
>>> assert logits.shape == (32, 10)

Performer model with orthogonal features:

>>> from spectrans.models.spectral_attention import PerformerTransformer
>>> performer = PerformerTransformer(
...     hidden_dim=512,
...     num_layers=6,
...     num_heads=8,
...     num_features=256,
...     use_orthogonal=True
... )

Notes
-----
Mathematical Foundation:

The spectral attention mechanism approximates standard attention as:

.. math::
    \text{Attention}(Q, K, V) \approx \boldsymbol{\Phi}(Q)
    \left(\boldsymbol{\Phi}(K)^T V\right) / Z

Where :math:`\boldsymbol{\Phi}` is a random feature map:

.. math::
    \boldsymbol{\varphi}(x) = \sqrt{\frac{2}{D}} \begin{bmatrix}
    \cos(\boldsymbol{\omega}_1^T x + b_1) \\
    \cos(\boldsymbol{\omega}_2^T x + b_2) \\
    \vdots \\
    \cos(\boldsymbol{\omega}_D^T x + b_D)
    \end{bmatrix}

With random frequencies :math:`\omega_i \sim \mathcal{N}(0, \sigma^2 I)` and
phases :math:`b_i \sim \text{Uniform}[0, 2\pi]`.

The approximation quality improves with more random features :math:`D`, with
error decreasing as :math:`O(1/\sqrt{D})`. The linear complexity :math:`O(nDd)`
becomes favorable over standard attention :math:`O(n^2d)` when :math:`D \ll n`.

For the Performer variant, orthogonal random features are used to reduce
the variance of the approximation, leading to better convergence.

References
----------
.. [1] Choromanski, K. et al., "Rethinking Attention with Performers",
       ICLR 2021.
.. [2] Peng, H. et al., "Random Feature Attention", ICLR 2021.
.. [3] Rahimi, A. & Recht, B., "Random Features for Large-Scale Kernel
       Machines", NeurIPS 2007.

See Also
--------
spectrans.layers.attention.spectral : Spectral attention layer implementations.
spectrans.kernels.rff : Random Fourier Features kernel approximations.
spectrans.models.lst : Linear Spectral Transform models for comparison.
"""

from typing import TYPE_CHECKING

import torch.nn as nn

from ..blocks.base import PreNormBlock
from ..core.registry import register_component
from ..core.types import KernelType, PositionalEncodingType
from ..layers.attention.spectral import PerformerAttention, SpectralAttention
from .base import BaseModel

if TYPE_CHECKING:
    from spectrans.config.models import SpectralAttentionModelConfig


@register_component("model", "spectral_attention")
class SpectralAttentionTransformer(BaseModel):
    """Spectral Attention transformer using Random Fourier Features.

    This model uses spectral attention layers with RFF approximation to achieve
    linear complexity attention computation. The model maintains the expressive
    power of standard transformers while being efficient for long sequences.

    Parameters
    ----------
    vocab_size : int | None, optional
        Size of the vocabulary for token embeddings. If None, expects pre-embedded inputs.
    hidden_dim : int
        Hidden dimension size for the model.
    num_layers : int
        Number of transformer blocks.
    max_sequence_length : int
        Maximum sequence length the model can process.
    num_heads : int, default=8
        Number of attention heads.
    num_features : int | None, optional
        Number of random features for RFF approximation. If None, uses hidden_dim.
    kernel_type : KernelType, default="softmax"
        Type of kernel to approximate.
    use_orthogonal : bool, default=False
        Whether to use orthogonal random features.
    num_classes : int | None, optional
        Number of output classes for classification.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the feedforward network. Default is 4 * hidden_dim.
    dropout : float, default=0.0
        Dropout probability.
    use_positional_encoding : bool, default=True
        Whether to use positional encoding.
    positional_encoding_type : str, default="sinusoidal"
        Type of positional encoding ("sinusoidal" or "learned").
    gradient_checkpointing : bool, default=False
        Whether to use gradient checkpointing to save memory.

    Attributes
    ----------
    blocks : nn.ModuleList
        Stack of spectral attention transformer blocks.
    complexity : ComplexityInfo
        Computational complexity information.

    Examples
    --------
    >>> model = SpectralAttentionTransformer(
    ...     hidden_dim=512,
    ...     num_layers=6,
    ...     num_heads=8,
    ...     num_features=256,
    ...     max_sequence_length=1024
    ... )
    >>> x = torch.randn(32, 100, 512)
    >>> output = model(inputs_embeds=x)
    >>> assert output.shape == x.shape
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_dim: int = 512,
        num_layers: int = 6,
        max_sequence_length: int = 1024,
        num_heads: int = 8,
        num_features: int | None = None,
        kernel_type: KernelType = "softmax",
        use_orthogonal: bool = False,
        num_classes: int | None = None,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        gradient_checkpointing: bool = False,
    ):
        # Store all parameters before calling super().__init__ since build_blocks needs them
        self.num_heads = num_heads
        self.num_features = num_features or hidden_dim
        self.kernel_type = kernel_type
        self.use_orthogonal = use_orthogonal
        self.dropout_rate = dropout  # Store as different name to avoid conflict

        super().__init__(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            num_classes=num_classes,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            positional_encoding_type=positional_encoding_type,
            gradient_checkpointing=gradient_checkpointing,
        )

    def build_blocks(self) -> nn.ModuleList:
        """Build transformer blocks with spectral attention layers.

        Returns
        -------
        nn.ModuleList
            List of spectral attention transformer blocks.
        """
        blocks = []
        for _ in range(self.num_layers):
            attention_layer = SpectralAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_features=self.num_features,
                kernel_type=self.kernel_type,
                use_orthogonal=self.use_orthogonal,
                dropout=self.dropout_rate,
            )

            block = PreNormBlock(
                mixing_layer=attention_layer,
                hidden_dim=self.hidden_dim,
                ffn_hidden_dim=self.ffn_hidden_dim,
                dropout=self.dropout_rate,
                norm_eps=1e-12,
            )
            blocks.append(block)

        return nn.ModuleList(blocks)

    @property
    def complexity(self) -> dict[str, str]:
        """Get computational complexity information.

        Returns
        -------
        dict[str, str]
            Complexity is O(n * D * d) where n is sequence length,
            D is number of random features, and d is hidden dimension.
        """
        return {
            "time": f"O(n * {self.num_features} * {self.hidden_dim})",
            "space": f"O(n * {self.num_features})",
            "description": "Linear complexity via RFF approximation",
        }

    @classmethod
    def from_config(cls, config: "SpectralAttentionModelConfig") -> "SpectralAttentionTransformer":  # type: ignore[override]
        """Create model from configuration.

        Parameters
        ----------
        config : SpectralAttentionModelConfig
            Model configuration object.

        Returns
        -------
        SpectralAttentionTransformer
            Configured model instance.
        """
        # Extract spectral attention specific config
        return cls(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            max_sequence_length=config.sequence_length,
            num_heads=config.num_heads,
            num_features=config.num_features,
            kernel_type=config.kernel_type,
            use_orthogonal=config.use_orthogonal,
            num_classes=config.num_classes,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
            use_positional_encoding=config.use_positional_encoding,
            positional_encoding_type=config.positional_encoding_type,
            gradient_checkpointing=config.gradient_checkpointing,
        )


@register_component("model", "spectral_attention_encoder")
class SpectralAttentionEncoder(BaseModel):
    """Encoder-only spectral attention model for representation learning.

    This model uses spectral attention layers without a classification head,
    suitable for generating embeddings or as a component in larger architectures.

    Parameters
    ----------
    vocab_size : int | None, optional
        Size of the vocabulary for token embeddings.
    hidden_dim : int
        Hidden dimension size.
    num_layers : int
        Number of transformer blocks.
    max_sequence_length : int
        Maximum sequence length.
    num_heads : int, default=8
        Number of attention heads.
    num_features : int | None, optional
        Number of random features.
    kernel_type : KernelType, default="softmax"
        Kernel type.
    use_orthogonal : bool, default=False
        Use orthogonal features.
    ffn_hidden_dim : int | None, optional
        FFN hidden dimension.
    dropout : float, default=0.0
        Dropout probability.
    use_positional_encoding : bool, default=True
        Use positional encoding.
    positional_encoding_type : str, default="sinusoidal"
        Positional encoding type.
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_dim: int = 512,
        num_layers: int = 6,
        max_sequence_length: int = 1024,
        num_heads: int = 8,
        num_features: int | None = None,
        kernel_type: KernelType = "softmax",
        use_orthogonal: bool = False,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
    ):
        # Store parameters before calling super().__init__ since build_blocks needs them
        self.num_heads = num_heads
        self.num_features = num_features or hidden_dim
        self.kernel_type = kernel_type
        self.use_orthogonal = use_orthogonal
        self.dropout_rate = dropout

        # Initialize without classification head
        super().__init__(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            num_classes=None,  # No classification head
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            positional_encoding_type=positional_encoding_type,
            gradient_checkpointing=False,
        )

        # Set output type to none for encoder
        self.output_type = "none"

    def build_blocks(self) -> nn.ModuleList:
        """Build encoder blocks with spectral attention.

        Returns
        -------
        nn.ModuleList
            List of spectral attention blocks.
        """
        blocks = []
        for _ in range(self.num_layers):
            attention_layer = SpectralAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_features=self.num_features,
                kernel_type=self.kernel_type,
                use_orthogonal=self.use_orthogonal,
                dropout=self.dropout_rate,
            )

            block = PreNormBlock(
                mixing_layer=attention_layer,
                hidden_dim=self.hidden_dim,
                ffn_hidden_dim=self.ffn_hidden_dim,
                dropout=self.dropout_rate,
                norm_eps=1e-12,
            )
            blocks.append(block)

        return nn.ModuleList(blocks)

    @property
    def complexity(self) -> dict[str, str]:
        """Get computational complexity information.

        Returns
        -------
        dict[str, str]
            Linear complexity information.
        """
        return {
            "time": f"O(n * {self.num_features} * {self.hidden_dim})",
            "space": f"O(n * {self.num_features})",
            "description": "Linear encoder via RFF approximation",
        }


@register_component("model", "performer")
class PerformerTransformer(BaseModel):
    """Performer transformer with positive orthogonal random features.

    This model implements the Performer architecture which uses positive
    orthogonal random features (PORF) to approximate the softmax kernel
    with improved variance reduction compared to standard RFF.

    Parameters
    ----------
    vocab_size : int | None, optional
        Vocabulary size.
    hidden_dim : int
        Hidden dimension.
    num_layers : int
        Number of layers.
    max_sequence_length : int
        Maximum sequence length.
    num_heads : int, default=8
        Number of heads.
    num_features : int | None, optional
        Number of random features.
    num_classes : int | None, optional
        Number of classes.
    ffn_hidden_dim : int | None, optional
        FFN dimension.
    dropout : float, default=0.0
        Dropout rate.
    use_positional_encoding : bool, default=True
        Use positional encoding.
    positional_encoding_type : str, default="sinusoidal"
        Positional encoding type.
    gradient_checkpointing : bool, default=False
        Use gradient checkpointing.

    Examples
    --------
    >>> performer = PerformerTransformer(
    ...     hidden_dim=512,
    ...     num_layers=6,
    ...     num_heads=8,
    ...     num_features=256,
    ...     max_sequence_length=1024
    ... )
    >>> x = torch.randn(32, 100, 512)
    >>> output = performer(inputs_embeds=x)
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_dim: int = 512,
        num_layers: int = 6,
        max_sequence_length: int = 1024,
        num_heads: int = 8,
        num_features: int | None = None,
        num_classes: int | None = None,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        gradient_checkpointing: bool = False,
    ):
        # Store parameters before calling super().__init__ since build_blocks needs them
        self.num_heads = num_heads
        self.num_features = num_features or hidden_dim
        self.dropout_rate = dropout

        super().__init__(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            num_classes=num_classes,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            positional_encoding_type=positional_encoding_type,
            gradient_checkpointing=gradient_checkpointing,
        )

    def build_blocks(self) -> nn.ModuleList:
        """Build Performer blocks with orthogonal features.

        Returns
        -------
        nn.ModuleList
            List of Performer blocks.
        """
        blocks = []
        for _ in range(self.num_layers):
            attention_layer = PerformerAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_features=self.num_features,
                dropout=self.dropout_rate,
            )

            block = PreNormBlock(
                mixing_layer=attention_layer,
                hidden_dim=self.hidden_dim,
                ffn_hidden_dim=self.ffn_hidden_dim,
                dropout=self.dropout_rate,
                norm_eps=1e-12,
            )
            blocks.append(block)

        return nn.ModuleList(blocks)

    @property
    def complexity(self) -> dict[str, str]:
        """Get computational complexity information.

        Returns
        -------
        dict[str, str]
            Linear complexity with orthogonal features.
        """
        return {
            "time": f"O(n * {self.num_features} * {self.hidden_dim})",
            "space": f"O(n * {self.num_features})",
            "description": "Performer with orthogonal random features",
        }
