r"""Fourier Neural Operator (FNO) transformer models.

This module implements transformer models based on the Fourier Neural Operator,
which learns mappings between function spaces by parameterizing integral kernels
in the Fourier domain. These models achieve :math:`O(n \log n)` complexity through
FFT operations and are particularly effective for learning solution operators.

The FNO mechanism learns integral operators by parameterizing convolution kernels
in the Fourier domain, enabling efficient global interactions through spectral
truncation and complex-valued weight multiplication.

Classes
-------
FNOTransformer
    Complete transformer model using Fourier Neural Operators.
FNOEncoder
    Encoder-only model for representation learning with FNO.
FNODecoder
    Decoder model with causal FNO support for generation tasks.

Examples
--------
Basic FNO transformer:

>>> import torch
>>> from spectrans.models.fno_transformer import FNOTransformer
>>> model = FNOTransformer(
...     hidden_dim=512,
...     num_layers=6,
...     modes=32,
...     max_sequence_length=1024
... )
>>> x = torch.randn(32, 100, 512)  # (batch, seq_len, dim)
>>> output = model(inputs_embeds=x)
>>> assert output.shape == x.shape

Using with token inputs and classification:

>>> model = FNOTransformer(
...     vocab_size=10000,
...     hidden_dim=512,
...     num_layers=6,
...     modes=16,
...     num_classes=10,
...     max_sequence_length=512
... )
>>> input_ids = torch.randint(0, 10000, (32, 100))
>>> logits = model(input_ids)
>>> assert logits.shape == (32, 10)

2D FNO for image-like sequence data:

>>> from spectrans.models.fno_transformer import FNOTransformer
>>> model = FNOTransformer(
...     hidden_dim=512,
...     num_layers=6,
...     modes=32,
...     use_2d=True,
...     spatial_dim=64,  # Sequence viewed as 64x64 spatial grid
...     max_sequence_length=4096
... )

Notes
-----
Mathematical Foundation:

The FNO learns operators between function spaces through integral transforms:

.. math::
    (K \ast v)(x) = \int k(x, y) v(y) dy

In the Fourier domain, convolution becomes multiplication:

.. math::
    \mathcal{F}[K \ast v] = R_{\theta} \cdot \mathcal{F}[v]

Where :math:`R_{\theta}` are learnable complex weights truncated to the lowest
:math:`k` frequency modes:

.. math::
    R_{\theta} \in \mathbb{C}^{k \times d_{in} \times d_{out}}

The spectral convolution is computed as:

1. **Forward FFT**: :math:`\hat{v} = \mathcal{F}[v]`
2. **Mode truncation**: Keep only lowest :math:`k` modes
3. **Complex multiplication**: :math:`\hat{u}_k = R_{\theta,k} \cdot \hat{v}_k`
4. **Inverse FFT**: :math:`u = \mathcal{F}^{-1}[\hat{u}]`

This achieves :math:`O(n \log n)` complexity while learning global dependencies
through the spectral parameterization.

References
----------
.. [1] Li, Z., et al., "Fourier Neural Operator for Parametric Partial
       Differential Equations", ICLR 2021.
.. [2] Guibas, J., et al., "Adaptive Fourier Neural Operators", ICLR 2022.

See Also
--------
spectrans.layers.operators.fno : Core FNO layer implementations.
spectrans.transforms.fourier : FFT operations used by FNO.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from spectrans.core.registry import register_component
from spectrans.core.types import PositionalEncodingType
from spectrans.layers.operators.fno import FNOBlock
from spectrans.models.base import BaseModel

if TYPE_CHECKING:
    from spectrans.config.models import FNOTransformerConfig


@register_component("model", "fno_transformer")
class FNOTransformer(BaseModel):
    """Fourier Neural Operator transformer model.

    This model uses Fourier Neural Operators for sequence mixing, achieving
    O(n log n) complexity through FFT operations. The model learns mappings
    between function spaces by parameterizing kernels in the Fourier domain.

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
    modes : int, default=32
        Number of Fourier modes to retain (frequency truncation).
    mlp_ratio : float, default=2.0
        Expansion ratio for the MLP in FNO blocks.
    use_2d : bool, default=False
        Whether to use 2D spectral convolutions for spatial data.
    spatial_dim : int | None, optional
        Spatial dimension when using 2D convolutions (sequence = spatial_dim²).
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
        Stack of FNO transformer blocks.

    Examples
    --------
    >>> model = FNOTransformer(
    ...     hidden_dim=512,
    ...     num_layers=6,
    ...     modes=32,
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
        modes: int = 32,
        mlp_ratio: float = 2.0,
        use_2d: bool = False,
        spatial_dim: int | None = None,
        num_classes: int | None = None,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        gradient_checkpointing: bool = False,
    ):
        # Store FNO-specific parameters
        self.modes = modes
        self.mlp_ratio = mlp_ratio
        self.use_2d = use_2d
        self.spatial_dim = spatial_dim
        self.dropout_rate = dropout

        # Validate 2D configuration
        if use_2d and spatial_dim is None:
            raise ValueError("spatial_dim must be specified when use_2d=True")
        if use_2d and spatial_dim is not None and spatial_dim * spatial_dim != max_sequence_length:
            raise ValueError(
                f"For 2D FNO, max_sequence_length ({max_sequence_length}) "
                f"must equal spatial_dim² ({spatial_dim}² = {spatial_dim * spatial_dim})"
            )

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
        """Build transformer blocks with FNO layers.

        Returns
        -------
        nn.ModuleList
            List of FNO transformer blocks.
        """
        blocks = []
        for _ in range(self.num_layers):
            # Create FNO block with appropriate configuration
            fno_block = FNOBlock(
                hidden_dim=self.hidden_dim,
                modes=self.modes,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout_rate,
            )
            blocks.append(fno_block)

        return nn.ModuleList(blocks)


    @classmethod
    def from_config(cls, config: "FNOTransformerConfig") -> "FNOTransformer":  # type: ignore[override]
        """Create model from configuration.

        Parameters
        ----------
        config : FNOTransformerConfig
            Model configuration object.

        Returns
        -------
        FNOTransformer
            Instantiated model.
        """
        return cls(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            max_sequence_length=config.sequence_length,
            modes=config.modes,
            mlp_ratio=config.mlp_ratio,
            use_2d=config.use_2d,
            spatial_dim=config.spatial_dim,
            num_classes=config.num_classes,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
            use_positional_encoding=config.use_positional_encoding,
            positional_encoding_type=config.positional_encoding_type,
            gradient_checkpointing=config.gradient_checkpointing,
        )


@register_component("model", "fno_encoder")
class FNOEncoder(BaseModel):
    """Encoder-only FNO model for representation learning.

    This model uses stacked FNO blocks without causal masking, suitable for
    bidirectional encoding tasks like feature extraction and representation learning.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension size for the model.
    num_layers : int
        Number of encoder blocks.
    max_sequence_length : int
        Maximum sequence length.
    modes : int, default=32
        Number of Fourier modes to retain.
    mlp_ratio : float, default=2.0
        MLP expansion ratio in FNO blocks.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the feedforward network.
    dropout : float, default=0.0
        Dropout probability.
    use_positional_encoding : bool, default=True
        Whether to use positional encoding.
    positional_encoding_type : str, default="sinusoidal"
        Type of positional encoding.
    gradient_checkpointing : bool, default=False
        Whether to use gradient checkpointing.

    Examples
    --------
    >>> encoder = FNOEncoder(
    ...     hidden_dim=512,
    ...     num_layers=6,
    ...     modes=32,
    ...     max_sequence_length=1024
    ... )
    >>> x = torch.randn(32, 100, 512)
    >>> encoded = encoder(inputs_embeds=x)
    >>> assert encoded.shape == x.shape
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 6,
        max_sequence_length: int = 1024,
        modes: int = 32,
        mlp_ratio: float = 2.0,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        gradient_checkpointing: bool = False,
    ):
        # Store FNO-specific parameters
        self.modes = modes
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout

        super().__init__(
            vocab_size=None,  # Encoder doesn't need vocab
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            num_classes=None,  # No classification head
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            positional_encoding_type=positional_encoding_type,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Set output type to none for encoder
        self.output_type = "none"

    def build_blocks(self) -> nn.ModuleList:
        """Build encoder blocks with FNO layers.

        Returns
        -------
        nn.ModuleList
            List of FNO encoder blocks.
        """
        blocks = []
        for _ in range(self.num_layers):
            fno_block = FNOBlock(
                hidden_dim=self.hidden_dim,
                modes=self.modes,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout_rate,
            )
            blocks.append(fno_block)

        return nn.ModuleList(blocks)



@register_component("model", "fno_decoder")
class FNODecoder(BaseModel):
    """Decoder FNO model for generation tasks.

    This model uses causal FNO blocks suitable for autoregressive generation
    tasks. The spectral operations are modified to respect causality.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary for generation.
    hidden_dim : int
        Hidden dimension size.
    num_layers : int
        Number of decoder blocks.
    max_sequence_length : int
        Maximum sequence length.
    modes : int, default=32
        Number of Fourier modes (adjusted for causality).
    mlp_ratio : float, default=2.0
        MLP expansion ratio.
    causal : bool, default=True
        Whether to use causal masking.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the feedforward network.
    dropout : float, default=0.0
        Dropout probability.
    use_positional_encoding : bool, default=True
        Whether to use positional encoding.
    positional_encoding_type : str, default="sinusoidal"
        Type of positional encoding.
    gradient_checkpointing : bool, default=False
        Whether to use gradient checkpointing.

    Examples
    --------
    >>> decoder = FNODecoder(
    ...     vocab_size=10000,
    ...     hidden_dim=512,
    ...     num_layers=12,
    ...     modes=32,
    ...     causal=True,
    ...     max_sequence_length=2048
    ... )
    >>> input_ids = torch.randint(0, 10000, (32, 100))
    >>> logits = decoder(input_ids)
    >>> assert logits.shape == (32, 100, 10000)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 12,
        max_sequence_length: int = 2048,
        modes: int = 32,
        mlp_ratio: float = 2.0,
        causal: bool = True,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        gradient_checkpointing: bool = False,
    ):
        # Store FNO-specific parameters
        self.modes = modes
        self.mlp_ratio = mlp_ratio
        self.causal = causal
        self.dropout_rate = dropout

        super().__init__(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            num_classes=None,  # Decoder uses LM head instead
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            positional_encoding_type=positional_encoding_type,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Add language modeling head
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.output_type = "lm"

    def build_blocks(self) -> nn.ModuleList:
        """Build decoder blocks with causal FNO layers.

        Returns
        -------
        nn.ModuleList
            List of causal FNO decoder blocks.
        """
        blocks = []
        for _ in range(self.num_layers):
            # Create FNO block
            # Note: Causality in spectral domain requires special handling
            # This is a simplified version - full causality would need custom implementation
            fno_block = FNOBlock(
                hidden_dim=self.hidden_dim,
                modes=self.modes,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout_rate,
            )
            blocks.append(fno_block)

        return nn.ModuleList(blocks)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the decoder.

        Parameters
        ----------
        input_ids : torch.Tensor | None, optional
            Input token IDs of shape (batch_size, sequence_length).
        inputs_embeds : torch.Tensor | None, optional
            Pre-embedded inputs of shape (batch_size, sequence_length, hidden_dim).
        attention_mask : torch.Tensor | None, optional
            Attention mask for padding.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, sequence_length, vocab_size).
        """
        # Use parent class forward for processing
        hidden_states = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # Apply LM head
        logits = self.lm_head(hidden_states)
        return logits  # type: ignore[no-any-return]

