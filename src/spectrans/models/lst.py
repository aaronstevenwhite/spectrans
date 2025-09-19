r"""Linear Spectral Transform (LST) models using efficient transforms.

This module implements transformer models that use linear spectral transforms
(DCT, DST, Hadamard) for sequence mixing instead of attention mechanisms. These
models achieve $O(n \log n)$ complexity through Fast Fourier-like algorithms,
providing an efficient alternative to quadratic attention.

The LST mechanism applies learned transformations in the spectral domain,
enabling global token interactions while maintaining computational efficiency.

Classes
-------
LSTTransformer
    Complete transformer model using linear spectral transforms.
LSTEncoder
    Encoder-only model for representation learning.
LSTDecoder
    Decoder model with causal masking support.

Examples
--------
Basic LST transformer:

>>> import torch
>>> from spectrans.models.lst import LSTTransformer
>>> model = LSTTransformer(
...     hidden_dim=512,
...     num_layers=6,
...     transform_type="dct",
...     max_sequence_length=1024
... )
>>> x = torch.randn(32, 100, 512)  # (batch, seq_len, dim)
>>> output = model(inputs_embeds=x)
>>> assert output.shape == x.shape

Using with token inputs and classification:

>>> model = LSTTransformer(
...     vocab_size=10000,
...     hidden_dim=512,
...     num_layers=6,
...     transform_type="hadamard",
...     num_classes=10,
...     max_sequence_length=512
... )
>>> input_ids = torch.randint(0, 10000, (32, 100))
>>> logits = model(input_ids)
>>> assert logits.shape == (32, 10)

Causal decoder model:

>>> from spectrans.models.lst import LSTDecoder
>>> decoder = LSTDecoder(
...     vocab_size=10000,
...     hidden_dim=512,
...     num_layers=12,
...     transform_type="dst",
...     causal=True,
...     max_sequence_length=2048
... )

Notes
-----
Mathematical Foundation:

The LST mechanism replaces attention with spectral domain operations:

$$
\text{LST}(X) = \mathcal{T}^{-1}(\mathbf{W} \odot \mathcal{T}(X))
$$

Where:
- $\mathcal{T}$ is the forward spectral transform (DCT/DST/Hadamard)
- $\mathcal{T}^{-1}$ is the inverse transform
- $\mathbf{W}$ is a learned spectral weighting matrix
- $\odot$ denotes element-wise multiplication

The transforms have efficient $O(n \log n)$ implementations:

1. **DCT (Discrete Cosine Transform)**:

   $$
   X_k = \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k(2n+1)}{2N}\right)
   $$

2. **DST (Discrete Sine Transform)**:

   $$
   X_k = \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi (k+1)(n+1)}{N+1}\right)
   $$

3. **Hadamard Transform**:

   $$
   H_N = H_2 \otimes H_{\frac{N}{2}} = \begin{bmatrix}
   H_{\frac{N}{2}} & H_{\frac{N}{2}} \\
   H_{\frac{N}{2}} & -H_{\frac{N}{2}}
   \end{bmatrix}
   $$

The spectral weights enable frequency-selective filtering, allowing the model
to learn which frequency components are important for the task.

References
----------
James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon. 2022.
FNet: Mixing tokens with Fourier transforms. In Proceedings of the 2022 Conference
of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (NAACL-HLT), pages 4296-4313, Seattle.

Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham,
Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. 2021. Long range
arena: A benchmark for efficient transformers. In Advances in Neural Information
Processing Systems 34 (NeurIPS 2021).

Nasir Ahmed, T. Natarajan, and Kamisetty R. Rao. 1974. Discrete cosine transform.
IEEE Transactions on Computers, C-23(1):90-93.

See Also
--------
spectrans.layers.mixing : Spectral mixing layer implementations.
spectrans.transforms.spectral : Core spectral transform implementations.
spectrans.models.spectral_attention : Spectral attention models for comparison.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..blocks.base import PreNormBlock
from ..core.registry import register_component
from ..core.types import PositionalEncodingType, TransformLSTType
from ..layers.attention import DCTAttention, HadamardAttention, LSTAttention
from .base import BaseModel

if TYPE_CHECKING:
    from spectrans.config.models import LSTModelConfig


@register_component("model", "lst")
class LSTTransformer(BaseModel):
    """Linear Spectral Transform transformer model.

    This model uses linear spectral transforms (DCT/DST/Hadamard) for sequence
    mixing, achieving O(n log n) complexity through fast transform algorithms.
    The model applies learned transformations in the spectral domain for
    efficient global token interactions.

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
    transform_type : TransformLSTType, default="dct"
        Type of spectral transform to use.
    use_conv_bias : bool, default=True
        Whether to use bias in spectral convolution.
    num_classes : int | None, optional
        Number of output classes for classification.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the feedforward network. Default is 4 * hidden_dim.
    dropout : float, default=0.0
        Dropout probability.
    use_positional_encoding : bool, default=True
        Whether to use positional encoding.
    positional_encoding_type : PositionalEncodingType, default="sinusoidal"
        Type of positional encoding ("sinusoidal", "learned", "rotary", "alibi", or "none").
    gradient_checkpointing : bool, default=False
        Whether to use gradient checkpointing to save memory.

    Attributes
    ----------
    blocks : nn.ModuleList
        Stack of LST transformer blocks.

    Examples
    --------
    >>> model = LSTTransformer(
    ...     hidden_dim=512,
    ...     num_layers=6,
    ...     transform_type="dct",
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
        transform_type: TransformLSTType = "dct",
        use_conv_bias: bool = True,
        num_classes: int | None = None,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        gradient_checkpointing: bool = False,
    ):
        # Store LST-specific parameters
        self.transform_type = transform_type
        self.use_conv_bias = use_conv_bias
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
        """Build transformer blocks with LST layers.

        Returns
        -------
        nn.ModuleList
            List of LST transformer blocks.
        """
        blocks = []
        for _ in range(self.num_layers):
            # Use appropriate LST attention based on transform type
            attention_layer: DCTAttention | HadamardAttention | LSTAttention
            if self.transform_type == "dct":
                attention_layer = DCTAttention(
                    hidden_dim=self.hidden_dim,
                    num_heads=8,  # Default num_heads
                    learnable_scale=self.use_conv_bias,
                    dropout=self.dropout_rate,
                )
            elif self.transform_type == "hadamard":
                attention_layer = HadamardAttention(
                    hidden_dim=self.hidden_dim,
                    num_heads=8,
                    learnable_scale=self.use_conv_bias,
                    dropout=self.dropout_rate,
                )
            else:  # dst - use general LST attention
                attention_layer = LSTAttention(
                    hidden_dim=self.hidden_dim,
                    num_heads=8,
                    transform_type=self.transform_type,
                    learnable_scale=self.use_conv_bias,
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


    @classmethod
    def from_config(cls, config: "LSTModelConfig") -> "LSTTransformer":  # type: ignore[override]
        """Create model from configuration.

        Parameters
        ----------
        config : LSTModelConfig
            Model configuration object.

        Returns
        -------
        LSTTransformer
            Configured model instance.
        """
        return cls(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            max_sequence_length=config.sequence_length,
            transform_type=config.transform_type,
            use_conv_bias=config.use_conv_bias,
            num_classes=config.num_classes,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
            use_positional_encoding=config.use_positional_encoding,
            positional_encoding_type=config.positional_encoding_type,
            gradient_checkpointing=config.gradient_checkpointing,
        )


@register_component("model", "lst_encoder")
class LSTEncoder(BaseModel):
    """Encoder-only LST model for representation learning.

    This model uses linear spectral transforms without a classification head,
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
    transform_type : TransformLSTType, default="dct"
        Type of spectral transform.
    use_conv_bias : bool, default=True
        Use bias in spectral convolution.
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
        transform_type: TransformLSTType = "dct",
        use_conv_bias: bool = True,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
    ):
        # Store parameters
        self.transform_type = transform_type
        self.use_conv_bias = use_conv_bias
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
        """Build encoder blocks with LST layers.

        Returns
        -------
        nn.ModuleList
            List of LST encoder blocks.
        """
        blocks = []
        for _ in range(self.num_layers):
            # Use appropriate LST attention based on transform type
            attention_layer: DCTAttention | HadamardAttention | LSTAttention
            if self.transform_type == "dct":
                attention_layer = DCTAttention(
                    hidden_dim=self.hidden_dim,
                    num_heads=8,
                    learnable_scale=self.use_conv_bias,
                    dropout=self.dropout_rate,
                )
            elif self.transform_type == "hadamard":
                attention_layer = HadamardAttention(
                    hidden_dim=self.hidden_dim,
                    num_heads=8,
                    learnable_scale=self.use_conv_bias,
                    dropout=self.dropout_rate,
                )
            else:  # dst
                attention_layer = LSTAttention(
                    hidden_dim=self.hidden_dim,
                    num_heads=8,
                    transform_type=self.transform_type,
                    learnable_scale=self.use_conv_bias,
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



@register_component("model", "lst_decoder")
class LSTDecoder(BaseModel):
    """Decoder LST model with optional causal masking.

    This model uses linear spectral transforms with support for causal masking,
    suitable for autoregressive generation tasks.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    hidden_dim : int
        Hidden dimension size.
    num_layers : int
        Number of transformer blocks.
    max_sequence_length : int
        Maximum sequence length.
    transform_type : TransformLSTType, default="dst"
        Type of spectral transform (DST is preferred for causal).
    causal : bool, default=True
        Whether to use causal masking.
    use_conv_bias : bool, default=True
        Use bias in spectral convolution.
    ffn_hidden_dim : int | None, optional
        FFN hidden dimension.
    dropout : float, default=0.0
        Dropout probability.
    use_positional_encoding : bool, default=True
        Use positional encoding.
    positional_encoding_type : str, default="sinusoidal"
        Positional encoding type.
    gradient_checkpointing : bool, default=False
        Use gradient checkpointing.

    Examples
    --------
    >>> decoder = LSTDecoder(
    ...     vocab_size=10000,
    ...     hidden_dim=512,
    ...     num_layers=12,
    ...     transform_type="dst",
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
        transform_type: TransformLSTType = "dst",
        causal: bool = True,
        use_conv_bias: bool = True,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_positional_encoding: bool = True,
        positional_encoding_type: PositionalEncodingType = "sinusoidal",
        gradient_checkpointing: bool = False,
    ):
        # Store decoder-specific parameters
        self.transform_type = transform_type
        self.causal = causal
        self.use_conv_bias = use_conv_bias
        self.dropout_rate = dropout
        self.vocab_size_decoder = vocab_size

        # Initialize with language modeling head
        super().__init__(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            num_classes=None,  # Use LM head instead
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
        """Build decoder blocks with causal LST layers.

        Returns
        -------
        nn.ModuleList
            List of causal LST decoder blocks.
        """
        blocks = []
        for _ in range(self.num_layers):
            # Use appropriate LST attention based on transform type
            # Note: For causal decoder, DST is preferred as it naturally handles causality
            attention_layer: DCTAttention | HadamardAttention | LSTAttention
            if self.transform_type == "dct":
                attention_layer = DCTAttention(
                    hidden_dim=self.hidden_dim,
                    num_heads=8,
                    learnable_scale=self.use_conv_bias,
                    dropout=self.dropout_rate,
                )
            elif self.transform_type == "hadamard":
                attention_layer = HadamardAttention(
                    hidden_dim=self.hidden_dim,
                    num_heads=8,
                    learnable_scale=self.use_conv_bias,
                    dropout=self.dropout_rate,
                )
            else:  # dst - DST is preferred for causal
                attention_layer = LSTAttention(
                    hidden_dim=self.hidden_dim,
                    num_heads=8,
                    transform_type=self.transform_type,
                    learnable_scale=self.use_conv_bias,
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

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the decoder.

        Parameters
        ----------
        input_ids : torch.Tensor | None
            Input token IDs of shape (batch_size, sequence_length).
        inputs_embeds : torch.Tensor | None
            Pre-embedded inputs of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        torch.Tensor
            Language modeling logits of shape (batch_size, sequence_length, vocab_size).
        """
        # Get hidden states from base forward
        hidden_states = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # Apply LM head
        logits = self.lm_head(hidden_states)
        return logits  # type: ignore[no-any-return]

