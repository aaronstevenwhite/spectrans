"""Base model classes for spectral transformers.

This module provides the base model classes and common functionality for building
complete spectral transformer models. The base classes handle common tasks like
embeddings, positional encoding, and output projections, while allowing specific
models to customize the core transformer blocks and mixing layers.

Classes
-------
BaseModel
    Abstract base class for all spectral transformer models.
PositionalEncoding
    Sinusoidal positional encoding following the original Transformer paper.
LearnedPositionalEncoding
    Learnable positional embeddings as an alternative to sinusoidal encoding.
ClassificationHead
    Output head for classification tasks.
RegressionHead
    Output head for regression tasks.
SequenceHead
    Output head for sequence-to-sequence tasks.

Examples
--------
Creating a custom model by extending BaseModel:

>>> from spectrans.models.base import BaseModel
>>> from spectrans.layers.mixing.fourier import FourierMixing
>>> from spectrans.blocks.base import PreNormBlock
>>> class MyModel(BaseModel):
...     def build_blocks(self):
...         return nn.ModuleList([
...             PreNormBlock(
...                 mixing_layer=FourierMixing(self.hidden_dim),
...                 hidden_dim=self.hidden_dim,
...                 ffn_hidden_dim=self.ffn_hidden_dim,
...                 dropout=self.dropout
...             )
...             for _ in range(self.num_layers)
...         ])

Using positional encoding:

>>> from spectrans.models.base import PositionalEncoding
>>> pos_encoder = PositionalEncoding(hidden_dim=768, max_sequence_length=1024)
>>> embeddings = torch.randn(32, 512, 768)
>>> encoded = pos_encoder(embeddings)

Notes
-----
The base model architecture follows the standard transformer pattern:
1. Input embedding (optional)
2. Positional encoding (optional)
3. Stack of transformer blocks
4. Output projection/head (task-specific)

All models support gradient checkpointing for memory-efficient training
and can be easily configured through Pydantic configuration objects.

References
----------
.. [1] Vaswani et al., "Attention is All You Need", NeurIPS 2017.
"""

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils import checkpoint

from ..core.base import SpectralComponent
from ..core.types import OutputHeadType, PoolingType, PositionalEncodingType, Tensor

if TYPE_CHECKING:
    from ..config.models import ModelConfig


class BaseModel(SpectralComponent, ABC):
    """Abstract base class for spectral transformer models.

    This class provides the common functionality shared by all spectral
    transformer models, including embeddings, positional encoding, and
    output heads. Subclasses must implement the build_blocks method to
    define their specific architecture.

    Parameters
    ----------
    vocab_size : int | None, optional
        Size of the vocabulary for token embeddings. If None, no input
        embedding layer is created (assumes pre-embedded inputs).
    hidden_dim : int
        Hidden dimension size for the model.
    num_layers : int
        Number of transformer blocks in the model.
    max_sequence_length : int
        Maximum sequence length the model can process.
    num_classes : int | None, optional
        Number of output classes for classification. If None, no classification
        head is added.
    use_positional_encoding : bool, optional
        Whether to use positional encoding. Default is True.
    positional_encoding_type : PositionalEncodingType, optional
        Type of positional encoding: 'sinusoidal', 'learned', 'rotary', 'alibi', or 'none'.
        Default is 'sinusoidal'.
    dropout : float, optional
        Dropout probability. Default is 0.1.
    ffn_hidden_dim : int | None, optional
        Hidden dimension for feedforward networks. If None, defaults to 4 * hidden_dim.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.
    output_type : OutputHeadType, optional
        Type of output head: 'classification', 'regression', 'sequence', 'lm', or 'none'.
        Default is 'classification'.
    gradient_checkpointing : bool, optional
        Whether to use gradient checkpointing for memory efficiency. Default is False.

    Attributes
    ----------
    hidden_dim : int
        Hidden dimension size.
    num_layers : int
        Number of transformer blocks.
    max_sequence_length : int
        Maximum sequence length.
    embedding : nn.Embedding | None
        Token embedding layer (if vocab_size is provided).
    positional_encoding : PositionalEncoding | LearnedPositionalEncoding | None
        Positional encoding module.
    blocks : nn.ModuleList
        List of transformer blocks.
    output_head : nn.Module | None
        Task-specific output head.
    dropout : nn.Dropout
        Dropout layer.
    """

    embedding: nn.Embedding | None
    positional_encoding: nn.Module | None
    output_head: nn.Module | None
    dropout: nn.Dropout

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
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        self.ffn_hidden_dim = ffn_hidden_dim or (4 * hidden_dim)
        self.norm_eps = norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.output_type = output_type
        self.num_classes = num_classes

        # Input embedding
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self._initialize_embeddings()
        else:
            self.embedding = None

        # Positional encoding
        self.positional_encoding_type = positional_encoding_type
        if use_positional_encoding and positional_encoding_type != "none":
            if positional_encoding_type == "sinusoidal":
                self.positional_encoding = PositionalEncoding(
                    hidden_dim=hidden_dim,
                    max_sequence_length=max_sequence_length,
                    dropout=dropout,
                )
            elif positional_encoding_type == "learned":
                self.positional_encoding = LearnedPositionalEncoding(
                    hidden_dim=hidden_dim,
                    max_sequence_length=max_sequence_length,
                    dropout=dropout,
                )
            elif positional_encoding_type == "rotary":
                self.positional_encoding = RotaryPositionalEncoding(
                    hidden_dim=hidden_dim,
                    max_sequence_length=max_sequence_length,
                )
            elif positional_encoding_type == "alibi":
                # ALiBi is handled differently - it's added to attention scores
                # We'll need to pass num_heads, which we'll get from the first block
                self.positional_encoding = None  # Will be initialized after blocks
            else:
                raise ValueError(f"Unknown positional encoding type: {positional_encoding_type}")
        else:
            self.positional_encoding = None

        # Build transformer blocks (implemented by subclasses)
        self.blocks = self.build_blocks()

        # Layer norm before output
        self.final_norm = nn.LayerNorm(hidden_dim, eps=norm_eps)

        # Output head
        if output_type == "classification" and num_classes is not None:
            self.output_head = ClassificationHead(hidden_dim, num_classes, dropout)
        elif output_type == "regression":
            self.output_head = RegressionHead(hidden_dim, dropout)
        elif output_type == "sequence":
            vocab_size_out = num_classes if num_classes is not None else (vocab_size if vocab_size is not None else hidden_dim)
            self.output_head = SequenceHead(hidden_dim, vocab_size_out, dropout)
        elif output_type == "lm":
            # Language modeling head outputs vocab_size
            if num_classes is not None:
                vocab_size_out = num_classes
            elif vocab_size is not None:
                vocab_size_out = vocab_size
            else:
                raise ValueError("Either num_classes or vocab_size must be specified for language modeling output")
            self.output_head = SequenceHead(hidden_dim, vocab_size_out, dropout)
        else:
            self.output_head = None

        # Dropout
        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def build_blocks(self) -> nn.ModuleList:
        """Build the transformer blocks for the model.

        This method must be implemented by subclasses to define
        the specific architecture using appropriate mixing layers.

        Returns
        -------
        nn.ModuleList
            List of transformer blocks.
        """
        pass

    def _initialize_embeddings(self) -> None:
        """Initialize embedding weights using Xavier uniform initialization."""
        if self.embedding is not None:
            nn.init.xavier_uniform_(self.embedding.weight)

    def forward(
        self,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        input_ids : Tensor | None, optional
            Input token IDs of shape (batch_size, sequence_length).
            Required if embedding layer exists.
        inputs_embeds : Tensor | None, optional
            Pre-embedded inputs of shape (batch_size, sequence_length, hidden_dim).
            Used if no embedding layer or to bypass embedding.
        attention_mask : Tensor | None, optional
            Attention mask of shape (batch_size, sequence_length).
            Values should be 0 or 1 (1 for tokens to attend to).

        Returns
        -------
        Tensor
            Output tensor. Shape depends on the output head:
            - Classification: (batch_size, num_classes)
            - Regression: (batch_size, 1)
            - Sequence: (batch_size, sequence_length, vocab_size)
            - None: (batch_size, sequence_length, hidden_dim)

        Raises
        ------
        ValueError
            If neither input_ids nor inputs_embeds is provided.
        """
        # Get embeddings
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif input_ids is not None and self.embedding is not None:
            hidden_states = self.embedding(input_ids)
        elif input_ids is not None:
            raise ValueError("Model has no embedding layer but input_ids was provided")
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Add positional encoding
        if self.positional_encoding is not None:
            hidden_states = self.positional_encoding(hidden_states)

        # Apply dropout
        hidden_states = self.dropout(hidden_states)

        # Process through transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint.checkpoint(
                    block, hidden_states, use_reentrant=False
                )
            else:
                hidden_states = block(hidden_states)

        # Final layer norm
        hidden_states = self.final_norm(hidden_states)

        # Apply output head if present
        if self.output_head is not None:
            output: Tensor = self.output_head(hidden_states, attention_mask)
        else:
            output = hidden_states

        return output

    @property
    def complexity(self) -> dict[str, str]:
        """Computational complexity of the model.

        Returns
        -------
        dict[str, str]
            Dictionary with 'time' and 'space' complexity.
        """
        n = self.max_sequence_length
        d = self.hidden_dim
        L = self.num_layers

        # Get complexity from one block (assumes all blocks have same complexity)
        if self.blocks and hasattr(self.blocks[0], "complexity"):
            block_complexity = self.blocks[0].complexity
            # Multiply by number of layers
            if isinstance(block_complexity, dict):
                time = f"O({L} * {block_complexity.get('time', 'n * d')[2:]})"
                space = f"O({L} * {block_complexity.get('space', 'n * d')[2:]})"
            else:
                time = f"O({L} * {n} * {d}^2)"
                space = f"O({L} * {n} * {d})"
        else:
            # Default complexity for transformer models
            time = f"O({L} * {n} * {d}^2)"
            space = f"O({L} * {n} * {d})"

        return {"time": time, "space": space}

    @classmethod
    def from_config(cls, config: "ModelConfig") -> "BaseModel":
        """Create model instance from configuration object.

        Parameters
        ----------
        config : ModelConfig
            Configuration object with model parameters.

        Returns
        -------
        BaseModel
            Configured model instance.
        """
        # Build model directly from config attributes
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
            gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
        )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    This module adds sinusoidal positional encodings to embeddings,
    following the approach in "Attention is All You Need".

    Parameters
    ----------
    hidden_dim : int
        Dimension of the embeddings.
    max_sequence_length : int
        Maximum sequence length to encode.
    dropout : float, optional
        Dropout probability. Default is 0.1.

    Attributes
    ----------
    dropout : nn.Dropout
        Dropout layer.
    pe : Tensor
        Precomputed positional encodings.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_sequence_length: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe: Tensor  # Type annotation for buffer

        # Create positional encodings
        pe = torch.zeros(max_sequence_length, hidden_dim)
        position = torch.arange(0, max_sequence_length).unsqueeze(1).float()

        # Create div_term for the sinusoidal pattern
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float()
            * -(math.log(10000.0) / hidden_dim)
        )

        # Apply sinusoidal functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        Tensor
            Tensor with positional encoding added.
        """
        # Add positional encoding
        seq_length = x.size(1)
        x = x + self.pe[:, :seq_length, :]
        result: Tensor = self.dropout(x)
        return result


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings.

    This module uses learnable positional embeddings instead of
    fixed sinusoidal encodings.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the embeddings.
    max_sequence_length : int
        Maximum sequence length to encode.
    dropout : float, optional
        Dropout probability. Default is 0.1.

    Attributes
    ----------
    position_embeddings : nn.Embedding
        Learnable position embeddings.
    dropout : nn.Dropout
        Dropout layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_sequence_length: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_sequence_length, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.position_embeddings.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Add learned positional embeddings to input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        Tensor
            Tensor with positional embeddings added.
        """
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)

        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        result: Tensor = self.dropout(x)
        return result


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE).

    This module implements Rotary Position Embeddings as described in the RoFormer paper.
    RoPE encodes absolute position with rotation matrix and naturally incorporates
    relative position dependency in self-attention formulation.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the embeddings. Must be even.
    max_sequence_length : int
        Maximum sequence length to encode.
    base : float, optional
        Base for the frequency calculation. Default is 10000.

    Attributes
    ----------
    inv_freq : Tensor
        Inverse frequencies for computing rotary embeddings.
    cos_cached : Tensor | None
        Cached cosine values for positions.
    sin_cached : Tensor | None
        Cached sine values for positions.

    References
    ----------
    .. [1] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding",
           arXiv:2104.09864, 2021.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_sequence_length: int = 5000,
        base: float = 10000.0,
    ):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even for RoPE, got {hidden_dim}")

        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for precomputed cos/sin
        self.cos_cached: Tensor | None = None
        self.sin_cached: Tensor | None = None
        self._build_cache(max_sequence_length)

    def _build_cache(self, seq_len: int) -> None:
        """Build cache for cos/sin values up to seq_len."""
        if self.cos_cached is not None and self.cos_cached.size(0) >= seq_len:
            return

        # Create position indices
        inv_freq = self.inv_freq
        assert isinstance(inv_freq, torch.Tensor)
        t = torch.arange(seq_len).type_as(inv_freq)

        # Compute frequencies for each position
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # Duplicate frequencies for cos/sin pairs
        emb = torch.cat((freqs, freqs), dim=-1)

        # Cache cos/sin values
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """Apply rotary position embedding to input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, num_heads, sequence_length, head_dim)
            or (batch_size, sequence_length, hidden_dim).
        offset : int, optional
            Position offset for incremental decoding. Default is 0.

        Returns
        -------
        Tensor
            Tensor with rotary position embeddings applied.
        """
        # Handle both 3D and 4D inputs
        if x.ndim == 3:
            batch_size, seq_len, hidden_dim = x.shape
            # For simplicity, we'll apply RoPE directly to the hidden dimension
            # In practice, this would be applied separately to Q and K in attention
            was_3d = True
        else:
            x.shape[0]
            seq_len = x.shape[2] if x.ndim == 4 else x.shape[1]
            was_3d = False

        # Rebuild cache if needed
        self._build_cache(seq_len + offset)

        if was_3d:
            # For 3D tensor, apply rotation directly
            assert self.cos_cached is not None
            assert self.sin_cached is not None
            cos = self.cos_cached[:, :, offset:offset + seq_len, :].squeeze(1)
            sin = self.sin_cached[:, :, offset:offset + seq_len, :].squeeze(1)

            # Split x into two halves for rotation
            x1, x2 = x.chunk(2, dim=-1)

            # Apply rotation
            rotated = torch.cat(
                [x1 * cos[:, :, :x1.shape[-1]] - x2 * sin[:, :, :x2.shape[-1]],
                 x1 * sin[:, :, :x1.shape[-1]] + x2 * cos[:, :, :x2.shape[-1]]],
                dim=-1
            )
        else:
            # For 4D tensor (batch, heads, seq, head_dim)
            assert self.cos_cached is not None
            assert self.sin_cached is not None
            cos = self.cos_cached[:, :, offset:offset + seq_len, :]
            sin = self.sin_cached[:, :, offset:offset + seq_len, :]

            # Split x into two halves for rotation
            x1, x2 = x.chunk(2, dim=-1)

            # Apply rotation
            rotated = torch.cat(
                [x1 * cos[:, :, :, :x1.shape[-1]] - x2 * sin[:, :, :, :x2.shape[-1]],
                 x1 * sin[:, :, :, :x1.shape[-1]] + x2 * cos[:, :, :, :x2.shape[-1]]],
                dim=-1
            )

        return rotated


class ALiBiPositionalBias(nn.Module):
    """Attention with Linear Biases (ALiBi) positional encoding.

    This module implements ALiBi, which adds a linear bias to attention scores
    based on the relative distance between tokens. Unlike traditional position
    embeddings, ALiBi enables extrapolation to longer sequences.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    max_sequence_length : int
        Maximum sequence length to encode.

    Attributes
    ----------
    num_heads : int
        Number of attention heads.
    slopes : Tensor
        Head-specific slope parameters.
    alibi : Tensor | None
        Cached linear bias matrix.

    References
    ----------
    .. [1] Press et al., "Train Short, Test Long: Attention with Linear Biases
           Enables Input Length Extrapolation", ICLR 2022.
    """

    def __init__(
        self,
        num_heads: int,
        max_sequence_length: int = 5000,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length

        # Compute slopes for each head
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes)

        # Cache for bias matrix
        self.alibi: Tensor | None = None
        self._build_alibi_tensor(max_sequence_length)

    def _get_slopes(self, num_heads: int) -> Tensor:
        """Compute slope parameters for each attention head.

        Following the paper, slopes are geometric sequence of ratios
        starting from 2^(-8/num_heads) for better extrapolation.
        """
        def get_slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-8 / n)
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = torch.tensor(get_slopes_power_of_2(num_heads))
        else:
            # If num_heads is not a power of 2, interpolate
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes_power_of_2 = get_slopes_power_of_2(closest_power_of_2)

            # Interpolate to get the remaining slopes
            extra_slopes = []
            for i in range(num_heads - closest_power_of_2):
                extra_slopes.append(
                    slopes_power_of_2[i % closest_power_of_2] * 0.5
                )

            slopes = torch.tensor(slopes_power_of_2 + extra_slopes)

        return slopes.view(1, num_heads, 1, 1)

    def _build_alibi_tensor(self, seq_len: int) -> None:
        """Build ALiBi bias tensor for given sequence length."""
        if self.alibi is not None and self.alibi.size(-1) >= seq_len:
            return

        # Create relative position matrix
        positions = torch.arange(seq_len)[None, :]
        distances = positions - positions.transpose(0, 1)

        # Apply slopes to get biases for each head
        slopes = self.slopes
        assert isinstance(slopes, torch.Tensor)
        alibi = distances[None, None, :, :] * slopes
        self.alibi = alibi

    def forward(self, attention_scores: Tensor) -> Tensor:
        """Add ALiBi bias to attention scores.

        Parameters
        ----------
        attention_scores : Tensor
            Attention scores of shape (batch_size, num_heads, seq_len, seq_len).

        Returns
        -------
        Tensor
            Attention scores with ALiBi bias added.
        """
        batch_size, num_heads, seq_len, _ = attention_scores.shape

        # Rebuild cache if needed
        self._build_alibi_tensor(seq_len)

        # Add ALiBi bias
        assert self.alibi is not None
        alibi_bias = self.alibi[:, :, :seq_len, :seq_len].to(attention_scores.device)
        return attention_scores + alibi_bias

    def get_bias(self, seq_len: int, device: torch.device | None = None) -> Tensor:
        """Get ALiBi bias matrix for a given sequence length.

        Parameters
        ----------
        seq_len : int
            Sequence length.
        device : torch.device | None, optional
            Device to place the bias tensor.

        Returns
        -------
        Tensor
            ALiBi bias of shape (1, num_heads, seq_len, seq_len).
        """
        self._build_alibi_tensor(seq_len)
        assert self.alibi is not None
        bias = self.alibi[:, :, :seq_len, :seq_len]
        if device is not None:
            bias = bias.to(device)
        return bias


class ClassificationHead(nn.Module):
    """Classification output head.

    This module pools sequence outputs and projects to class logits.

    Parameters
    ----------
    hidden_dim : int
        Input hidden dimension.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Dropout probability. Default is 0.1.
    pooling : PoolingType, optional
        Pooling strategy: 'cls', 'mean', or 'max'. Default is 'cls'.

    Attributes
    ----------
    pooling : PoolingType
        Pooling strategy.
    dropout : nn.Dropout
        Dropout layer.
    classifier : nn.Linear
        Output projection layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        pooling: PoolingType = "cls",
    ):
        super().__init__()
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through classification head.

        Parameters
        ----------
        hidden_states : Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).
        attention_mask : Tensor | None, optional
            Attention mask for pooling operations.

        Returns
        -------
        Tensor
            Classification logits of shape (batch_size, num_classes).
        """
        # Pool the sequence
        if self.pooling == "cls":
            # Use first token (CLS token)
            pooled = hidden_states[:, 0, :]
        elif self.pooling == "mean":
            # Mean pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = hidden_states.mean(dim=1)
        elif self.pooling == "max":
            # Max pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            pooled = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Apply dropout and classifier
        pooled = self.dropout(pooled)
        logits: Tensor = self.classifier(pooled)
        return logits


class RegressionHead(nn.Module):
    """Regression output head.

    This module pools sequence outputs and projects to a scalar value.

    Parameters
    ----------
    hidden_dim : int
        Input hidden dimension.
    dropout : float, optional
        Dropout probability. Default is 0.1.
    pooling : PoolingType, optional
        Pooling strategy: 'cls', 'mean', or 'max'. Default is 'mean'.

    Attributes
    ----------
    pooling : PoolingType
        Pooling strategy.
    dropout : nn.Dropout
        Dropout layer.
    regressor : nn.Linear
        Output projection layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        pooling: PoolingType = "mean",
    ):
        super().__init__()
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through regression head.

        Parameters
        ----------
        hidden_states : Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).
        attention_mask : Tensor | None, optional
            Attention mask for pooling operations.

        Returns
        -------
        Tensor
            Regression output of shape (batch_size, 1).
        """
        # Pool the sequence (same logic as classification)
        if self.pooling == "cls":
            pooled = hidden_states[:, 0, :]
        elif self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = hidden_states.mean(dim=1)
        elif self.pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            pooled = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Apply dropout and regressor
        pooled = self.dropout(pooled)
        output: Tensor = self.regressor(pooled)
        return output


class SequenceHead(nn.Module):
    """Sequence-to-sequence output head.

    This module projects hidden states to vocabulary logits for
    sequence generation tasks.

    Parameters
    ----------
    hidden_dim : int
        Input hidden dimension.
    vocab_size : int
        Output vocabulary size.
    dropout : float, optional
        Dropout probability. Default is 0.1.

    Attributes
    ----------
    dropout : nn.Dropout
        Dropout layer.
    lm_head : nn.Linear
        Language modeling head.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,  # noqa: ARG002
    ) -> Tensor:
        """Forward pass through sequence head.

        Parameters
        ----------
        hidden_states : Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).
        attention_mask : Tensor | None, optional
            Not used, kept for interface consistency.

        Returns
        -------
        Tensor
            Vocabulary logits of shape (batch_size, sequence_length, vocab_size).
        """
        hidden_states = self.dropout(hidden_states)
        logits: Tensor = self.lm_head(hidden_states)
        return logits
