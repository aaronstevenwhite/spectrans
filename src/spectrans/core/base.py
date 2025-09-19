r"""Base classes and interfaces for spectral transformer components.

This module defines the core abstract base classes and interfaces that all spectral
transformer components must implement. These classes establish consistent APIs for
forward propagation and component composition throughout the spectrans library.

The inheritance hierarchy provides both mathematical rigor and software engineering
best practices, ensuring that all spectral transforms maintain proper interfaces
while allowing for flexible implementation strategies.

Classes
-------
SpectralComponent
    Abstract base class requiring forward() implementation.
SpectralTransform
    Interface for spectral transforms with transform/inverse_transform methods.
MixingLayer
    Base class for token mixing layers with dropout support.
AttentionLayer
    Base class for attention mechanisms with multi-head support.
TransformerBlock
    Complete transformer block with mixing, FFN, and residual connections.
BaseModel
    Full model class with embedding, positional encoding, and classification.

Examples
--------
Implementing a custom spectral component:

>>> import torch.nn as nn
>>> from spectrans.core.base import SpectralComponent
>>> class CustomComponent(SpectralComponent):
...     def forward(self, x):
...         return x * 2  # Simple scaling

Building a transformer block:

>>> import torch.nn as nn
>>> from spectrans.core.base import TransformerBlock
>>> from spectrans.layers.mixing.fourier import FourierMixing
>>> mixing_layer = FourierMixing(hidden_dim=768)
>>> ffn = nn.Sequential(nn.Linear(768, 3072), nn.GELU(), nn.Linear(3072, 768))
>>> block = TransformerBlock(mixing_layer, ffn)

Notes
-----
The base classes implement several key design patterns:

1. **Template Method Pattern**: TransformerBlock defines the structure while allowing
   flexible mixing layer implementations
2. **Strategy Pattern**: Different spectral transforms can be swapped via the same interface
3. **Composition over Inheritance**: Complex behaviors built by composing simple components

Mathematical Properties:
- All spectral components preserve tensor shapes in the sequence dimension
- Residual connections maintain gradient flow and training stability
- Dropout is applied consistently after each sub-layer
- Layer normalization follows the pre-norm architecture pattern

The TransformerBlock implements the standard architecture:

$$
\begin{align}
\mathbf{H}_l &= \text{LayerNorm}(\mathbf{X}_l + \text{MixingLayer}(\mathbf{X}_l)) \\
\mathbf{X}_{l+1} &= \text{LayerNorm}(\mathbf{H}_l + \text{FFN}(\mathbf{H}_l))
\end{align}
$$

See Also
--------
spectrans.core.types : Type definitions used in base classes
spectrans.transforms.base : Transform-specific base classes
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..layers.mixing.base import MixingLayer


class SpectralComponent(nn.Module, ABC):
    """Base class for all spectral components.

    This abstract base class defines the interface that all spectral
    transformer components must implement.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]:  # type: ignore[no-untyped-def]
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).
        *args : Any
            Additional positional arguments for subclass-specific parameters.
        **kwargs : Any
            Additional keyword arguments for subclass-specific parameters.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, ...]
            Output tensor(s). Single tensor for most cases, tuple for attention
            layers that optionally return attention weights.
        """
        pass


class AttentionLayer(SpectralComponent):
    """Base class for attention layers.

    Attention layers implement various forms of spectral attention
    mechanisms as alternatives to standard multi-head attention.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the model.
    num_heads : int, default=1
        Number of attention heads.
    dropout : float, default=0.0
        Dropout probability.

    Attributes
    ----------
    hidden_dim : int
        Hidden dimension of the model.
    num_heads : int
        Number of attention heads.
    dropout : nn.Module
        Dropout layer or identity if dropout is 0.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


class TransformerBlock(SpectralComponent):
    """Base class for transformer blocks.

    Transformer blocks combine mixing/attention layers with feedforward
    networks and normalization to form complete transformer layers.

    Parameters
    ----------
    mixing_layer : MixingLayer | AttentionLayer
        The mixing or attention layer for token interactions.
    ffn : nn.Module | None, default=None
        Feedforward network module. If None, no FFN is used.
    norm_layer : type[nn.Module], default=nn.LayerNorm
        Normalization layer class to use.
    dropout : float, default=0.0
        Dropout probability for residual connections.

    Attributes
    ----------
    mixing_layer : MixingLayer | AttentionLayer
        The mixing or attention layer.
    ffn : nn.Module | None
        Feedforward network module.
    hidden_dim : int
        Hidden dimension extracted from mixing layer.
    norm1 : nn.Module
        First normalization layer.
    norm2 : nn.Module | None
        Second normalization layer (if FFN is used).
    dropout : nn.Module
        Dropout layer for residual connections.
    """

    def __init__(
        self,
        mixing_layer: "MixingLayer | AttentionLayer",
        ffn: nn.Module | None = None,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mixing_layer = mixing_layer
        self.ffn = ffn

        # Get hidden dimension from mixing layer
        self.hidden_dim = mixing_layer.hidden_dim

        # Setup normalization layers
        self.norm1 = norm_layer(self.hidden_dim)
        self.norm2 = norm_layer(self.hidden_dim) if ffn is not None else None

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of same shape as input.
        """
        # Mixing/attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.mixing_layer(x)
        x = self.dropout(x)
        x = residual + x

        # FFN with residual connection (if FFN exists)
        if self.ffn is not None and self.norm2 is not None:
            residual = x
            x = self.norm2(x)
            x = self.ffn(x)
            x = self.dropout(x)
            x = residual + x

        return x


class BaseModel(nn.Module):
    """Base class for spectral transformer models.

    This class provides common functionality for all spectral
    transformer model variants.

    Parameters
    ----------
    num_layers : int
        Number of transformer layers.
    hidden_dim : int
        Hidden dimension of the model.
    max_seq_length : int, default=512
        Maximum sequence length.
    vocab_size : int | None, default=None
        Vocabulary size for embedding layer. If None, no embedding is used.
    num_classes : int | None, default=None
        Number of output classes. If None, no classification head is used.
    dropout : float, default=0.0
        Dropout probability.

    Attributes
    ----------
    num_layers : int
        Number of transformer layers.
    hidden_dim : int
        Hidden dimension of the model.
    max_seq_length : int
        Maximum sequence length.
    vocab_size : int | None
        Vocabulary size.
    num_classes : int | None
        Number of output classes.
    embedding : nn.Embedding | None
        Optional embedding layer.
    pos_embedding : nn.Parameter
        Positional embedding parameters.
    dropout : nn.Module
        Dropout layer.
    classifier : nn.Linear | None
        Optional classification head.
    blocks : nn.ModuleList
        List of transformer blocks (populated by subclasses).
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        max_seq_length: int = 512,
        vocab_size: int | None = None,
        num_classes: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        # Optional embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim) if vocab_size else None

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, hidden_dim))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Optional classification head
        self.classifier = nn.Linear(hidden_dim, num_classes) if num_classes else None

        # Transformer blocks will be defined in subclasses
        self.blocks = nn.ModuleList()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Shape depends on whether embedding is used:
            - With embedding: (batch_size, sequence_length) containing token indices
            - Without embedding: (batch_size, sequence_length, hidden_dim)
        mask : torch.Tensor | None, default=None
            Optional attention mask of shape (batch_size, sequence_length).

        Returns
        -------
        torch.Tensor
            Output tensor. Shape depends on whether classifier is used:
            - With classifier: (batch_size, num_classes)
            - Without classifier: (batch_size, sequence_length, hidden_dim)
        """
        # Apply embedding if needed
        if self.embedding is not None:
            x = self.embedding(x)

        # Add positional embeddings
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]

        # Apply dropout
        x = self.dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply classification head if needed
        if self.classifier is not None:
            # Use [CLS] token or mean pooling
            x = x.mean(dim=1)  # Mean pooling
            x = self.classifier(x)

        return x
