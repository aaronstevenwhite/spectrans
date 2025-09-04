"""Base classes and interfaces for transformer blocks.

This module provides the base classes and interfaces for building transformer
blocks in the spectrans library. Transformer blocks are composed of mixing/attention
layers followed by feedforward networks, with residual connections and normalization.

Classes
-------
TransformerBlock
    Base class for all transformer blocks.
FeedForwardNetwork
    Standard feedforward network with configurable activation.
PreNormBlock
    Transformer block with pre-layer normalization.
PostNormBlock
    Transformer block with post-layer normalization.
ParallelBlock
    Transformer block with parallel mixing and FFN branches.

Examples
--------
Creating a custom transformer block:

>>> from spectrans.blocks.base import TransformerBlock
>>> from spectrans.layers.mixing.fourier import FourierMixing
>>> block = TransformerBlock(
...     mixing_layer=FourierMixing(hidden_dim=768),
...     hidden_dim=768,
...     use_pre_norm=True
... )

Notes
-----
The transformer block architecture follows the standard pattern:
- Mixing/Attention layer with residual connection
- Feedforward network with residual connection
- Layer normalization (pre-norm or post-norm)
- Optional dropout for regularization

References
----------
.. [1] Vaswani et al., "Attention is All You Need", NeurIPS 2017.
.. [2] Xiong et al., "On Layer Normalization in the Transformer Architecture", ICML 2020.
"""


import torch
import torch.nn as nn

from ..core.base import SpectralComponent
from ..core.registry import register_component
from ..layers.mixing.base import MixingLayer


class TransformerBlock(SpectralComponent):
    """Base class for transformer blocks.

    A transformer block combines a mixing/attention layer with a feedforward
    network, using residual connections and layer normalization.

    Parameters
    ----------
    mixing_layer : MixingLayer | nn.Module
        The mixing or attention layer for token interaction.
    hidden_dim : int
        Hidden dimension of the model.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the feedforward network. Default is 4 * hidden_dim.
    activation : str, optional
        Activation function for the FFN. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    use_pre_norm : bool, optional
        Whether to use pre-layer normalization. Default is True.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.

    Attributes
    ----------
    mixing_layer : MixingLayer | nn.Module
        The mixing or attention layer.
    ffn : FeedForwardNetwork | None
        The feedforward network.
    norm1 : nn.LayerNorm
        First layer normalization.
    norm2 : nn.LayerNorm | None
        Second layer normalization (if FFN is used).
    dropout : nn.Dropout
        Dropout layer.
    use_pre_norm : bool
        Whether pre-normalization is used.
    """

    # Explicit type annotations for attributes that can be None
    ffn: "FeedForwardNetwork | None"
    norm2: nn.LayerNorm | None

    def __init__(
        self,
        mixing_layer: MixingLayer | nn.Module,
        hidden_dim: int,
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_pre_norm: bool = True,
        norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mixing_layer = mixing_layer
        self.use_pre_norm = use_pre_norm

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim, eps=norm_eps)

        # Feedforward network
        if ffn_hidden_dim is not None:
            self.ffn = FeedForwardNetwork(
                hidden_dim=hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                activation=activation,
                dropout=dropout,
            )
            self.norm2 = nn.LayerNorm(hidden_dim, eps=norm_eps)
        else:
            self.ffn = None
            self.norm2 = None

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, hidden_dim).
        """
        output: torch.Tensor
        if self.use_pre_norm:
            # Pre-norm: normalize before mixing
            h = x + self.dropout(self.mixing_layer(self.norm1(x)))
            if self.ffn is not None and self.norm2 is not None:
                output = h + self.dropout(self.ffn(self.norm2(h)))
            else:
                output = h
        else:
            # Post-norm: normalize after mixing
            h = self.norm1(x + self.dropout(self.mixing_layer(x)))
            if self.ffn is not None and self.norm2 is not None:
                output = self.norm2(h + self.dropout(self.ffn(h)))
            else:
                output = h

        return output

    @property
    def complexity(self) -> dict[str, str]:
        """Computational complexity of the transformer block.

        Returns
        -------
        dict[str, str]
            Dictionary with 'time' and 'space' complexity.
        """
        n, d = "n", "d"  # sequence length and hidden dimension

        # Get mixing layer complexity
        if hasattr(self.mixing_layer, "complexity"):
            layer_complexity = self.mixing_layer.complexity
            # Ensure we have a proper dict with string keys
            if isinstance(layer_complexity, dict):
                mixing_complexity = layer_complexity
            else:
                mixing_complexity = {"time": f"O({n}*{d})", "space": f"O({n}*{d})"}
        else:
            mixing_complexity = {"time": f"O({n}*{d})", "space": f"O({n}*{d})"}

        # FFN complexity
        if self.ffn is not None:
            ffn_complexity = self.ffn.complexity
            # Combine complexities (take max)
            time = f"max({mixing_complexity['time']}, {ffn_complexity['time']})"
            space = f"max({mixing_complexity['space']}, {ffn_complexity['space']})"
        else:
            time = mixing_complexity["time"]
            space = mixing_complexity["space"]

        return {"time": time, "space": space}


class FeedForwardNetwork(nn.Module):
    """Standard feedforward network for transformer blocks.

    A two-layer MLP with configurable activation function and dropout.

    Parameters
    ----------
    hidden_dim : int
        Input and output dimension.
    ffn_hidden_dim : int
        Hidden dimension of the FFN.
    activation : str, optional
        Activation function name. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.

    Attributes
    ----------
    fc1 : nn.Linear
        First linear layer.
    fc2 : nn.Linear
        Second linear layer.
    activation : nn.Module
        Activation function.
    dropout : nn.Dropout
        Dropout layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_hidden_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        # Linear layers
        self.fc1 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, hidden_dim)

        # Activation function
        activation_functions = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        if activation not in activation_functions:
            raise ValueError(f"Unknown activation: {activation}")
        self.activation = activation_functions[activation]

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FFN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., hidden_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., hidden_dim).
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    @property
    def complexity(self) -> dict[str, str]:
        """Computational complexity of the FFN.

        Returns
        -------
        dict[str, str]
            Dictionary with 'time' and 'space' complexity.
        """
        n, d, h = "n", self.hidden_dim, self.ffn_hidden_dim
        return {
            "time": f"O({n} * {d} * {h})",
            "space": f"O({d} * {h})",
        }


@register_component("block", "pre_norm")
class PreNormBlock(TransformerBlock):
    """Transformer block with pre-layer normalization.

    This block applies layer normalization before the mixing layer and FFN,
    which has been shown to improve training stability.

    Parameters
    ----------
    mixing_layer : MixingLayer | nn.Module
        The mixing or attention layer.
    hidden_dim : int
        Hidden dimension of the model.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the FFN. Default is 4 * hidden_dim.
    activation : str, optional
        Activation function. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.
    """

    def __init__(
        self,
        mixing_layer: MixingLayer | nn.Module,
        hidden_dim: int,
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        if ffn_hidden_dim is None:
            ffn_hidden_dim = 4 * hidden_dim
        super().__init__(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=activation,
            dropout=dropout,
            use_pre_norm=True,
            norm_eps=norm_eps,
        )


@register_component("block", "post_norm")
class PostNormBlock(TransformerBlock):
    """Transformer block with post-layer normalization.

    This block applies layer normalization after the mixing layer and FFN,
    following the original transformer architecture.

    Parameters
    ----------
    mixing_layer : MixingLayer | nn.Module
        The mixing or attention layer.
    hidden_dim : int
        Hidden dimension of the model.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the FFN. Default is 4 * hidden_dim.
    activation : str, optional
        Activation function. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.
    """

    def __init__(
        self,
        mixing_layer: MixingLayer | nn.Module,
        hidden_dim: int,
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        if ffn_hidden_dim is None:
            ffn_hidden_dim = 4 * hidden_dim
        super().__init__(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=activation,
            dropout=dropout,
            use_pre_norm=False,
            norm_eps=norm_eps,
        )


@register_component("block", "parallel")
class ParallelBlock(SpectralComponent):
    """Transformer block with parallel mixing and FFN branches.

    This block processes the mixing layer and FFN in parallel rather than
    sequentially, which can improve efficiency and has been shown to work
    well in practice.

    Parameters
    ----------
    mixing_layer : MixingLayer | nn.Module
        The mixing or attention layer.
    hidden_dim : int
        Hidden dimension of the model.
    ffn_hidden_dim : int | None, optional
        Hidden dimension of the FFN. Default is 4 * hidden_dim.
    activation : str, optional
        Activation function. Default is 'gelu'.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    norm_eps : float, optional
        Epsilon for layer normalization. Default is 1e-12.

    Attributes
    ----------
    mixing_layer : MixingLayer | nn.Module
        The mixing or attention layer.
    ffn : FeedForwardNetwork
        The feedforward network.
    norm : nn.LayerNorm
        Layer normalization.
    dropout : nn.Dropout
        Dropout layer.
    """

    def __init__(
        self,
        mixing_layer: MixingLayer | nn.Module,
        hidden_dim: int,
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mixing_layer = mixing_layer

        # Default FFN dimension
        if ffn_hidden_dim is None:
            ffn_hidden_dim = 4 * hidden_dim

        # Components
        self.ffn = FeedForwardNetwork(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=activation,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the parallel block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, hidden_dim).
        """
        # Normalize input
        normed = self.norm(x)

        # Process mixing and FFN in parallel
        mixed = self.mixing_layer(normed)
        ffn_out = self.ffn(normed)

        # Combine and add residual
        output: torch.Tensor = x + self.dropout(mixed + ffn_out)

        return output

    @property
    def complexity(self) -> dict[str, str]:
        """Computational complexity of the parallel block.

        Returns
        -------
        dict[str, str]
            Dictionary with 'time' and 'space' complexity.
        """
        # Similar to sequential but potentially more efficient
        if hasattr(self.mixing_layer, "complexity"):
            layer_complexity = self.mixing_layer.complexity
            # Ensure we have a proper dict with string keys
            if isinstance(layer_complexity, dict):
                mixing_complexity = layer_complexity
            else:
                mixing_complexity = {"time": "O(n*d)", "space": "O(n*d)"}
        else:
            mixing_complexity = {"time": "O(n*d)", "space": "O(n*d)"}
        ffn_complexity = self.ffn.complexity

        return {
            "time": f"max({mixing_complexity['time']}, {ffn_complexity['time']})",
            "space": f"max({mixing_complexity['space']}, {ffn_complexity['space']})",
        }
