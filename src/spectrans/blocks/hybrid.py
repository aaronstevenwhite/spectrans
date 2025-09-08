"""Hybrid spatial-spectral transformer blocks.

This module provides transformer blocks that combine different types of mixing
layers, alternating between spectral and spatial processing or using adaptive
selection mechanisms.

Classes
-------
HybridBlock
    Base class for hybrid transformer blocks.
AlternatingBlock
    Block that alternates between two different mixing layers.
AdaptiveBlock
    Block that adaptively selects between mixing strategies.
MultiscaleBlock
    Block that processes multiple scales in parallel.
CascadeBlock
    Block that cascades multiple mixing strategies sequentially.

Examples
--------
Creating hybrid blocks with different strategies:

>>> from spectrans.blocks.hybrid import AlternatingBlock
>>> from spectrans.layers.mixing.fourier import FourierMixing
>>> from spectrans.layers.attention.spectral import SpectralAttention
>>> block = AlternatingBlock(
...     layer1=FourierMixing(hidden_dim=768),
...     layer2=SpectralAttention(hidden_dim=768, num_heads=8),
...     hidden_dim=768
... )

Notes
-----
Hybrid blocks are useful for:
- Combining complementary strengths of different mixing strategies
- Adapting to different types of patterns in the data
- Balancing computational efficiency with modeling capacity
- Capturing both local and global dependencies

References
----------
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly,
Jakob Uszkoreit, and Neil Houlsby. 2021. An image is worth 16x16 words: Transformers
for image recognition at scale. In Proceedings of the International Conference on
Learning Representations (ICLR).

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and
Baining Guo. 2021. Swin transformer: Hierarchical vision transformer using shifted
windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision
(ICCV), pages 10012-10022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.base import SpectralComponent
from ..core.registry import register_component
from ..layers.mixing.base import MixingLayer
from .base import FeedForwardNetwork


class HybridBlock(SpectralComponent):
    """Base class for hybrid transformer blocks.

    This class provides the foundation for blocks that combine multiple
    mixing strategies in various ways.

    Parameters
    ----------
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
    hidden_dim : int
        Hidden dimension of the model.
    ffn : FeedForwardNetwork | None
        The feedforward network.
    dropout : nn.Dropout
        Dropout layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Default FFN dimension
        if ffn_hidden_dim is None:
            ffn_hidden_dim = 4 * hidden_dim

        # Feedforward network
        self.ffn = FeedForwardNetwork(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=activation,
            dropout=dropout,
        )

        # Normalization layers (to be used by subclasses)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=norm_eps)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout)



@register_component("block", "alternating")
class AlternatingBlock(HybridBlock):
    """Transformer block that alternates between two mixing strategies.

    This block can be used in alternating patterns, e.g., even layers use
    one type of mixing and odd layers use another.

    Parameters
    ----------
    layer1 : MixingLayer | nn.Module
        First mixing layer.
    layer2 : MixingLayer | nn.Module
        Second mixing layer.
    hidden_dim : int
        Hidden dimension of the model.
    use_layer1 : bool, optional
        Whether to use layer1 (True) or layer2 (False). Default is True.
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
    layer1 : MixingLayer | nn.Module
        First mixing layer.
    layer2 : MixingLayer | nn.Module
        Second mixing layer.
    use_layer1 : bool
        Which layer to use for this block.
    """

    def __init__(
        self,
        layer1: MixingLayer | nn.Module,
        layer2: MixingLayer | nn.Module,
        hidden_dim: int,
        use_layer1: bool = True,
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=activation,
            dropout=dropout,
            norm_eps=norm_eps,
        )
        self.layer1 = layer1
        self.layer2 = layer2
        self.use_layer1 = use_layer1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the alternating block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, hidden_dim).
        """
        # Select which layer to use
        mixing_layer = self.layer1 if self.use_layer1 else self.layer2

        # Apply mixing with pre-norm
        h = x + self.dropout(mixing_layer(self.norm1(x)))

        # Apply FFN with pre-norm
        output: Tensor = h + self.dropout(self.ffn(self.norm2(h)))

        return output

    def set_layer(self, use_layer1: bool) -> None:
        """Set which layer to use.

        Parameters
        ----------
        use_layer1 : bool
            Whether to use layer1 (True) or layer2 (False).
        """
        self.use_layer1 = use_layer1



@register_component("block", "adaptive")
class AdaptiveBlock(HybridBlock):
    """Transformer block that adaptively selects between mixing strategies.

    This block uses a gating mechanism to dynamically choose or blend
    between different mixing strategies based on the input.

    Parameters
    ----------
    layers : list[MixingLayer | nn.Module]
        List of mixing layers to choose from.
    hidden_dim : int
        Hidden dimension of the model.
    gate_type : str, optional
        Type of gating ('soft' for weighted sum, 'hard' for selection).
        Default is 'soft'.
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
    layers : nn.ModuleList
        List of mixing layers.
    gate : nn.Linear
        Gating network for layer selection.
    gate_type : str
        Type of gating mechanism.
    """

    def __init__(
        self,
        layers: list[MixingLayer | nn.Module],
        hidden_dim: int,
        gate_type: str = "soft",
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=activation,
            dropout=dropout,
            norm_eps=norm_eps,
        )
        self.layers = nn.ModuleList(layers)
        self.num_layers = len(layers)
        self.gate_type = gate_type

        # Gating network
        self.gate = nn.Linear(hidden_dim, self.num_layers)

        # Initialize gate to uniform weights
        nn.init.constant_(self.gate.weight, 0)
        nn.init.constant_(self.gate.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adaptive block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, hidden_dim).
        """
        # Normalize input for mixing
        normed = self.norm1(x)

        # Compute gate values
        gate_input = normed.mean(dim=1)  # (batch_size, hidden_dim)
        gate_logits = self.gate(gate_input)  # (batch_size, num_layers)

        if self.gate_type == "soft":
            # Soft gating: weighted sum of all layers
            gate_weights = F.softmax(gate_logits, dim=-1)  # (batch_size, num_layers)

            # Apply each layer and combine
            mixed = torch.zeros_like(x)
            for i, layer in enumerate(self.layers):
                weight = gate_weights[:, i:i+1].unsqueeze(1)  # (batch_size, 1, 1)
                mixed = mixed + weight * layer(normed)
        else:  # hard gating
            # Hard gating: select single layer
            gate_idx = torch.argmax(gate_logits, dim=-1)  # (batch_size,)

            # Apply selected layer for each sample
            mixed = torch.zeros_like(x)
            for i in range(x.shape[0]):
                idx = int(gate_idx[i].item())
                mixed[i] = self.layers[idx](normed[i:i+1])

        # Add residual
        h = x + self.dropout(mixed)

        # Apply FFN with pre-norm
        output: Tensor = h + self.dropout(self.ffn(self.norm2(h)))

        return output



@register_component("block", "multiscale")
class MultiscaleBlock(HybridBlock):
    """Transformer block that processes multiple scales in parallel.

    This block applies different mixing strategies at different scales
    and combines their outputs, capturing both local and global patterns.

    Parameters
    ----------
    layers : list[MixingLayer | nn.Module]
        List of mixing layers for different scales.
    hidden_dim : int
        Hidden dimension of the model.
    fusion_type : str, optional
        How to fuse outputs ('concat', 'add', 'weighted'). Default is 'add'.
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
    layers : nn.ModuleList
        List of mixing layers for different scales.
    fusion_type : str
        Type of fusion mechanism.
    fusion_weights : nn.Parameter | None
        Learnable weights for fusion (if fusion_type is 'weighted').
    fusion_proj : nn.Linear | None
        Projection for concatenation fusion.
    """

    def __init__(
        self,
        layers: list[MixingLayer | nn.Module],
        hidden_dim: int,
        fusion_type: str = "add",
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=activation,
            dropout=dropout,
            norm_eps=norm_eps,
        )
        self.layers = nn.ModuleList(layers)
        self.num_scales = len(layers)
        self.fusion_type = fusion_type

        # Type annotations for optional attributes
        self.fusion_weights: nn.Parameter | None
        self.fusion_proj: nn.Linear | None

        # Fusion mechanisms
        if fusion_type == "weighted":
            self.fusion_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        else:
            self.fusion_weights = None

        if fusion_type == "concat":
            self.fusion_proj = nn.Linear(hidden_dim * self.num_scales, hidden_dim)
        else:
            self.fusion_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the multiscale block.

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
        normed = self.norm1(x)

        # Apply each scale
        outputs = []
        for layer in self.layers:
            outputs.append(layer(normed))

        # Fuse outputs
        if self.fusion_type == "add":
            mixed = sum(outputs) / self.num_scales
        elif self.fusion_type == "weighted":
            assert self.fusion_weights is not None, "fusion_weights should not be None for weighted fusion"
            weights = F.softmax(self.fusion_weights, dim=0)
            mixed = sum(w * out for w, out in zip(weights, outputs, strict=False))
        elif self.fusion_type == "concat":
            mixed = torch.cat(outputs, dim=-1)
            assert self.fusion_proj is not None, "fusion_proj should not be None for concat fusion"
            mixed = self.fusion_proj(mixed)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # Add residual
        h = x + self.dropout(mixed)

        # Apply FFN with pre-norm
        output: Tensor = h + self.dropout(self.ffn(self.norm2(h)))

        return output



@register_component("block", "cascade")
class CascadeBlock(HybridBlock):
    """Transformer block that cascades multiple mixing strategies.

    This block applies mixing layers sequentially, allowing each layer
    to refine the representations produced by the previous one.

    Parameters
    ----------
    layers : list[MixingLayer | nn.Module]
        List of mixing layers to cascade.
    hidden_dim : int
        Hidden dimension of the model.
    share_norm : bool, optional
        Whether to share normalization across layers. Default is False.
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
    layers : nn.ModuleList
        List of mixing layers to cascade.
    norms : nn.ModuleList
        Normalization layers for each mixing layer.
    share_norm : bool
        Whether normalization is shared.
    """

    def __init__(
        self,
        layers: list[MixingLayer | nn.Module],
        hidden_dim: int,
        share_norm: bool = False,
        ffn_hidden_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_eps: float = 1e-12,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=activation,
            dropout=dropout,
            norm_eps=norm_eps,
        )
        self.layers = nn.ModuleList(layers)
        self.share_norm = share_norm

        # Create normalization layers
        if share_norm:
            # Use the same norm for all layers
            self.norms = nn.ModuleList([self.norm1] * len(layers))
        else:
            # Create separate norms for each layer
            self.norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim, eps=norm_eps) for _ in layers
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the cascade block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, hidden_dim).
        """
        # Cascade through mixing layers
        h = x
        for layer, norm in zip(self.layers, self.norms, strict=False):
            h = h + self.dropout(layer(norm(h)))

        # Apply FFN with pre-norm
        output: Tensor = h + self.dropout(self.ffn(self.norm2(h)))

        return output

