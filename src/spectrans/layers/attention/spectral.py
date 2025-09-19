r"""Spectral attention mechanisms using kernel approximations.

Implements attention mechanisms based on spectral methods and kernel approximations,
particularly Random Fourier Features (RFF). These methods achieve linear complexity
$O(n)$ instead of the quadratic $O(n^2)$ complexity of standard attention.

Implementations follow the Performer architecture and related work on linearizing
attention through kernel feature maps.

Classes
-------
SpectralAttention
    Multi-head spectral attention using RFF approximation.
PerformerAttention
    Performer-style attention with positive random features.
KernelAttention
    General kernel-based attention with various kernel options.

Examples
--------
Basic spectral attention:

>>> import torch
>>> from spectrans.layers.attention.spectral import SpectralAttention
>>> attn = SpectralAttention(hidden_dim=512, num_heads=8, num_features=256)
>>> x = torch.randn(32, 100, 512)  # (batch, seq_len, dim)
>>> output = attn(x)
>>> assert output.shape == x.shape

Performer attention:

>>> from spectrans.layers.attention.spectral import PerformerAttention
>>> performer = PerformerAttention(
...     hidden_dim=512,
...     num_heads=8,
...     num_features=256,
...     use_orthogonal=True
... )
>>> output = performer(x)

Notes
-----
The spectral attention approximates standard attention as:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \approx \varphi(\mathbf{Q}) (\varphi(\mathbf{K})^T \mathbf{V}) / \text{normalization}
$$

Where $\varphi$ is a feature map (such as RFF) that linearizes the computation. Standard
attention requires $O(n^2d)$ time and $O(n^2)$ space, while spectral attention reduces this
to $O(nrd)$ time and $O(nr)$ space for $r$ features.

Approximation quality scales as $O(1/\sqrt{r})$ with the number of random features.

References
----------
Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane,
Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger,
Lucy Colwell, and Adrian Weller. 2021. Rethinking attention with performers. In Proceedings
of the International Conference on Learning Representations (ICLR).

Hao Peng, Nikolaos Pappas, Dani Yogatama, Roy Schwartz, Noah A. Smith, and Lingpeng Kong.
2021. Random feature attention. In Proceedings of the International Conference on Learning
Representations (ICLR).

See Also
--------
spectrans.kernels.rff : Random Fourier Features implementation.
spectrans.layers.attention.lst : Linear spectral transform attention.
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.base import AttentionLayer
from ...core.registry import register_component
from ...core.types import Tensor
from ...kernels.base import KernelFunction, RandomFeatureMap
from ...kernels.rff import GaussianRFFKernel, RFFAttentionKernel


@register_component("attention", "spectral")
class SpectralAttention(AttentionLayer):
    """Multi-head spectral attention using RFF approximation.

    Implements attention using Random Fourier Features to approximate
    the softmax kernel with linear complexity.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the model.
    num_heads : int, default=8
        Number of attention heads.
    num_features : int | None, default=None
        Number of random features. If None, uses hidden_dim.
    head_dim : int | None, default=None
        Dimension per head. If None, uses hidden_dim // num_heads.
    kernel_type : Literal["gaussian", "softmax"], default="softmax"
        Type of kernel to approximate.
    use_orthogonal : bool, default=True
        Whether to use orthogonal random features.
    feature_redraw : bool, default=False
        Whether to redraw features at each forward pass.
    dropout : float, default=0.0
        Dropout probability.
    use_bias : bool, default=True
        Whether to use bias in projections.

    Attributes
    ----------
    head_dim : int
        Dimension per attention head.
    num_features : int
        Number of random features used.
    q_proj : nn.Linear
        Query projection.
    k_proj : nn.Linear
        Key projection.
    v_proj : nn.Linear
        Value projection.
    out_proj : nn.Linear
        Output projection.
    kernel : RandomFeatureMap | KernelFunction
        Kernel for attention approximation.
    """

    kernel: RandomFeatureMap  # Type annotation for the kernel attribute

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_features: int | None = None,
        head_dim: int | None = None,
        kernel_type: Literal["gaussian", "softmax"] = "softmax",
        use_orthogonal: bool = True,
        feature_redraw: bool = False,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__(hidden_dim, num_heads, dropout)

        # Determine head dimension
        self.head_dim = head_dim or (hidden_dim // num_heads)
        assert (
            self.head_dim * num_heads == hidden_dim
        ), f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"

        # Number of random features
        self.num_features = num_features or self.head_dim

        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)

        # Kernel for approximation
        if kernel_type == "softmax":
            self.kernel = RFFAttentionKernel(
                input_dim=self.head_dim,
                num_features=self.num_features,
                kernel_type="softmax",
                use_orthogonal=use_orthogonal,
                redraw=feature_redraw,
            )
        else:  # gaussian
            self.kernel = GaussianRFFKernel(
                input_dim=self.head_dim,
                num_features=self.num_features,
                sigma=1.0 / math.sqrt(self.head_dim),
                orthogonal=use_orthogonal,
            )

        # Normalization
        self.scale = 1.0 / math.sqrt(self.num_features)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Forward pass of spectral attention.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, seq_len, hidden_dim).
        mask : Tensor | None, default=None
            Attention mask of shape (batch_size, seq_len).
        return_attention : bool, default=False
            Whether to return attention weights (not supported).

        Returns
        -------
        Tensor or tuple[Tensor, Tensor]
            Output tensor of shape (batch_size, seq_len, hidden_dim).
            If return_attention=True, also returns None (weights not available).
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Apply kernel feature maps
        Q_features = self.kernel(Q)  # (batch, heads, seq_len, num_features)
        K_features = self.kernel(K)  # (batch, heads, seq_len, num_features)

        # Apply mask if provided
        if mask is not None:
            # Expand mask for heads dimension
            mask = mask.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq_len, 1)
            K_features = K_features.masked_fill(~mask, 0)
            V = V.masked_fill(~mask, 0)

        # Compute KV (batch, heads, num_features, head_dim)
        KV = torch.matmul(K_features.transpose(-2, -1), V)

        # Compute QKV (batch, heads, seq_len, head_dim)
        out: Tensor = torch.matmul(Q_features, KV)

        # Normalize
        # Compute normalizer: Q_features @ (K_features^T @ 1)
        K_sum = K_features.sum(dim=-2, keepdim=True).transpose(-2, -1)
        normalizer = torch.matmul(Q_features, K_sum) + 1e-6
        out = out / normalizer

        # Transpose back and reshape
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.hidden_dim)

        # Output projection and dropout
        out = self.out_proj(out)
        out = self.dropout(out)

        if return_attention:
            # Attention weights not directly available in linear attention
            return out, None  # type: ignore[return-value]
        return out


@register_component("attention", "performer")
class PerformerAttention(SpectralAttention):
    """Performer-style attention with FAVOR+ algorithm.

    Implements the Performer architecture with positive orthogonal
    random features (FAVOR+) for softmax kernel approximation.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension.
    num_heads : int, default=8
        Number of attention heads.
    num_features : int | None, default=None
        Number of random features.
    generalized : bool, default=False
        Whether to use generalized attention (without softmax).
    dropout : float, default=0.0
        Dropout probability.

    Attributes
    ----------
    generalized : bool
        Whether using generalized attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_features: int | None = None,
        generalized: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_features=num_features,
            kernel_type="softmax",
            use_orthogonal=True,
            feature_redraw=False,
            dropout=dropout,
        )

        self.generalized = generalized

        if generalized:
            # For generalized attention, use different kernel
            self.kernel = RFFAttentionKernel(
                input_dim=self.head_dim,
                num_features=self.num_features,
                kernel_type="relu",
                use_orthogonal=True,
            )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Forward pass of Performer attention.

        Parameters
        ----------
        x : Tensor
            Input of shape (batch_size, seq_len, hidden_dim).
        mask : Tensor | None, default=None
            Attention mask.
        return_attention : bool, default=False
            Whether to return attention weights.

        Returns
        -------
        Tensor or tuple[Tensor, Tensor]
            Output tensor and optionally None for weights.
        """
        if self.generalized:
            # Generalized attention without exponential
            return self._generalized_attention(x, mask)
        else:
            # Standard Performer with softmax approximation
            return super().forward(x, mask, return_attention)

    def _generalized_attention(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Generalized attention without softmax."""
        batch_size, seq_len, _ = x.shape

        # Projections
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Rearrange for heads
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Simple linear attention without kernel features
        # Just use ReLU for positivity
        Q = F.relu(Q) / math.sqrt(self.head_dim)
        K = F.relu(K)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)
            K = K.masked_fill(~mask, 0)
            V = V.masked_fill(~mask, 0)

        # Compute attention
        KV = torch.matmul(K.transpose(-2, -1), V)
        out = torch.matmul(Q, KV)

        # Reshape
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.hidden_dim)

        output: Tensor = self.dropout(self.out_proj(out))
        return output


@register_component("attention", "kernel")
class KernelAttention(AttentionLayer):
    """General kernel-based attention with various kernel options.

    Supports multiple kernel types including Gaussian, polynomial,
    and learnable spectral kernels.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension.
    num_heads : int, default=8
        Number of heads.
    kernel_type : Literal["gaussian", "polynomial", "spectral"], default="gaussian"
        Type of kernel to use.
    rank : int | None, default=None
        Rank for low-rank approximations.
    num_features : int | None, default=None
        Number of features for RFF kernels.
    dropout : float, default=0.0
        Dropout probability.

    Attributes
    ----------
    kernel_type : str
        Type of kernel being used.
    rank : int | None
        Rank for approximations.
    """

    kernel: RandomFeatureMap | KernelFunction  # Type annotation to handle different kernel types

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        kernel_type: Literal["gaussian", "polynomial", "spectral"] = "gaussian",
        rank: int | None = None,
        num_features: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__(hidden_dim, num_heads, dropout)

        self.head_dim = hidden_dim // num_heads
        self.kernel_type = kernel_type
        self.rank = rank or min(64, self.head_dim)

        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Initialize kernel - using union type to handle different kernel types
        if kernel_type == "gaussian":
            from ...kernels import GaussianRFFKernel

            self.kernel = GaussianRFFKernel(
                input_dim=self.head_dim,
                num_features=num_features or self.head_dim,
                sigma=1.0 / math.sqrt(self.head_dim),
            )
            self.use_features = True

        elif kernel_type == "polynomial":
            from ...kernels import PolynomialSpectralKernel

            self.kernel = PolynomialSpectralKernel(
                rank=self.rank,
                degree=2,
            )
            self.use_features = False

        else:  # spectral
            from ...kernels import LearnableSpectralKernel

            self.kernel = LearnableSpectralKernel(
                input_dim=self.head_dim,
                rank=self.rank,
                trainable_eigenvectors=True,
            )
            self.use_features = True

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Forward pass of kernel attention.

        Parameters
        ----------
        x : Tensor
            Input of shape (batch_size, seq_len, hidden_dim).
        mask : Tensor | None, default=None
            Attention mask.
        return_attention : bool, default=False
            Whether to return attention weights.

        Returns
        -------
        Tensor or tuple[Tensor, Tensor]
            Output and optionally attention weights.
        """
        batch_size, seq_len, _ = x.shape

        # Projections and reshape
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        if self.use_features:
            # Use feature-based approximation - kernel should be a callable (RandomFeatureMap)
            if hasattr(self.kernel, "extract_features"):
                Q_feat = self.kernel.extract_features(Q)  # type: ignore[operator]
                K_feat = self.kernel.extract_features(K)  # type: ignore[operator]
            else:
                # Call the kernel as a function
                Q_feat = self.kernel(Q)  # type: ignore[operator]
                K_feat = self.kernel(K)  # type: ignore[operator]

            if mask is not None:
                mask_exp = mask.unsqueeze(1).unsqueeze(-1)
                K_feat = K_feat.masked_fill(~mask_exp, 0)
                V = V.masked_fill(~mask_exp, 0)

            # Linear attention computation
            KV = torch.matmul(K_feat.transpose(-2, -1), V)
            out: Tensor = torch.matmul(Q_feat, KV)

            # Normalize
            K_sum = K_feat.sum(dim=-2, keepdim=True).transpose(-2, -1)
            normalizer = torch.matmul(Q_feat, K_sum) + 1e-6
            out = out / normalizer

            attention_weights: Tensor | None = None

        else:
            # Direct kernel computation (for small sequences)
            # Flatten heads and batch for kernel computation
            Q_flat = Q.reshape(-1, seq_len, self.head_dim)
            K_flat = K.reshape(-1, seq_len, self.head_dim)

            # Compute kernel matrix
            attention_weights = self.kernel.compute(Q_flat, K_flat)  # type: ignore[operator]
            attention_weights = attention_weights.view(batch_size, self.num_heads, seq_len, seq_len)

            if mask is not None:
                mask_exp = mask.unsqueeze(1).unsqueeze(2)
                attention_weights = attention_weights.masked_fill(~mask_exp, -1e9)

            # Normalize
            attention_weights = F.softmax(attention_weights, dim=-1)
            attention_weights = self.dropout(attention_weights)

            # Apply to values
            out = torch.matmul(attention_weights, V)

        # Reshape output
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        if return_attention:
            return out, attention_weights  # type: ignore[return-value]
        return out


__all__ = [
    "KernelAttention",
    "PerformerAttention",
    "SpectralAttention",
]
