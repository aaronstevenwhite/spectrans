r"""Linear Spectral Transform (LST) attention mechanisms.

This module implements attention mechanisms based on Linear Spectral Transforms
such as Discrete Cosine Transform (DCT), Discrete Sine Transform (DST), and
Hadamard Transform. These transforms enable efficient $O(n \log n)$ attention
computation while maintaining orthogonality properties.

The LST attention mechanism replaces the standard $QK^T$ computation with
element-wise multiplication in the transform domain, significantly reducing
computational complexity for long sequences.

Classes
-------
LSTAttention
    Linear Spectral Transform attention with various transform options.
DCTAttention
    Attention using Discrete Cosine Transform.
HadamardAttention
    Attention using fast Hadamard transform.

Examples
--------
Basic LST attention with DCT:

>>> import torch
>>> from spectrans.layers.attention.lst import LSTAttention
>>> attn = LSTAttention(
...     hidden_dim=512,
...     num_heads=8,
...     transform_type="dct"
... )
>>> x = torch.randn(32, 100, 512)
>>> output = attn(x)
>>> assert output.shape == x.shape

Multi-transform attention:

>>> from spectrans.layers.attention.lst import LSTAttention
>>> attn = LSTAttention(
...     hidden_dim=512,
...     num_heads=8,
...     transform_type="mixed",  # Uses different transforms per head
... )
>>> output = attn(x)

Notes
-----
The LST attention computes:

$$
\text{Attention}(Q, K, V) = T^{-1}(\Lambda \odot (T(Q) \odot T(K) \odot T(V)))
$$

Where:

- $T$ is an orthogonal transform (DCT, DST, Hadamard)
- $\Lambda$ is a learnable diagonal scaling matrix
- $\odot$ denotes element-wise multiplication

Complexity:

- Standard attention: $O(n^2d)$
- LST attention: $O(nd \log n)$

The orthogonality of transforms preserves information while
enabling efficient computation in the frequency domain.

References
----------
James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon. 2022.
FNet: Mixing tokens with Fourier transforms. In Proceedings of the 2022 Conference
of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (NAACL-HLT), pages 4296-4313, Seattle.

Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham,
Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. 2021. Long range arena:
A benchmark for efficient transformers. In Proceedings of the International Conference
on Learning Representations (ICLR).

See Also
--------
spectrans.transforms.cosine : DCT/DST implementations.
spectrans.transforms.hadamard : Hadamard transform.
spectrans.layers.attention.spectral : Spectral attention with RFF.
"""

import math
from typing import Literal

import torch
import torch.nn as nn

from ...core.base import AttentionLayer
from ...core.registry import register_component
from ...core.types import Tensor
from ...transforms import DCT, DST, HadamardTransform
from ...transforms.base import SpectralTransform


@register_component("attention", "lst")
class LSTAttention(AttentionLayer):
    """Linear Spectral Transform attention mechanism.

    Implements efficient attention using orthogonal transforms
    (DCT, DST, Hadamard) with learnable diagonal scaling.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the model.
    num_heads : int, default=8
        Number of attention heads.
    transform_type : Literal["dct", "dst", "hadamard", "mixed"], default="dct"
        Type of transform to use. "mixed" uses different transforms per head.
    learnable_scale : bool, default=True
        Whether to use learnable diagonal scaling matrix.
    normalize : bool, default=True
        Whether to normalize in transform domain.
    dropout : float, default=0.0
        Dropout probability.
    use_bias : bool, default=True
        Whether to use bias in projections.

    Attributes
    ----------
    head_dim : int
        Dimension per attention head.
    transform_type : str
        Type of transform being used.
    transforms : nn.ModuleList
        List of transforms (one per head if mixed).
    scale : nn.Parameter | None
        Learnable diagonal scaling if enabled.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        transform_type: Literal["dct", "dst", "hadamard", "mixed"] = "dct",
        learnable_scale: bool = True,
        normalize: bool = True,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__(hidden_dim, num_heads, dropout)

        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, \
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"

        self.transform_type = transform_type
        self.normalize = normalize

        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)

        # Initialize transforms
        self.transforms: nn.ModuleList = nn.ModuleList()  # Contains SpectralTransform objects
        if transform_type == "mixed":
            # Use different transforms for different heads
            transform_types = ["dct", "dst", "hadamard"]
            for i in range(num_heads):
                t_type = transform_types[i % len(transform_types)]
                self.transforms.append(self._create_transform(t_type))
        else:
            # Use same transform for all heads
            transform = self._create_transform(transform_type)
            for _ in range(num_heads):
                self.transforms.append(transform)

        # Learnable diagonal scaling
        if learnable_scale:
            # Different scale per head and position
            self.scale = nn.Parameter(
                torch.ones(num_heads, 1, self.head_dim)
            )
        else:
            self.register_buffer(
                'scale',
                torch.ones(num_heads, 1, self.head_dim)
            )

    def _create_transform(
        self,
        transform_type: str,
    ) -> SpectralTransform:
        """Create a transform module.

        Parameters
        ----------
        transform_type : str
            Type of transform ("dct", "dst", or "hadamard").

        Returns
        -------
        SpectralTransform
            Transform module.
        """
        if transform_type == "dct":
            return DCT(normalized=True)
        elif transform_type == "dst":
            return DST(normalized=True)
        elif transform_type == "hadamard":
            return HadamardTransform(normalized=True)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Forward pass of LST attention.

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
            If return_attention=True, returns (output, None).
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

        # Apply transforms per head
        outputs = []
        for head_idx in range(self.num_heads):
            q_head = Q[:, head_idx]  # (batch, seq_len, head_dim)
            k_head = K[:, head_idx]
            v_head = V[:, head_idx]

            # Get transform for this head
            transform: SpectralTransform
            if self.transform_type == "mixed":
                transform = self.transforms[head_idx]  # type: ignore[assignment]
            else:
                transform = self.transforms[0]  # type: ignore[assignment]

            # Apply transform along sequence dimension
            q_transformed = transform.transform(q_head, dim=-2)
            k_transformed = transform.transform(k_head, dim=-2)
            v_transformed = transform.transform(v_head, dim=-2)

            # Apply mask in transform domain if provided
            if mask is not None:
                # Transform mask to frequency domain
                mask_float = mask.float().unsqueeze(-1)  # (batch, seq_len, 1)
                mask_transformed = transform.transform(mask_float, dim=-2)
                k_transformed = k_transformed * mask_transformed
                v_transformed = v_transformed * mask_transformed

            # Element-wise multiplication in transform domain
            # This replaces the QK^T computation
            attention_transformed = q_transformed * k_transformed * self.scale[head_idx]

            # Apply to values
            output_transformed = attention_transformed * v_transformed

            # Normalize if requested
            if self.normalize:
                # Compute normalization factor
                norm_factor = torch.abs(attention_transformed).sum(
                    dim=-1, keepdim=True
                ) + 1e-6
                output_transformed = output_transformed / norm_factor

            # Inverse transform
            output_head = transform.inverse_transform(output_transformed, dim=-2)

            # Real part for numerical stability
            if torch.is_complex(output_head):
                output_head = output_head.real

            outputs.append(output_head.unsqueeze(1))

        # Concatenate heads
        out = torch.cat(outputs, dim=1)  # (batch, heads, seq_len, head_dim)

        # Reshape
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.hidden_dim)

        # Output projection and dropout
        out = self.out_proj(out)
        out = self.dropout(out)

        output: Tensor = out
        if return_attention:
            # Attention weights not available in LST
            return output, None  # type: ignore[return-value]
        return output



@register_component("attention", "dct")
class DCTAttention(LSTAttention):
    """Attention using Discrete Cosine Transform.

    Specialized LST attention that uses DCT for all heads,
    optimized for real-valued signals with energy compaction.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension.
    num_heads : int, default=8
        Number of attention heads.
    dct_type : int, default=2
        DCT type (2 is most common).
    learnable_scale : bool, default=True
        Whether to use learnable scaling.
    dropout : float, default=0.0
        Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dct_type: int = 2,
        learnable_scale: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            transform_type="dct",
            learnable_scale=learnable_scale,
            normalize=True,
            dropout=dropout,
        )

        self.dct_type = dct_type

        # Override transform with specific DCT type
        # Note: Current DCT implementation only supports type 2
        # Future versions may support other types
        if dct_type != 2:
            # For now, still use type 2 DCT
            pass


@register_component("attention", "hadamard")
class HadamardAttention(LSTAttention):
    """Attention using fast Hadamard transform.

    Uses Hadamard transform for O(n log n) attention computation
    with binary coefficients for efficiency.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension.
    num_heads : int, default=8
        Number of attention heads.
    scale_by_sqrt : bool, default=True
        Whether to scale by sqrt(n) for orthogonality.
    learnable_scale : bool, default=True
        Whether to use learnable diagonal scaling.
    dropout : float, default=0.0
        Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        scale_by_sqrt: bool = True,
        learnable_scale: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            transform_type="hadamard",
            learnable_scale=learnable_scale,
            normalize=True,
            dropout=dropout,
        )

        self.scale_by_sqrt = scale_by_sqrt

        # Additional scaling for Hadamard
        if scale_by_sqrt:
            # Initialize scale with 1/sqrt(n) factor
            with torch.no_grad():
                self.scale.data = self.scale.data / math.sqrt(self.head_dim)


@register_component("attention", "mixed_spectral")
class MixedSpectralAttention(AttentionLayer):
    """Mixed spectral attention using multiple transform types.

    Combines different spectral transforms across heads for
    diverse frequency representations.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension.
    num_heads : int, default=8
        Number of attention heads (should be divisible by 3 for even split).
    use_fft : bool, default=True
        Whether to include FFT heads.
    use_dct : bool, default=True
        Whether to include DCT heads.
    use_hadamard : bool, default=True
        Whether to include Hadamard heads.
    dropout : float, default=0.0
        Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 9,  # Divisible by 3
        use_fft: bool = True,
        use_dct: bool = True,
        use_hadamard: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(hidden_dim, num_heads, dropout)

        self.head_dim = hidden_dim // num_heads

        # Count enabled transform types
        enabled_transforms = []
        if use_fft:
            enabled_transforms.append("fft")
        if use_dct:
            enabled_transforms.append("dct")
        if use_hadamard:
            enabled_transforms.append("hadamard")

        if not enabled_transforms:
            raise ValueError("At least one transform type must be enabled")

        self.enabled_transforms = enabled_transforms

        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Assign transforms to heads
        self.head_transforms = []
        for i in range(num_heads):
            transform_type = enabled_transforms[i % len(enabled_transforms)]
            self.head_transforms.append(transform_type)

        # Create transform modules
        from ...transforms import FFT1D

        self.fft = FFT1D() if use_fft else None
        self.dct = DCT(normalized=True) if use_dct else None
        self.hadamard = HadamardTransform(normalized=True) if use_hadamard else None

        # Learnable scales per transform type
        self.scales = nn.ParameterDict({
            t: nn.Parameter(torch.ones(1, 1, self.head_dim))
            for t in enabled_transforms
        })

    def forward(
        self,
        x: Tensor,
        _mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Forward pass of mixed spectral attention.

        Parameters
        ----------
        x : Tensor
            Input of shape (batch_size, seq_len, hidden_dim).
        _mask : Tensor | None, default=None
            Attention mask (not implemented for spectral attention).
        return_attention : bool, default=False
            Whether to return attention weights.

        Returns
        -------
        Tensor or tuple[Tensor, Tensor]
            Output and optionally None for weights.
        """
        batch_size, seq_len, _ = x.shape

        # Projections
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Process each head with its assigned transform
        outputs = []
        for head_idx in range(self.num_heads):
            transform_type = self.head_transforms[head_idx]
            scale = self.scales[transform_type]

            q_head = Q[:, head_idx]
            k_head = K[:, head_idx]
            v_head = V[:, head_idx]

            # Apply appropriate transform
            if transform_type == "fft":
                if self.fft is None:
                    raise RuntimeError("FFT transform not initialized")
                q_t = self.fft.transform(q_head, dim=-2)
                k_t = self.fft.transform(k_head, dim=-2)
                v_t = self.fft.transform(v_head, dim=-2)

                # Complex multiplication in frequency domain
                attn_t = q_t * k_t.conj() * scale
                out_t = attn_t * v_t

                # Inverse transform
                out_head = self.fft.inverse_transform(out_t, dim=-2).real

            elif transform_type == "dct":
                if self.dct is None:
                    raise RuntimeError("DCT transform not initialized")
                q_t = self.dct.transform(q_head, dim=-2)
                k_t = self.dct.transform(k_head, dim=-2)
                v_t = self.dct.transform(v_head, dim=-2)

                attn_t = q_t * k_t * scale
                out_t = attn_t * v_t

                out_head = self.dct.inverse_transform(out_t, dim=-2)

            else:  # hadamard
                if self.hadamard is None:
                    raise RuntimeError("Hadamard transform not initialized")
                q_t = self.hadamard.transform(q_head, dim=-2)
                k_t = self.hadamard.transform(k_head, dim=-2)
                v_t = self.hadamard.transform(v_head, dim=-2)

                attn_t = q_t * k_t * scale
                out_t = attn_t * v_t

                out_head = self.hadamard.inverse_transform(out_t, dim=-2)

            outputs.append(out_head.unsqueeze(1))

        # Concatenate and reshape
        out = torch.cat(outputs, dim=1)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.hidden_dim)

        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        output: Tensor = out
        if return_attention:
            return output, None  # type: ignore[return-value]
        return output



__all__ = [
    "DCTAttention",
    "HadamardAttention",
    "LSTAttention",
    "MixedSpectralAttention",
]
