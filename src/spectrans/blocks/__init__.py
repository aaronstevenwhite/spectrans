r"""Transformer block implementations for spectral architectures.

This module provides transformer blocks that combine spectral mixing or attention
layers with feedforward networks, residual connections, and normalization. The blocks
implement different architectural patterns including pre-norm, post-norm, parallel,
and hybrid configurations for various spectral transformer models.

Modules
-------
base
    Base classes for transformer blocks.
hybrid
    Hybrid blocks combining multiple mixing strategies.
spectral
    Spectral transformer blocks using frequency-domain methods.

Classes
-------
AFNOBlock
    Adaptive Fourier Neural Operator block with mode truncation.
AdaptiveBlock
    Block with adaptive routing between components.
AlternatingBlock
    Alternates between different mixing strategies.
CascadeBlock
    Cascades multiple blocks with different configurations.
FeedForwardNetwork
    Standard MLP feedforward network.
FNetBlock
    FNet-style block with Fourier mixing.
FNO2DBlock
    2D Fourier Neural Operator block for spatial data.
FNOBlock
    1D Fourier Neural Operator block.
GFNetBlock
    Global Filter Network block with learnable filters.
HybridBlock
    Combines multiple mixing strategies in parallel.
LSTBlock
    Linear Spectral Transform block.
MultiscaleBlock
    Multi-resolution processing with wavelets.
ParallelBlock
    Parallel execution of mixing and feedforward.
PostNormBlock
    Post-normalization transformer block.
PreNormBlock
    Pre-normalization transformer block.
SpectralAttentionBlock
    Block using spectral attention mechanisms.
TransformerBlock
    Base class for all transformer blocks.
WaveletBlock
    Block using wavelet transforms for mixing.

Examples
--------
Using a FNet block:

>>> import torch
>>> from spectrans.blocks import FNetBlock
>>>
>>> block = FNetBlock(hidden_dim=768, ffn_hidden_dim=3072)
>>> x = torch.randn(32, 512, 768)
>>> output = block(x)
>>> assert output.shape == x.shape

Using a hybrid block with multiple mixing strategies:

>>> from spectrans.blocks import AlternatingBlock
>>> from spectrans.layers.mixing.fourier import FourierMixing
>>> from spectrans.layers.mixing.wavelet import WaveletMixing
>>>
>>> layer1 = FourierMixing(hidden_dim=512)
>>> layer2 = WaveletMixing(hidden_dim=512, wavelet='db4')
>>> block = AlternatingBlock(layer1=layer1, layer2=layer2, hidden_dim=512)
>>> output = block(x)

Using parallel execution:

>>> from spectrans.blocks import ParallelBlock
>>> from spectrans.layers.mixing.fourier import FourierMixing
>>>
>>> mixing = FourierMixing(hidden_dim=768)
>>> block = ParallelBlock(mixing_layer=mixing, hidden_dim=768)
>>> output = block(x)

Notes
-----
**Architectural Patterns:**

1. **Pre-Norm**: LayerNorm → Mixing → Residual → LayerNorm → FFN → Residual
2. **Post-Norm**: Mixing → Residual → LayerNorm → FFN → Residual → LayerNorm
3. **Parallel**: Mixing and FFN execute simultaneously with single residual
4. **Hybrid**: Multiple mixing strategies combined with learnable or fixed weights

**Complexity Comparison:**

- Standard Transformer: $O(n^2 d)$ per block
- FNet Block: $O(nd \log n)$ per block
- GFNet Block: $O(nd \log n)$ with learnable parameters
- Wavelet Block: $O(nd)$ with multi-resolution analysis
- Hybrid Block: Weighted combination of component complexities

All blocks maintain:
- Residual connections for gradient flow
- LayerNorm for training stability
- Dropout for regularization
- Optional activation checkpointing

References
----------
James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon. 2022.
FNet: Mixing tokens with Fourier transforms. In Proceedings of the 2022 Conference
of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (NAACL-HLT), pages 4296-4313, Seattle.

Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou. 2021.
Global filter networks for image classification. In Advances in Neural
Information Processing Systems 34 (NeurIPS 2021), pages 980-993.

See Also
--------
[`spectrans.layers`][] : Layer implementations used in blocks.
[`spectrans.models`][] : Models built from these blocks.
[`spectrans.blocks.base`][] : Base classes and interfaces.
"""

from .base import FeedForwardNetwork, ParallelBlock, PostNormBlock, PreNormBlock, TransformerBlock
from .hybrid import AdaptiveBlock, AlternatingBlock, CascadeBlock, HybridBlock, MultiscaleBlock
from .spectral import (
    AFNOBlock,
    FNetBlock,
    FNO2DBlock,
    FNOBlock,
    GFNetBlock,
    LSTBlock,
    SpectralAttentionBlock,
    WaveletBlock,
)

# Public API - alphabetically sorted
__all__ = [
    "AFNOBlock",
    "AdaptiveBlock",
    "AlternatingBlock",
    "CascadeBlock",
    "FNO2DBlock",
    "FNOBlock",
    "FNetBlock",
    "FeedForwardNetwork",
    "GFNetBlock",
    "HybridBlock",
    "LSTBlock",
    "MultiscaleBlock",
    "ParallelBlock",
    "PostNormBlock",
    "PreNormBlock",
    "SpectralAttentionBlock",
    "TransformerBlock",
    "WaveletBlock",
]
