"""Transformer blocks module for spectrans.

This module provides various transformer block implementations that combine
mixing/attention layers with feedforward networks using residual connections
and normalization.

Modules
-------
base
    Base classes for transformer blocks.
spectral
    Spectral transformer blocks using various frequency-domain methods.
hybrid
    Hybrid blocks combining multiple mixing strategies.
"""

from .base import (
    FeedForwardNetwork,
    ParallelBlock,
    PostNormBlock,
    PreNormBlock,
    TransformerBlock,
)
from .hybrid import (
    AdaptiveBlock,
    AlternatingBlock,
    CascadeBlock,
    HybridBlock,
    MultiscaleBlock,
)
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

__all__ = [
    "AFNOBlock",
    "AdaptiveBlock",
    "AlternatingBlock",
    "CascadeBlock",
    "FNO2DBlock",
    "FNOBlock",
    # Spectral blocks
    "FNetBlock",
    "FeedForwardNetwork",
    "GFNetBlock",
    # Hybrid blocks
    "HybridBlock",
    "LSTBlock",
    "MultiscaleBlock",
    "ParallelBlock",
    "PostNormBlock",
    "PreNormBlock",
    "SpectralAttentionBlock",
    # Base blocks
    "TransformerBlock",
    "WaveletBlock",
]
