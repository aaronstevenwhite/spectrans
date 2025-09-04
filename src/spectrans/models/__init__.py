"""Model implementations for spectral transformers.

This module provides complete transformer model implementations that use
various spectral mixing mechanisms instead of traditional attention.
"""

from .afno import AFNOEncoder, AFNOModel
from .base import (
    BaseModel,
    ClassificationHead,
    LearnedPositionalEncoding,
    PositionalEncoding,
    RegressionHead,
    SequenceHead,
)
from .fnet import FNet, FNetEncoder
from .fno_transformer import FNODecoder, FNOEncoder, FNOTransformer
from .gfnet import GFNet, GFNetEncoder
from .hybrid import AlternatingTransformer, HybridEncoder, HybridTransformer
from .lst import LSTDecoder, LSTEncoder, LSTTransformer
from .spectral_attention import (
    PerformerTransformer,
    SpectralAttentionEncoder,
    SpectralAttentionTransformer,
)
from .wavenet_transformer import WaveletDecoder, WaveletEncoder, WaveletTransformer

__all__ = [
    "AFNOEncoder",
    "AFNOModel",
    "AlternatingTransformer",
    "BaseModel",
    "ClassificationHead",
    "FNODecoder",
    "FNOEncoder",
    "FNOTransformer",
    "FNet",
    "FNetEncoder",
    "GFNet",
    "GFNetEncoder",
    "HybridEncoder",
    "HybridTransformer",
    "LSTDecoder",
    "LSTEncoder",
    "LSTTransformer",
    "LearnedPositionalEncoding",
    "PerformerTransformer",
    "PositionalEncoding",
    "RegressionHead",
    "SequenceHead",
    "SpectralAttentionEncoder",
    "SpectralAttentionTransformer",
    "WaveletDecoder",
    "WaveletEncoder",
    "WaveletTransformer",
]
