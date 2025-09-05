r"""Complete spectral transformer model implementations.

This module provides full transformer model implementations that replace
traditional attention mechanisms with various spectral mixing approaches.
Each model is optimized for different use cases while maintaining the core
transformer architecture with residual connections, layer normalization,
and feedforward networks.

The models implement state-of-the-art spectral transformers including FNet,
Global Filter Networks, AFNO, spectral attention variants, and hybrid
architectures that combine spectral and spatial processing.

Classes
-------
BaseModel
    Abstract base class for all spectral transformer models.
PositionalEncoding
    Sinusoidal positional encoding for sequence models.
LearnedPositionalEncoding
    Learnable positional embedding layer.
RotaryPositionalEncoding
    Rotary Position Embedding (RoPE) for improved length generalization.
ALiBiPositionalBias
    Attention with Linear Biases (ALiBi) positional encoding.
ClassificationHead
    Classification head for sequence classification tasks.
RegressionHead
    Regression head for continuous prediction tasks.
SequenceHead
    Generic sequence-to-sequence head for various tasks.
FNet
    Complete FNet model with Fourier mixing layers.
FNetEncoder
    FNet encoder stack for encoder-only architectures.
GFNet
    Global Filter Network model with learnable spectral filters.
GFNetEncoder
    GFNet encoder stack implementation.
AFNOEncoder
    Adaptive Fourier Neural Operator encoder.
AFNOModel
    Complete AFNO model for various tasks.
SpectralAttentionEncoder
    Encoder using spectral attention with random Fourier features.
SpectralAttentionTransformer
    Complete spectral attention transformer model.
PerformerTransformer
    Performer-style transformer with linear attention approximation.
LSTEncoder
    Linear Spectral Transform encoder using DCT/DST.
LSTDecoder
    Linear Spectral Transform decoder implementation.
LSTTransformer
    Complete LST transformer with encoder-decoder architecture.
FNOEncoder
    Fourier Neural Operator encoder for function space learning.
FNODecoder
    FNO decoder for continuous function approximation.
FNOTransformer
    Complete FNO transformer for operator learning.
WaveletEncoder
    Wavelet transform encoder with multiresolution analysis.
WaveletDecoder
    Wavelet decoder for signal reconstruction.
WaveletTransformer
    Complete wavelet transformer model.
HybridEncoder
    Encoder combining spectral and spatial attention layers.
HybridTransformer
    Hybrid model alternating between spectral and attention mechanisms.
AlternatingTransformer
    Transformer with alternating spectral and attention layers.
StandardAttention
    Standard multi-head self-attention wrapper for hybrid models.

Examples
--------
Basic FNet usage:

>>> import torch
>>> from spectrans.models import FNet
>>> 
>>> # Create FNet model
>>> model = FNet(
...     hidden_dim=512,
...     num_layers=12,
...     vocab_size=32000,
...     max_seq_len=512
... )
>>> 
>>> # Forward pass
>>> input_ids = torch.randint(0, 32000, (2, 128))
>>> outputs = model(input_ids)
>>> print(outputs.shape)  # torch.Size([2, 128, 512])

Global Filter Network example:

>>> from spectrans.models import GFNet
>>> 
>>> # Create GFNet for sequence classification
>>> model = GFNet(
...     hidden_dim=768,
...     num_layers=12,
...     num_classes=10,
...     sequence_length=256
... )
>>> 
>>> # Classification forward pass
>>> x = torch.randn(4, 256, 768)
>>> logits = model(x)
>>> print(logits.shape)  # torch.Size([4, 10])

AFNO for continuous functions:

>>> from spectrans.models import AFNOModel
>>> 
>>> # Create AFNO model
>>> model = AFNOModel(
...     hidden_dim=512,
...     num_layers=8,
...     n_modes=32,
...     input_dim=2,
...     output_dim=1
... )
>>> 
>>> # Function approximation
>>> x = torch.randn(8, 64, 64, 2)  # Batch of 2D functions
>>> output = model(x)
>>> print(output.shape)  # torch.Size([8, 64, 64, 1])

Hybrid spectral-attention model:

>>> from spectrans.models import HybridTransformer
>>> 
>>> # Create hybrid model alternating spectral and attention
>>> model = HybridTransformer(
...     hidden_dim=512,
...     num_layers=12,
...     num_heads=8,
...     spectral_type="fourier",
...     vocab_size=50000
... )
>>> 
>>> input_ids = torch.randint(0, 50000, (2, 256))
>>> outputs = model(input_ids)
>>> print(outputs.shape)  # torch.Size([2, 256, 512])

Wavelet transformer for multiresolution analysis:

>>> from spectrans.models import WaveletTransformer
>>> 
>>> # Create wavelet transformer
>>> model = WaveletTransformer(
...     hidden_dim=512,
...     num_layers=8,
...     wavelet="db4",
...     levels=3,
...     vocab_size=32000
... )
>>> 
>>> input_ids = torch.randint(0, 32000, (2, 512))
>>> outputs = model(input_ids)
>>> print(outputs.shape)  # torch.Size([2, 512, 512])

Advanced positional encodings:

>>> from spectrans.models import FNet, RotaryPositionalEncoding, ALiBiPositionalBias
>>> 
>>> # FNet with RoPE
>>> model = FNet(
...     hidden_dim=512,
...     num_layers=12,
...     vocab_size=50000,
...     pos_encoding=RotaryPositionalEncoding(dim=512)
... )
>>> 
>>> # Or with ALiBi (no positional embeddings needed)
>>> model_alibi = FNet(
...     hidden_dim=512,
...     num_layers=12,
...     vocab_size=50000,
...     pos_encoding=ALiBiPositionalBias(num_heads=8)
... )

Notes
-----
All models in this module follow the same architectural principles:

1. **Spectral Processing**: Replace quadratic attention with efficient spectral
   transforms that scale as :math:`O(n \log n)` or :math:`O(n)`.

2. **Residual Connections**: Maintain gradient flow through residual connections
   around each spectral layer and feedforward network.

3. **Layer Normalization**: Apply layer normalization before spectral mixing
   and feedforward operations for training stability.

4. **Positional Encoding**: Support multiple positional encoding methods including
   sinusoidal, learned embeddings, RoPE, and ALiBi for various sequence modeling needs.

5. **Task Heads**: Provide specialized output heads for classification,
   regression, and sequence-to-sequence tasks.

The mathematical foundation for spectral mixing is based on the convolution
theorem, which states that convolution in the spatial domain is equivalent
to element-wise multiplication in the frequency domain:

.. math::
    \mathcal{F}[f * g] = \mathcal{F}[f] \odot \mathcal{F}[g]

This enables efficient global mixing of sequence elements through spectral
transforms like FFT, DCT, or DWT, avoiding the quadratic complexity of
traditional attention mechanisms.

**Model Complexity Comparison:**

- Standard Transformer: :math:`O(n^2 d + nd^2)` time, :math:`O(n^2 + nd)` space
- FNet: :math:`O(nd \log n + nd^2)` time, :math:`O(nd)` space  
- GFNet: :math:`O(nd \log n + nd^2)` time, :math:`O(nd)` space
- AFNO: :math:`O(k_n k_d d + nd \log n)` time, :math:`O(k_n k_d d)` space
- LST: :math:`O(nd \log n + nd^2)` time, :math:`O(nd)` space
- Wavelet: :math:`O(nd + nd^2)` time, :math:`O(nd)` space

Where :math:`n` is sequence length, :math:`d` is hidden dimension, and
:math:`k_n, k_d` are retained spectral modes.

References
----------
.. [1] Lee-Thorp, J., et al., "FNet: Mixing Tokens with Fourier Transforms", 
       NAACL 2022.
.. [2] Rao, Y., et al., "Global Filter Networks for Image Classification", 
       NeurIPS 2021.  
.. [3] Guibas, J., et al., "Adaptive Fourier Neural Operators", ICLR 2022.
.. [4] Li, Z., et al., "Fourier Neural Operator for Parametric Partial 
       Differential Equations", ICLR 2021.
.. [5] Choromanski, K., et al., "Rethinking Attention with Performers", 
       ICLR 2021.

See Also
--------
spectrans.layers.mixing : Spectral mixing layer implementations.
spectrans.layers.attention : Spectral attention mechanisms.
spectrans.layers.operators : Neural operator layers.
spectrans.blocks : Transformer block implementations.
"""

from .afno import AFNOEncoder, AFNOModel
from .base import (
    ALiBiPositionalBias,
    BaseModel,
    ClassificationHead,
    LearnedPositionalEncoding,
    PositionalEncoding,
    RegressionHead,
    RotaryPositionalEncoding,
    SequenceHead,
)
from .fnet import FNet, FNetEncoder
from .fno_transformer import FNODecoder, FNOEncoder, FNOTransformer
from .gfnet import GFNet, GFNetEncoder
from .hybrid import AlternatingTransformer, HybridEncoder, HybridTransformer, StandardAttention
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
    "ALiBiPositionalBias",
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
    "RotaryPositionalEncoding",
    "SequenceHead",
    "SpectralAttentionEncoder",
    "SpectralAttentionTransformer",
    "StandardAttention",
    "WaveletDecoder",
    "WaveletEncoder",
    "WaveletTransformer",
]
