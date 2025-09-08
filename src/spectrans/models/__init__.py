r"""Spectral transformer model implementations.

This module provides transformer model implementations that replace
traditional attention mechanisms with various spectral mixing approaches.
Each model is optimized for different use cases while maintaining the core
transformer architecture with residual connections, layer normalization,
and feedforward networks.

The models implement spectral transformers including FNet,
Global Filter Networks, AFNO, spectral attention variants, and hybrid
architectures that combine spectral and spatial processing.

Modules
-------
afno
    Adaptive Fourier Neural Operator models.
base
    Base classes for models and components.
fnet
    FNet models using Fourier mixing.
fno_transformer
    Fourier Neural Operator transformer models.
gfnet
    Global Filter Network models.
hybrid
    Hybrid models combining spectral and attention.
lst
    Linear Spectral Transform models.
spectral_attention
    Models using spectral attention mechanisms.
wavenet_transformer
    Wavelet-based transformer models.

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
    FNet model with Fourier mixing layers.
FNetEncoder
    FNet encoder stack for encoder-only architectures.
GFNet
    Global Filter Network model with learnable spectral filters.
GFNetEncoder
    GFNet encoder stack implementation.
AFNOEncoder
    Adaptive Fourier Neural Operator encoder.
AFNOModel
    AFNO model for various tasks.
SpectralAttentionEncoder
    Encoder using spectral attention with random Fourier features.
SpectralAttentionTransformer
    Spectral attention transformer model.
PerformerTransformer
    Performer-style transformer with linear attention approximation.
LSTEncoder
    Linear Spectral Transform encoder using DCT/DST.
LSTDecoder
    Linear Spectral Transform decoder implementation.
LSTTransformer
    LST transformer with encoder-decoder architecture.
FNOEncoder
    Fourier Neural Operator encoder for function space learning.
FNODecoder
    FNO decoder for continuous function approximation.
FNOTransformer
    FNO transformer for operator learning.
WaveletEncoder
    Wavelet transform encoder with multiresolution analysis.
WaveletDecoder
    Wavelet decoder for signal reconstruction.
WaveletTransformer
    Wavelet transformer model.
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

Positional encodings with RoPE and ALiBi:

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
   transforms that scale as $O(n \log n)$ or $O(n)$.

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

$$
\mathcal{F}[f * g] = \mathcal{F}[f] \odot \mathcal{F}[g]
$$

This enables efficient global mixing of sequence elements through spectral
transforms like FFT, DCT, or DWT, avoiding the quadratic complexity of
traditional attention mechanisms.

**Model Complexity Comparison:**

- Standard Transformer: $O(n^2 d + nd^2)$ time, $O(n^2 + nd)$ space
- FNet: $O(nd \log n + nd^2)$ time, $O(nd)$ space  
- GFNet: $O(nd \log n + nd^2)$ time, $O(nd)$ space
- AFNO: $O(k_n k_d d + nd \log n)$ time, $O(k_n k_d d)$ space
- LST: $O(nd \log n + nd^2)$ time, $O(nd)$ space
- Wavelet: $O(nd + nd^2)$ time, $O(nd)$ space

Where $n$ is sequence length, $d$ is hidden dimension, and
$k_n, k_d$ are retained spectral modes.

References
----------
James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon. 2022.
FNet: Mixing tokens with Fourier transforms. In Proceedings of the 2022 Conference
of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (NAACL-HLT), pages 4296-4313, Seattle.

Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou. 2021.
Global filter networks for image classification. In Advances in Neural
Information Processing Systems 34 (NeurIPS 2021), pages 980-993.

John Guibas, Morteza Mardani, Zongyi Li, Andrew Tao, Anima Anandkumar, and
Bryan Catanzaro. 2022. Adaptive Fourier neural operators: Efficient token
mixers for transformers. In Proceedings of the International Conference on
Learning Representations (ICLR).

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik
Bhattacharya, Andrew Stuart, and Anima Anandkumar. 2021. Fourier neural
operator for parametric partial differential equations. In Proceedings of
the International Conference on Learning Representations (ICLR).

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song,
Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin,
Lukasz Kaiser, David Belanger, Lucy Colwell, and Adrian Weller. 2021.
Rethinking attention with performers. In Proceedings of the International
Conference on Learning Representations (ICLR).

See Also
--------
[`spectrans.layers.mixing`][] : Spectral mixing layer implementations.
[`spectrans.layers.attention`][] : Spectral attention mechanisms.
[`spectrans.layers.operators`][] : Neural operator layers.
[`spectrans.blocks`][] : Transformer block implementations.
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

# Public API - alphabetically sorted
__all__ = [
    "AFNOEncoder",
    "AFNOModel",
    "ALiBiPositionalBias",
    "AlternatingTransformer",
    "BaseModel",
    "ClassificationHead",
    "FNet",
    "FNetEncoder",
    "FNODecoder",
    "FNOEncoder",
    "FNOTransformer",
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
