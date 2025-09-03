# Spectrans: Spectral Transformers

Welcome to Spectrans, a comprehensive library for spectral transformers and frequency-domain neural networks.

## What is Spectrans?

Spectrans provides efficient implementations of spectral methods for deep learning, including:

- **Spectral Transforms**: FFT, wavelet, cosine, and Hadamard transforms
- **Frequency-Domain Layers**: Fourier mixing, wavelet mixing, and spectral attention
- **Neural Operators**: Fourier Neural Operators (FNO) for PDE solving
- **Kernel Methods**: Random Fourier features and spectral kernel approximations

## Key Features

### 🚀 Performance
- GPU-accelerated spectral operations
- Linear complexity :math:`O(n \log n)` for sequence mixing
- Memory-efficient real-valued FFT variants

### 🧮 Mathematical Rigor
- Energy-preserving unitary transforms
- Parseval's theorem compliance
- Comprehensive mathematical documentation

### 🔧 Flexibility
- Modular design with composable components
- Support for 1D, 2D, and 3D operations
- Easy integration with existing PyTorch models

### 📊 Applications
- Computer vision (image classification, segmentation)
- Natural language processing (long-context modeling)
- Scientific computing (PDE solving, weather prediction)
- Signal processing (time series analysis)

## Quick Example

```python
import torch
from spectrans.layers.mixing import FourierMixing
from spectrans.blocks import SpectralTransformerBlock

# Create a spectral transformer block
block = SpectralTransformerBlock(
    hidden_dim=768,
    num_heads=12,
    mixing_type="fourier"
)

# Process input sequences
x = torch.randn(32, 512, 768)  # (batch, seq_len, hidden_dim)
output = block(x)
```

## Installation

```bash
pip install spectrans
```

For development installation:
```bash
git clone https://github.com/aaronstevenwhite/spectrans.git
cd spectrans
pip install -e ".[dev]"
```

## Getting Started

- [API Reference](api/index.md) - Detailed documentation of all modules

## Mathematical Foundation

Spectrans is built on solid mathematical foundations:

- **Fourier Transform**: Global frequency analysis with :math:`O(n \log n)` complexity
- **Wavelet Transform**: Multi-resolution analysis for hierarchical features
- **Kernel Methods**: Efficient approximations of infinite-dimensional mappings
- **Neural Operators**: Learning solution operators for differential equations

## Why Spectral Methods?

Traditional attention mechanisms have :math:`O(n^2)` complexity, limiting their scalability. Spectral methods offer:

1. **Efficiency**: :math:`O(n \log n)` complexity for global interactions
2. **Interpretability**: Frequency-domain processing reveals signal structure
3. **Invariances**: Natural translation and rotation invariances
4. **Generalization**: Resolution-independent learned representations

## Community

- [GitHub Repository](https://github.com/aaronstevenwhite/spectrans)
- [Issue Tracker](https://github.com/aaronstevenwhite/spectrans/issues)

## License

Spectrans is released under the MIT License. See [LICENSE](https://github.com/aaronstevenwhite/spectrans/blob/main/LICENSE) for details.