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
- Linearithmic complexity $O(n \log n)$ for sequence mixing
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
from spectrans.models import FNet
from spectrans.layers.mixing import FourierMixing

# Create a complete FNet model for text classification
model = FNet(
    vocab_size=30000,
    hidden_dim=768,
    num_layers=12,
    max_sequence_length=512,
    num_classes=2
)

# Forward pass with token IDs
input_ids = torch.randint(0, 30000, (32, 256))  # (batch, seq_len)
logits = model(input_ids=input_ids)
print(f"Classification logits: {logits.shape}")  # (32, 2)

# Or use individual components
fourier_mixing = FourierMixing(hidden_dim=768, dropout=0.1)
embeddings = torch.randn(32, 256, 768)  # (batch, seq_len, hidden_dim)
mixed = fourier_mixing(embeddings)
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

- [Installation](installation.md) - Complete installation guide
- [Quick Start](quickstart.md) - Step-by-step tutorial with examples
- [Contributing](contributing.md) - Development setup and contribution workflow
- [API Reference](api/index.md) - Detailed documentation of all modules

## Mathematical Foundation

Spectrans is built on solid mathematical foundations:

- **Fourier Transform**: Global frequency analysis with $O(n \log n)$ complexity
- **Wavelet Transform**: Multi-resolution analysis for hierarchical features
- **Kernel Methods**: Efficient approximations of infinite-dimensional mappings
- **Neural Operators**: Learning solution operators for differential equations

## Why Spectral Methods?

Traditional attention mechanisms have $O(n^2)$ complexity, limiting their scalability. Spectral methods offer:

1. **Efficiency**: $O(n \log n)$ complexity for global interactions
2. **Interpretability**: Frequency-domain processing reveals signal structure
3. **Invariances**: Natural translation and rotation invariances
4. **Generalization**: Resolution-independent learned representations

## Community

- [GitHub Repository](https://github.com/aaronstevenwhite/spectrans)
- [Issue Tracker](https://github.com/aaronstevenwhite/spectrans/issues)

## Citation

If you use Spectrans in your research, please cite:

```bibtex
@software{spectrans,
  title = {spectrans: Modular Spectral Transformers in PyTorch},
  author = {Aaron Steven White},
  year = {2025},
  url = {https://github.com/aaronstevenwhite/spectrans},
  doi = {10.5281/zenodo.17171169}
}
```

For more citation formats, see the [Citation Guide](citation.md).

## License

Spectrans is released under the MIT License. See [LICENSE](https://github.com/aaronstevenwhite/spectrans/blob/main/LICENSE) for details.
