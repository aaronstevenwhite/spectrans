# `spectrans`: Spectral Transformers in PyTorch

[![PyPI version](https://badge.fury.io/py/spectrans.svg)](https://badge.fury.io/py/spectrans)
[![Python](https://img.shields.io/pypi/pyversions/spectrans.svg)](https://pypi.org/project/spectrans/)
[![CI](https://github.com/aaronstevenwhite/spectrans/actions/workflows/ci.yml/badge.svg)](https://github.com/aaronstevenwhite/spectrans/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/spectrans/badge/?version=latest)](https://spectrans.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular library for spectral transformer implementations in PyTorch. Replaces traditional attention mechanisms with Fourier transforms, wavelets, and other spectral methods.

## Features

- **Modular Design**: Mix and match components to create custom architectures
- **Multiple Spectral Methods**: FFT, DCT, DWT, Hadamard transforms, and more
- **Efficient**: Fast token mixing via frequency domain operations
- **Type-Safe**: Full type hints with Python 3.13+ support
- **Well-Tested**: Comprehensive test coverage
- **Easy to Use**: Consistent API across all models

## Installation

```bash
pip install spectrans
```

For development:
```bash
git clone https://github.com/aaronstevenwhite/spectrans.git
cd spectrans
pip install -e ".[dev]"
```

**Note**: Windows is not currently supported. Please use Linux or macOS.

## Quick Start

```python
import torch
from spectrans.models import FNet

# Create FNet model for classification
model = FNet(
    vocab_size=30000,
    hidden_dim=768,
    num_layers=12,
    max_sequence_length=512,
    num_classes=2
)

# Forward pass with token IDs
input_ids = torch.randint(0, 30000, (2, 128))  # (batch, seq_len)
logits = model(input_ids=input_ids)
print(f"Output shape: {logits.shape}")  # torch.Size([2, 2])

# Or with embeddings directly
embeddings = torch.randn(2, 128, 768)  # (batch, seq_len, hidden_dim)
logits = model(inputs_embeds=embeddings)
```

## Available Models

| Model | Description | Key Operation |
|-------|------------|---------------|
| [`FNet`](https://spectrans.readthedocs.io/en/latest/api/models/#spectrans.models.FNet) | Token mixing via 2D Fourier transforms | `FFT2D(tokens × features)` |
| [`GFNet`](https://spectrans.readthedocs.io/en/latest/api/models/#spectrans.models.GFNet) | Learnable frequency domain filters | `FFT → element-wise multiply → iFFT` |
| [`AFNO`](https://spectrans.readthedocs.io/en/latest/api/models/#spectrans.models.AFNOModel) | Adaptive Fourier neural operators | `FFT → keep top-k modes → MLP → iFFT` |
| [`WaveletTransformer`](https://spectrans.readthedocs.io/en/latest/api/models/#spectrans.models.WaveletTransformer) | Multi-resolution wavelet decomposition | `DWT → process scales → iDWT` |
| [`SpectralAttention`](https://spectrans.readthedocs.io/en/latest/api/models/#spectrans.models.SpectralAttentionTransformer) | Attention via random Fourier features | `φ(Q)φ(K)ᵀV` where `φ = RFF` |
| [`LSTTransformer`](https://spectrans.readthedocs.io/en/latest/api/models/#spectrans.models.LSTTransformer) | Low-rank spectral approximation | `DCT → low-rank projection → iDCT` |
| [`FNOTransformer`](https://spectrans.readthedocs.io/en/latest/api/models/#spectrans.models.FNOTransformer) | Spectral convolution operators | `FFT → spectral conv → iFFT + residual` |
| [`HybridTransformer`](https://spectrans.readthedocs.io/en/latest/api/models/#spectrans.models.HybridTransformer) | Alternating spectral and attention layers | `[Spectral, Attention, Spectral, ...]` |

## Usage Examples

### Training

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from spectrans.models import FNet

model = FNet(vocab_size=30000, hidden_dim=256, num_layers=6,
             max_sequence_length=128, num_classes=2)
optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(3):
    input_ids = torch.randint(0, 30000, (8, 128))
    labels = torch.randint(0, 2, (8,))

    logits = model(input_ids=input_ids)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Using Different Models

```python
from spectrans.models import GFNet, AFNOModel, WaveletTransformer

# Global Filter Network with learnable filters
gfnet = GFNet(vocab_size=30000, hidden_dim=512, num_layers=8,
              max_sequence_length=256, num_classes=10)

# Adaptive Fourier Neural Operator
afno = AFNOModel(vocab_size=30000, hidden_dim=512, num_layers=8,
                 max_sequence_length=256, modes_seq=32, num_classes=10)

# Wavelet Transformer
wavelet = WaveletTransformer(vocab_size=30000, hidden_dim=512,
                              num_layers=8, wavelet="db4", levels=3,
                              max_sequence_length=256, num_classes=10)

# All models share the same interface
input_ids = torch.randint(0, 30000, (4, 256))
output = gfnet(input_ids=input_ids)  # Shape: (4, 10)
```

### Hybrid Models

```python
from spectrans.models import HybridTransformer

# Alternate between spectral and attention layers
hybrid = HybridTransformer(
    vocab_size=30000,
    hidden_dim=768,
    num_layers=12,
    spectral_type="fourier",
    spatial_type="attention",
    alternation_pattern="even_spectral",  # Even layers use spectral
    num_heads=8,
    max_sequence_length=512,
    num_classes=2
)

output = hybrid(input_ids=input_ids)
```

### Configuration-Based Creation

```python
from spectrans.config import ConfigBuilder

# Load model from YAML
builder = ConfigBuilder()
model = builder.build_model("examples/configs/fnet.yaml")

# Or create programmatically
from spectrans.config.models import FNetModelConfig
from spectrans.config import build_model_from_config

config = FNetModelConfig(hidden_dim=512, num_layers=10,
                          sequence_length=128, vocab_size=8000,
                          num_classes=3)
model = build_model_from_config({"model": config.model_dump()})
```

## Custom Components

```python
import torch
from spectrans.layers.mixing.base import MixingLayer
from spectrans import register_component

@register_component("mixing", "my_custom_mixing")
class MyCustomMixing(MixingLayer):
    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Your implementation here
        return x

    def get_spectral_properties(self) -> dict[str, str | bool]:
        """Return spectral properties of this layer."""
        return {
            "transform_type": "identity",
            "preserves_energy": True,
        }

    @property
    def complexity(self) -> dict[str, str]:
        return {"time": "O(n)", "space": "O(1)"}

# Use the custom component
custom_layer = MyCustomMixing(hidden_dim=768)
x = torch.randn(2, 128, 768)
output = custom_layer(x)
```

## Documentation

- **Full Documentation**: [https://spectrans.readthedocs.io](https://spectrans.readthedocs.io)
- **Examples**: See the `examples/` directory for complete working examples
- **API Reference**: Available in the documentation

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you use Spectrans in your research, please cite:

```bibtex
@software{spectrans,
  title = {spectrans: Modular Spectral Transformers in PyTorch},
  author = {Aaron Steven White},
  year = {2025},
  url = {https://github.com/aaronstevenwhite/spectrans}
}
```

## License

See [LICENSE](LICENSE) for details.
