# Spectrans: Spectral Transformers in PyTorch

[![PyPI version](https://badge.fury.io/py/spectrans.svg)](https://badge.fury.io/py/spectrans)
[![Python](https://img.shields.io/pypi/pyversions/spectrans.svg)](https://pypi.org/project/spectrans/)
[![Documentation Status](https://readthedocs.org/projects/spectrans/badge/?version=latest)](https://spectrans.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/aaronstevenwhite/spectrans/actions/workflows/ci.yml/badge.svg)](https://github.com/aaronstevenwhite/spectrans/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular library for spectral transformer implementations in PyTorch. Replaces traditional attention mechanisms with Fourier transforms, wavelets, and other spectral methods.

## Features

- **Modular Design**: Mix and match components to create custom architectures
- **Multiple Spectral Methods**: FFT, DCT, DWT, Hadamard transforms, and more
- **Efficient Implementations**: Leverages PyTorch's optimized operations
- **Extensible**: Easy to add new transforms, layers, and models
- **Well-Tested**: Comprehensive test coverage with pytest
- **Type-Safe**: Full type hints with Python 3.13+ support

## Implemented Models

- **FNet**: Token mixing with Fourier transforms
- **GFNet**: Global filter networks with learnable frequency filters
- **AFNO**: Adaptive Fourier neural operators
- **WaveletTransformer**: Token mixing with wavelet transforms
- **SpectralAttentionTransformer**: Attention approximation using random Fourier features
- **LSTTransformer**: Linear spectral transform attention
- **FNOTransformer**: Fourier neural operator transformers
- **HybridTransformer**: Combine spectral and spatial attention

## Installation

**Note**: Windows is not currently supported. Please use Linux or macOS.

```bash
pip install spectrans
```

For development:
```bash
git clone https://github.com/aaronstevenwhite/spectrans.git
cd spectrans
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage with FNet

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
assert logits.shape == (2, 2)  # (batch, num_classes)

# Or with embeddings directly
embeddings = torch.randn(2, 128, 768)  # (batch, seq_len, hidden_dim)
logits = model(inputs_embeds=embeddings)
```

### Training Example

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from spectrans.models import FNet

# Setup model and optimizer
model = FNet(
    vocab_size=30000,
    hidden_dim=256,
    num_layers=6,
    max_sequence_length=128,
    num_classes=2
)
optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(3):
    # Sample batch
    input_ids = torch.randint(0, 30000, (8, 128))
    labels = torch.randint(0, 2, (8,))

    # Forward pass
    logits = model(input_ids=input_ids)
    loss = criterion(logits, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Using Different Models

```python
from spectrans.models import GFNet, AFNOModel, WaveletTransformer

# Global Filter Network with learnable frequency filters
gfnet = GFNet(
    vocab_size=30000,
    hidden_dim=512,
    num_layers=8,
    max_sequence_length=256,
    num_classes=10
)

# Adaptive Fourier Neural Operator
afno = AFNOModel(
    vocab_size=30000,
    hidden_dim=512,
    num_layers=8,
    max_sequence_length=256,
    modes_seq=32,  # Number of Fourier modes to keep
    num_classes=10
)

# Wavelet Transformer for multi-resolution analysis
wavelet_model = WaveletTransformer(
    vocab_size=30000,
    hidden_dim=512,
    num_layers=8,
    wavelet="db4",  # Daubechies-4 wavelet
    levels=3,        # 3 decomposition levels
    max_sequence_length=256,
    num_classes=10
)

# All models follow the same interface
input_ids = torch.randint(0, 30000, (4, 256))
gfnet_output = gfnet(input_ids=input_ids)        # Shape: (4, 10)
afno_output = afno(input_ids=input_ids)          # Shape: (4, 10)
wavelet_output = wavelet_model(input_ids=input_ids)  # Shape: (4, 10)
```

### Building Hybrid Models

```python
from spectrans.models import HybridTransformer, AlternatingTransformer

# Hybrid model alternates between spectral and attention
hybrid_model = HybridTransformer(
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

# Alternating transformer with different layer types
alternating_model = AlternatingTransformer(
    vocab_size=30000,
    hidden_dim=768,
    num_layers=12,
    layer1_type="fourier",  # Odd layers use Fourier
    layer2_type="attention",  # Even layers use attention
    layer1_config={"use_real_fft": True},
    layer2_config={"num_heads": 8},
    num_classes=2,
    max_sequence_length=512
)

# Forward passes
input_ids = torch.randint(0, 30000, (2, 256))
hybrid_output = hybrid_model(input_ids=input_ids)  # Shape: (2, 2)
alternating_output = alternating_model(input_ids=input_ids)  # Shape: (2, 2)
```

### Configuration-Based Model Creation

```python
from spectrans.config import ConfigBuilder, build_model_from_config
from spectrans.config.models import FNetModelConfig, GFNetModelConfig

# Load model from YAML configuration
builder = ConfigBuilder()
model = builder.build_model("examples/configs/fnet.yaml")

# Or create configurations programmatically
fnet_config = FNetModelConfig(
    hidden_dim=512,
    num_layers=10,
    sequence_length=128,
    dropout=0.1,
    vocab_size=8000,
    num_classes=3
)

# Build model from configuration
config_dict = {"model": fnet_config.model_dump()}
model = build_model_from_config(config_dict)

# Forward pass
input_ids = torch.randint(0, 8000, (2, 128))
logits = model(input_ids=input_ids)  # Shape: (2, 3)
```

### Model Evaluation and Inference

```python
from spectrans.models import FNet

# Create and evaluate model
model = FNet(
    vocab_size=5000,
    hidden_dim=256,
    num_layers=4,
    max_sequence_length=64,
    num_classes=3
)

model.eval()  # Set to evaluation mode

# Inference on test data
test_input = torch.randint(1, 5000, (10, 32))
with torch.no_grad():
    logits = model(input_ids=test_input)
    probs = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)

print(f"Predictions: {predictions.tolist()}")
print(f"Probabilities: {probs[0].tolist()}")  # First sample
```

## Running Examples

Explore the `examples/` directory for complete working examples:

```bash
# Basic FNet for text classification
python examples/basic_fnet.py

# Configuration-based model creation
python examples/config_usage.py
```

## Custom Components

```python
from spectrans.layers.mixing.base import MixingLayer
from spectrans import register_component
import torch

@register_component("mixing", "my_custom_mixing")
class MyCustomMixing(MixingLayer):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom mixing implementation
        return self.dropout(x)  # Simplified example

    @property
    def complexity(self) -> dict[str, str]:
        return {"time": "O(n)", "space": "O(1)"}

# Use in models or blocks
from spectrans.blocks import SpectralTransformerBlock
block = SpectralTransformerBlock(
    mixing_layer=MyCustomMixing(hidden_dim=768),
    hidden_dim=768
)
```

## Documentation

Full documentation available at [https://spectrans.readthedocs.io](https://spectrans.readthedocs.io)

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
