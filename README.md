# Spectrans: Spectral Transformers in PyTorch

A comprehensive and modular library for spectral transformer implementations, providing efficient alternatives to traditional attention mechanisms using Fourier transforms, wavelets, and other spectral methods.

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
- **Spectral Attention**: Attention approximation using random Fourier features
- **LST**: Linear spectral transform attention
- **FNO-Transformer**: Fourier neural operator transformers
- **Wavelet Transformer**: Token mixing with wavelet transforms
- **Hybrid Models**: Combine spectral and spatial attention

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

## Quick Start

```python
import torch
from spectrans.models import FNet

# Create FNet model directly
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

## Configuration-Based Usage

```python
from spectrans.config import ConfigBuilder

# Load model from YAML configuration
builder = ConfigBuilder()
model = builder.build_model("examples/configs/fnet.yaml")

# Forward pass
input_ids = torch.randint(0, 30000, (2, 512))
logits = model(input_ids=input_ids)
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

## Acknowledgments

This library implements methods from various research papers. See [references](docs/references.md) for citations.