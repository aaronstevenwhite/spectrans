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
from spectrans import create_component

# Create a model from the registry
model = create_component("model", "fnet", 
    num_layers=12,
    hidden_dim=768,
    max_seq_length=512
)

# Forward pass
x = torch.randn(2, 128, 768)  # (batch, seq_len, hidden_dim)
output = model(x)
```

## Configuration-Based Usage

```python
from spectrans.config import ConfigParser, ModelBuilder

# Load model from YAML configuration
parser = ConfigParser("configs/fnet.yaml")
model = ModelBuilder.from_config(parser.parse_model())
```

## Custom Components

```python
from spectrans import register_component, MixingLayer

@register_component("mixing", "my_custom_mixing")
class MyCustomMixing(MixingLayer):
    def forward(self, x):
        # Your implementation here
        return x
    
    @property
    def complexity(self):
        return {"time": "O(n log n)", "space": "O(n)"}
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