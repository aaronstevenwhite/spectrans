# Installation

This guide covers how to install Spectrans for both usage and development.

## Platform Requirements

**Important**: Spectrans currently supports Linux and macOS only. Windows is not supported at this time.

## Requirements

Spectrans requires Python 3.13 or later and has the following dependencies:

- **PyTorch**: >= 2.5.0 (for neural network operations)
- **NumPy**: >= 2.0.0 (for numerical computations)
- **Pydantic**: >= 2.0.0 (for configuration management)
- **PyYAML**: >= 6.0.2 (for YAML configuration files)
- **einops**: >= 0.8.0 (for tensor operations)

## Installation Options

### Option 1: Install from PyPI (Recommended for Users)

```bash
pip install spectrans
```

This installs the latest stable version from PyPI with all required dependencies.

### Option 2: Development Installation

For contributors or users who want the latest features:

#### Step 1: Clone the Repository

```bash
git clone https://github.com/aaronstevenwhite/spectrans.git
cd spectrans
```

#### Step 2: Create Virtual Environment

We recommend using Python 3.13 with a virtual environment:

```bash
# Create virtual environment
python3.13 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

#### Step 3: Install in Development Mode

```bash
# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

This installs:

- The main package in editable mode
- Development dependencies (pytest, mypy, ruff, etc.)
- Documentation dependencies (mkdocs, etc.)

## Verify Installation

Test your installation by running:

```python
import spectrans
import torch

# Check version
print(f"Spectrans version: {spectrans.__version__}")

# Quick functionality test
from spectrans.models import FNet

model = FNet(
    vocab_size=1000,
    hidden_dim=256,
    num_layers=6,
    max_sequence_length=128,
    num_classes=2
)

# Test forward pass
input_ids = torch.randint(0, 1000, (2, 64))
output = model(input_ids=input_ids)
print(f"Output shape: {output.shape}")  # Should be (2, 2)
print("✓ Installation successful!")
```

## Development Setup

For contributors and developers, see the [Contributing Guide](contributing.md) for complete development setup instructions including pre-commit hooks, testing, and code quality tools.

## GPU Support

Spectrans automatically uses CUDA if available:

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

## Getting Help

If you encounter installation issues:

1. Ensure you're using Python 3.13+
2. Try reinstalling: `pip install --upgrade spectrans`
3. Check [existing issues](https://github.com/aaronstevenwhite/spectrans/issues)
4. For development setup, see the [Contributing Guide](contributing.md)

## Next Steps

- Follow the [Quick Start Guide](quickstart.md) to build your first model
- See the [Contributing Guide](contributing.md) for development setup
- Explore [Examples](../examples/) for common use cases
- Read the [API Reference](api/index.md) for detailed documentation
