# Installation

This guide covers how to install Spectrans for both usage and development.

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

If you're planning to contribute to Spectrans:

### Install Pre-commit Hooks

```bash
pre-commit install
```

This sets up automatic code formatting and linting.

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=spectrans

# Run specific test module
pytest tests/unit/test_models/test_fnet.py -v
```

### Type Checking

```bash
mypy src/spectrans
```

### Code Formatting and Linting

```bash
# Check code style
ruff check src tests

# Format code
black src tests
isort src tests
```

## Configuration

Spectrans supports YAML-based configuration. Example configurations are included in the `examples/configs/` directory:

```bash
ls examples/configs/
# afno.yaml  fnet.yaml  gfnet.yaml  hybrid.yaml  ...
```

## GPU Support

Spectrans automatically uses CUDA if available. To verify GPU support:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

# Models will automatically use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'spectrans'**
- Ensure you've activated your virtual environment
- Try reinstalling: `pip install -e .`

**CUDA out of memory**
- Reduce batch size or sequence length
- Use gradient checkpointing: `model.enable_gradient_checkpointing()`

**Type checking errors**
- Ensure you're using Python 3.13+
- Install development dependencies: `pip install -e ".[dev]"`

**Tests failing**
- Activate virtual environment: `source venv/bin/activate`
- Ensure all dependencies installed: `pip install -e ".[dev]"`

### Getting Help

If you encounter issues:

1. Check the [documentation](https://spectrans.readthedocs.io)
2. Search [existing issues](https://github.com/aaronstevenwhite/spectrans/issues)
3. Create a new issue with:
   - Python version: `python --version`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - Spectrans version: `python -c "import spectrans; print(spectrans.__version__)"`
   - Full error traceback

## Next Steps

- Follow the [Quick Start Guide](quickstart.md) to build your first model
- Explore [Examples](../examples/) for common use cases
- Read the [API Reference](api/index.md) for detailed documentation