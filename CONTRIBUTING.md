# Contributing to Spectrans

Thank you for your interest in contributing to Spectrans! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Process](#contribution-process)
- [Code Standards](#code-standards)
- [Documentation Guidelines](#documentation-guidelines)
- [Testing Requirements](#testing-requirements)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on constructive criticism and helpful feedback
- Respect differing viewpoints and experiences

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/spectrans.git
   cd spectrans
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/aaronstevenwhite/spectrans.git
   ```

## Development Setup

### Prerequisites

- Python 3.13 or higher
- Git
- A C compiler (for PyTorch compilation if building from source)

### Environment Setup

1. **Create and activate a virtual environment**:
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install the package in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   pre-commit install --hook-type pre-push
   ```

### Development Tools

The project uses several tools to maintain code quality:

- **pytest**: Testing framework
- **mypy**: Static type checking
- **ruff**: Fast Python linter
- **black**: Code formatter
- **isort**: Import sorting

### Running Code Quality Checks

Run all checks before committing:
```bash
# Format code and fix imports
ruff check --fix src tests
ruff format src tests

# Check linting, imports, and formatting
ruff check src tests
ruff format --check src tests

# Type checking (main source only)
mypy src

# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/benchmarks/

# Run tests with verbose output
pytest tests/unit tests/integration -v

# Test coverage
pytest tests/unit tests/integration --cov=spectrans --cov-report=html

# Pre-commit hooks
pre-commit run --all-files
```

### Debugging CI Failures

If tests pass locally but fail in CI, you can simulate the GitHub Actions environment using `act`:

#### Installing act

[act](https://github.com/nektos/act) allows you to run GitHub Actions workflows locally in Docker containers that closely match GitHub's hosted runners.

```bash
# macOS/Linux via Homebrew
brew install act

# Other installation methods available at https://github.com/nektos/act
```

#### Basic Usage

```bash
# Run default push event workflows
act

# Run the full CI workflow
act -W .github/workflows/ci.yml

# Run a specific job (e.g., test job)
act -j test

# Run with specific event
act pull_request

# List all workflows and jobs
act -l

# Run with specific matrix configuration
act -j test --matrix os:ubuntu-latest --matrix python-version:3.13

# Run with CI environment variables (e.g., for MKL FFT issues)
act --env SPECTRANS_DISABLE_MKL_FFT=1
```

#### Advanced Debugging

When CI tests fail but pass locally (especially MKL FFT errors on Ubuntu):

```bash
# Run the exact test job that's failing
act -j test --matrix os:ubuntu-latest

# Run with verbose output for debugging
act -v -j test

# Run with secrets from .env file (create .env.local, don't commit!)
act --secret-file .env.local

# Run specific workflow with specific Python version
act -W .github/workflows/ci.yml -j test --matrix python-version:3.13

# Combine environment variables and matrix selection
act -j test --matrix os:ubuntu-latest --env SPECTRANS_DISABLE_MKL_FFT=1
```

**Note for macOS users**: `act` is particularly useful as it simulates the Linux environment used in GitHub Actions, helping catch platform-specific issues (like MKL FFT compatibility) before pushing.

## Contribution Process

### 1. Find or Create an Issue

- Check existing issues for something you'd like to work on
- If proposing a new feature, create an issue first to discuss it
- Comment on the issue to indicate you're working on it

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Make Your Changes

Follow the project structure and coding standards (see below).

### 4. Test Your Changes

```bash
# Run specific test module
pytest tests/unit/test_transforms.py -v

# Run all tests with coverage
pytest --cov=spectrans --cov-report=html

# Run benchmarks if performance-related
pytest tests/benchmarks/ -v
```

### 5. Update Documentation

- Update docstrings for any modified functions/classes
- Update README.md if adding new features
- Add examples if introducing new functionality

## Code Standards

### Project Structure

Follow the established module organization:

```
src/spectrans/
├── core/          # Core interfaces and base classes
├── transforms/    # Spectral transform implementations
├── kernels/       # Kernel functions and features
├── layers/        # Neural network layers
│   ├── mixing/    # Mixing layers (Fourier, wavelet, etc.)
│   ├── attention/ # Attention mechanisms
│   └── operators/ # Neural operators
├── blocks/        # Transformer blocks
├── models/        # Complete model implementations
├── config/        # Configuration system
└── utils/         # Utility functions
```

### Type Hints

Use Python 3.13+ type hint conventions:

```python
# Use | for unions
def process(data: torch.Tensor | None) -> torch.Tensor: ...

# Use type statement for aliases
type TensorPair = tuple[torch.Tensor, torch.Tensor]

# Import from collections.abc
from collections.abc import Sequence, Mapping
```

### Class Design

```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class SpectralComponent(nn.Module, ABC):
    """Base class for spectral components.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the model.
    dropout : float, optional
        Dropout rate, by default 0.0.

    Attributes
    ----------
    hidden_dim : int
        Hidden dimension of the model.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of same shape as input.
        """
        pass

    @property
    @abstractmethod
    def complexity(self) -> dict[str, str]:
        """Computational complexity information.

        Returns
        -------
        dict[str, str]
            Dictionary with 'time' and 'space' complexity.
        """
        pass
```

### Component Registration

Use the registry system for new components:

```python
from spectrans.core.registry import register_component

@register_component("mixing", "my_custom_mixing")
class MyCustomMixing(MixingLayer):
    """Custom mixing layer implementation."""
    pass
```

## Documentation Guidelines

### Docstring Format

Use NumPy-style docstrings:

```python
def spectral_transform(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Apply spectral transform to input tensor.

    Computes the Discrete Fourier Transform along the specified dimension,
    applying the transform according to the formula:

    $$X[k] = \sum_{n=0}^{N-1} x[n] \exp\left(-2\pi i \frac{kn}{N}\right)$$

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to transform.
    dim : int, optional
        Dimension along which to apply transform, by default -1.

    Returns
    -------
    torch.Tensor
        Transformed tensor with same shape as input.

    Notes
    -----
    This implementation uses PyTorch's optimized FFT operations which
    leverage CUDA acceleration when available. The complexity is
    $O(n \log n)$ where $n$ is the size along the transform dimension.

    References
    ----------
    James W. Cooley and John W. Tukey. 1965. An algorithm for the machine
    calculation of complex Fourier series. Mathematics of Computation,
    19(90):297-301.

    Examples
    --------
    >>> x = torch.randn(32, 128, 512)
    >>> x_freq = spectral_transform(x, dim=1)
    >>> assert x_freq.shape == x.shape
    """
    pass
```

### Mathematical Notation

- Use LaTeX format in raw strings (r""")
- Use $ for inline math: `$O(n \log n)$`
- Use $$ for display equations
- Never use Unicode symbols (use `$\alpha$` not α)

### Module Documentation

Include comprehensive module docstrings:

```python
r"""Fourier transform implementations for spectral layers.

This module provides efficient implementations of various Fourier transforms
used in spectral transformer architectures. All transforms maintain gradient
flow and support both forward and inverse operations.

Classes
-------
FFT1D
    1D Fast Fourier Transform with gradient support.
FFT2D
    2D Fast Fourier Transform for image-like data.
RealFFT1D
    Real-valued FFT for improved efficiency.

Functions
---------
apply_fourier_mixing(x)
    Apply Fourier mixing to sequence.

See Also
--------
spectrans.transforms.wavelet : Wavelet transform implementations.
spectrans.layers.mixing.fourier : Fourier-based mixing layers.
"""
```

## Testing Requirements

### Test Organization

- Each module must have corresponding tests in `tests/unit/`
- Use descriptive test names that explain what is being tested
- Group related tests in classes

### Test Coverage

All contributions must include tests:

```python
import pytest
import torch
from spectrans.transforms import FFT1D

class TestFFT1D:
    """Test suite for 1D FFT transform."""

    def test_forward_inverse_reconstruction(self):
        """Test perfect reconstruction property."""
        x = torch.randn(32, 128, 512)
        transform = FFT1D()

        x_freq = transform.transform(x)
        x_recon = transform.inverse_transform(x_freq)

        # Use appropriate tolerance for floating point
        torch.testing.assert_close(x, x_recon, rtol=1e-5, atol=1e-7)

    def test_parseval_theorem(self):
        """Test energy conservation (Parseval's theorem)."""
        x = torch.randn(16, 64, 256)
        transform = FFT1D()

        energy_spatial = torch.sum(x ** 2)
        x_freq = transform.transform(x)
        energy_freq = torch.sum(torch.abs(x_freq) ** 2) / x.shape[-1]

        torch.testing.assert_close(energy_spatial, energy_freq, rtol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow through transform."""
        x = torch.randn(8, 32, 128, requires_grad=True)
        transform = FFT1D()

        y = transform.transform(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.any(torch.isnan(x.grad))
```

### Performance Benchmarks

For performance-critical changes:

```python
import pytest
from spectrans.layers.mixing import FourierMixing

@pytest.mark.benchmark
def test_fourier_mixing_performance(benchmark):
    """Benchmark Fourier mixing layer."""
    layer = FourierMixing(hidden_dim=768)
    x = torch.randn(32, 512, 768)

    # Run benchmark
    result = benchmark(layer, x)

    # Verify output shape
    assert result.shape == x.shape
```

## Submitting Changes

### Pre-submission Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass (`pytest`)
- [ ] Type hints are correct (`mypy src`)
- [ ] Code is properly formatted (`black src tests`)
- [ ] Imports are sorted (`isort src tests`)
- [ ] No linting errors (`ruff check src tests`)
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive

### Commit Messages

Follow conventional commit format:

```
type(scope): description

Longer explanation if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### Pull Request Process

1. **Update your branch** with latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request** on GitHub with:
   - Clear title describing the change
   - Reference to related issue(s)
   - Description of changes made
   - Any breaking changes noted
   - Test results or benchmarks if relevant

4. **Address Review Comments**:
   - Make requested changes
   - Push new commits (don't force-push during review)
   - Re-request review when ready

5. **After Approval**:
   - Squash commits if requested
   - Ensure CI passes
   - Maintainer will merge

## Getting Help

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the [documentation](https://spectrans.readthedocs.io)

## Recognition

Contributors will be recognized in:
- Release notes for significant contributions
- Special thanks in documentation

Thank you for contributing to Spectrans!
