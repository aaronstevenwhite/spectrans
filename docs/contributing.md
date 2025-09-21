# Contributing to Spectrans

Thank you for your interest in contributing to Spectrans! This guide covers the development setup and workflow for contributors.

## Development Setup

### Prerequisites

- Python 3.13 or higher
- Git
- A C compiler (for PyTorch compilation if building from source)

### Environment Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-username/spectrans.git
   cd spectrans
   git remote add upstream https://github.com/aaronstevenwhite/spectrans.git
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3.13 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   pre-commit install --hook-type pre-push
   ```

## Development Tools

The project uses several tools to maintain code quality:

- **pytest**: Testing framework
- **mypy**: Static type checking
- **ruff**: Fast Python linter
- **black**: Code formatter
- **isort**: Import sorting

### Code Quality Checks

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

# Pre-commit hooks
pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/benchmarks/

# Run specific test module
pytest tests/unit/test_models/test_fnet.py -v

# Run tests with verbose output
pytest tests/unit tests/integration -v

# Run with coverage
pytest tests/unit tests/integration --cov=spectrans --cov-report=html

# Run benchmarks for performance-critical changes
pytest tests/benchmarks/ -v --benchmark-only
```

### Debugging CI Failures

If tests pass locally but fail in CI, you can simulate the GitHub Actions environment using `act`:

```bash
# Install act (macOS/Linux via Homebrew)
brew install act

# Run the full CI workflow
act -W .github/workflows/ci.yml

# Run a specific job (e.g., test job)
act -j test

# Run with specific matrix configuration
act -j test --matrix os:ubuntu-latest --matrix python-version:3.13

# Run with CI environment variables (e.g., for MKL FFT issues)
act --env SPECTRANS_DISABLE_MKL_FFT=1

# Run with verbose output for debugging
act -v -j test
```

This is particularly useful for macOS developers to test Linux-specific issues before pushing.

For detailed MKL FFT troubleshooting, see the [Troubleshooting Guide](troubleshooting.md).

## Project Structure

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

## Coding Standards

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

### Documentation

All code must include proper docstrings following NumPy style.

### Component Registration

Use the registry system for new components:

```python
from spectrans.core.registry import register_component

@register_component("mixing", "my_custom_mixing")
class MyCustomMixing(MixingLayer):
    """Custom mixing layer implementation."""
    pass
```

## Testing Requirements

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

### Test Organization

- Each module must have corresponding tests in `tests/unit/`
- Use descriptive test names that explain what is being tested
- Group related tests in classes
- Include performance benchmarks for performance-critical changes

## Contribution Workflow

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

Follow the project structure and coding standards outlined above.

### 4. Update Documentation

- Update docstrings for any modified functions/classes
- Update README.md if adding new features
- Add examples if introducing new functionality
- Update CHANGELOG.md under "Unreleased" section (see below)

### 5. Changelog Updates

When making changes, update the CHANGELOG.md file:

1. After the first release, add an "Unreleased" section at the top if not already present
2. Add your changes under the appropriate category:
   - **Added** for new features
   - **Changed** for changes in existing functionality
   - **Deprecated** for soon-to-be removed features
   - **Removed** for now removed features
   - **Fixed** for any bug fixes
   - **Security** in case of vulnerabilities

Example entry:
```markdown
## [Unreleased]

### Added
- New WaveletAttention layer for multi-resolution attention mechanisms

### Fixed
- Memory leak in FFT2D transform when using CUDA
```

The changelog follows [Keep a Changelog](https://keepachangelog.com/) format.

### 6. Pre-submission Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass (`pytest`)
- [ ] Type hints are correct (`mypy src`)
- [ ] Code is properly formatted (`black src tests`)
- [ ] Imports are sorted (`isort src tests`)
- [ ] No linting errors (`ruff check src tests`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated with your changes
- [ ] Commit messages are clear and descriptive

### 7. Submit Pull Request

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

## Commit Message Format

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

## Getting Help

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the [main documentation](index.md)

Thank you for contributing to Spectrans!
