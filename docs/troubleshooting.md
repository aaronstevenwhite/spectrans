# Troubleshooting Guide

This guide covers common issues and their solutions when using Spectrans.

## MKL FFT Compatibility Issues

### Problem

On certain platforms, especially in CI/CD environments like GitHub Actions, you may encounter errors related to Intel MKL (Math Kernel Library) FFT operations:

```
RuntimeError: MKL FFT error: Inconsistent configuration
```

This occurs when there's a mismatch between PyTorch's FFT implementation and the available MKL libraries on the system.

### Solution

Spectrans provides automatic fallback mechanisms for FFT operations that activate when MKL errors are detected. You can also manually force the use of fallback implementations.

#### Automatic Fallback

The library automatically detects MKL FFT errors and switches to a compatible implementation:

```python
from spectrans.transforms import FFT1D

# This will automatically use fallback if MKL errors occur
transform = FFT1D(norm="ortho")
x = torch.randn(32, 64)
y = transform.transform(x)  # Uses fallback if needed
```

#### Manual Fallback Control

To force the use of fallback implementations (useful in CI/CD):

```bash
# Set environment variable before running
export SPECTRANS_DISABLE_MKL_FFT=1
python your_script.py
```

Or within Python:

```python
import os
os.environ["SPECTRANS_DISABLE_MKL_FFT"] = "1"

# Now all FFT operations will use the fallback
import spectrans
```

### CI/CD Configuration

For GitHub Actions and other CI environments, add the environment variable to your workflow:

```yaml
- name: Run tests
  env:
    SPECTRANS_DISABLE_MKL_FFT: 1
    MKL_SERVICE_FORCE_INTEL: 1
  run: pytest tests/
```

### Performance Considerations

The fallback implementation uses DFT (Discrete Fourier Transform) matrix multiplication:

- **Computational complexity**: $O(n^2)$ instead of FFT's $O(n \log n)$
- **Memory usage**: Higher due to explicit matrix construction
- **Numerical precision**: Slightly different numerical properties (tolerances of ~5e-3 for relative error)
- **Gradient computation**: Fully supported with automatic differentiation

For most deep learning applications, the performance difference is negligible as FFT operations typically constitute a small fraction of total computation time.

### Verifying Fallback Usage

To check if the fallback is being used:

```python
import os
print(f"FFT Fallback enabled: {os.environ.get('SPECTRANS_DISABLE_MKL_FFT') == '1'}")

# The library will also emit warnings when fallback is triggered
import warnings
warnings.filterwarnings("default", category=RuntimeWarning)
```

## Numerical Precision Issues

### Problem

Tests may fail due to numerical precision differences between different FFT implementations or hardware.

### Solution

The library uses adaptive tolerances based on the FFT implementation:

```python
# For testing with different tolerances
if os.environ.get("SPECTRANS_DISABLE_MKL_FFT") == "1":
    # DFT fallback has looser tolerances
    rtol, atol = 5e-3, 5e-4
else:
    # Native FFT has tighter tolerances
    rtol, atol = 1e-5, 1e-6

torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
```

## Platform-Specific Issues

### Linux (Ubuntu/Debian)

If you encounter MKL issues on Linux:

```bash
# Install MKL libraries
sudo apt-get update
sudo apt-get install intel-mkl

# Or use the fallback
export SPECTRANS_DISABLE_MKL_FFT=1
```

### macOS

On Apple Silicon Macs, MKL is not available. The library automatically uses appropriate alternatives:

```bash
# No special configuration needed on Apple Silicon
# Fallback is automatically used when needed
```

### Windows

Windows is currently not supported. Please use Linux or macOS, or consider using WSL2 (Windows Subsystem for Linux).

## Memory Issues

### Problem

Out of memory errors when using large sequence lengths or batch sizes.

### Solution

1. **Reduce batch size**:
```python
# Instead of large batches
# output = model(torch.randn(128, 1024, 512))

# Use smaller batches
batch_size = 32
for i in range(0, 128, batch_size):
    batch = data[i:i+batch_size]
    output = model(batch)
```

2. **Enable gradient checkpointing**:
```python
from spectrans.models import FNet

model = FNet(
    hidden_dim=512,
    num_layers=12,
    gradient_checkpointing=True  # Reduces memory usage
)
```

3. **Use mixed precision training**:
```python
from torch.cuda.amp import autocast

with autocast():
    output = model(input_ids)
    loss = criterion(output, labels)
```

## Installation Issues

### Problem

Installation fails with dependency conflicts.

### Solution

1. **Use Python 3.13+**:
```bash
python --version  # Should be 3.13 or higher
```

2. **Create a clean virtual environment**:
```bash
python3.13 -m venv fresh_env
source fresh_env/bin/activate
pip install --upgrade pip
pip install spectrans
```

3. **For development installation**:
```bash
git clone https://github.com/aaronstevenwhite/spectrans.git
cd spectrans
pip install -e ".[dev]"
```

## Getting Further Help

If you continue to experience issues:

1. Check the [GitHub Issues](https://github.com/aaronstevenwhite/spectrans/issues) for similar problems
2. Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
3. Create a minimal reproducible example
4. Open a new issue with:
   - System information (OS, Python version, PyTorch version)
   - Complete error message and traceback
   - Minimal code to reproduce the issue
