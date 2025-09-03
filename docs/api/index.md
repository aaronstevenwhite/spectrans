# API Reference

Welcome to the Spectrans API reference documentation. This section provides detailed information about all modules, classes, and functions in the Spectrans library.

## Core Modules

### [Transforms](transforms.md)
Spectral transform implementations including FFT, wavelet, cosine, and Hadamard transforms.

### [Kernels](kernels.md)
Kernel approximation methods for efficient attention computation.

### [Layers](layers.md)
Neural network layers including attention mechanisms, mixing layers, and operators.

### [Blocks](blocks.md)
Pre-built transformer blocks combining attention and feedforward components.

### [Models](models.md)
Complete model architectures for various tasks.

### [Utils](utils.md)
Utility functions for complex operations, padding, and initialization.

## Quick Links

- **Home**: Return to [main documentation](../index.md)

## Module Organization

The library is organized into the following main packages:

```
spectrans/
├── transforms/      # Spectral transforms (FFT, DWT, DCT, etc.)
├── kernels/        # Kernel approximation methods
├── layers/         # Neural network layers
│   ├── attention/  # Attention mechanisms
│   ├── mixing/     # Mixing layers (Fourier, wavelet, etc.)
│   └── operators/  # Neural operators (FNO, etc.)
├── blocks/         # Transformer blocks
├── models/         # Complete architectures
└── utils/          # Utility functions
```

## Mathematical Notation

Throughout the documentation, we use standard mathematical notation:

- Vectors are denoted by lowercase bold letters: **x**, **y**
- Matrices are denoted by uppercase bold letters: **W**, **K**
- Scalars are denoted by lowercase letters: *n*, *d*, *k*
- Functions and operators use calligraphic letters: 𝒢, ℱ
- Complex numbers use ℂ, real numbers use ℝ

## API Stability

The Spectrans API follows semantic versioning. Classes and functions marked as stable will maintain backward compatibility within major versions. Experimental features are clearly marked in the documentation.