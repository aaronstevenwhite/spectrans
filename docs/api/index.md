# API Reference

Welcome to the Spectrans API reference documentation. This section provides detailed information about all modules, classes, and functions in the Spectrans library.

## Core Modules

### [Transforms](transforms/index.md)
Spectral transform implementations including FFT, wavelet, cosine, and Hadamard transforms.

### [Kernels](kernels/index.md)
Kernel approximation methods for efficient attention computation.

### [Layers](layers/index.md)
Neural network layers including attention mechanisms, mixing layers, and operators.

### [Blocks](blocks/index.md)
Pre-built transformer blocks combining attention and feedforward components.

### [Models](models/index.md)
Complete model architectures for various tasks.

### [Utils](utils/index.md)
Utility functions for complex operations, padding, and initialization.

### [Core](core/index.md)
Core interfaces, base classes, and registry system.

### [Configuration](config/index.md)
Configuration system for models and components.

## Quick Links

- **Home**: Return to [main documentation](../index.md)

## Module Organization

The library is organized into the following main packages:

```
spectrans/
├── core/           # Core interfaces and registry
│   ├── base        # Base classes
│   ├── types       # Type definitions
│   └── registry    # Component registry
├── transforms/     # Spectral transforms
│   ├── base        # Base transform classes
│   ├── fourier     # FFT implementations
│   ├── cosine      # DCT/DST transforms
│   ├── hadamard    # Hadamard transforms
│   └── wavelet     # Wavelet transforms
├── kernels/        # Kernel approximations
│   ├── base        # Base kernel classes
│   ├── rff         # Random Fourier Features
│   └── spectral    # Spectral kernels
├── layers/         # Neural network layers
│   ├── attention/  # Attention mechanisms
│   │   ├── lst     # Linear Spectral Transform
│   │   └── spectral # Spectral attention
│   ├── mixing/     # Mixing layers
│   │   ├── base    # Base mixing classes
│   │   ├── afno    # AFNO mixing
│   │   ├── fourier # Fourier mixing
│   │   ├── global_filter # Global filters
│   │   └── wavelet # Wavelet mixing
│   └── operators/  # Neural operators
│       └── fno     # Fourier Neural Operators
├── blocks/         # Transformer blocks
│   ├── base        # Base block classes
│   ├── hybrid      # Hybrid blocks
│   └── spectral    # Spectral blocks
├── models/         # Complete architectures
│   ├── base        # Base model classes
│   ├── afno        # AFNO models
│   ├── fnet        # FNet models
│   ├── gfnet       # GFNet models
│   └── ...         # Other models
├── utils/          # Utility functions
│   ├── complex     # Complex operations
│   ├── initialization # Initialization
│   └── padding     # Padding utilities
└── config/         # Configuration system
    ├── core        # Core configuration
    ├── builder     # Model builder
    ├── models      # Model configs
    └── layers/     # Layer configs
        ├── attention # Attention configs
        └── mixing   # Mixing configs
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