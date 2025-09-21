# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-09-21

Initial release of `spectrans`.

### Added
- Core spectral transformer implementations:
  - Fourier Neural Operator (FNO)
  - FNet (Fourier Transform based mixing)
  - Global Filter Network (GFNet)
  - Adaptive Fourier Neural Operator (AFNO)
  - Wavelet-based transformers
  - Linear Spectral Transform (LST)
  - Spectral Cross-Attention
- Transform implementations:
  - FFT (1D and 2D)
  - DCT (Discrete Cosine Transform)
  - DST (Discrete Sine Transform)
  - DWT (Discrete Wavelet Transform, 1D and 2D)
  - Hadamard Transform
  - Short-Time Fourier Transform (STFT)
- Kernel functions:
  - Random Fourier Features (RFF)
  - Spectral kernel implementations
- Comprehensive configuration system using Pydantic
- Component registry for dynamic model creation
- Full test suite with unit, integration, and benchmark tests
- Documentation with MkDocs and ReadTheDocs integration
- CI/CD pipeline with GitHub Actions
- Pre-commit hooks for code quality

[0.1.0]: https://github.com/yourusername/spectrans/releases/tag/v0.1.0
