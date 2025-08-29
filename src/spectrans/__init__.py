"""Modular spectral transformer implementations in PyTorch.

Spectrans is a comprehensive library for spectral transformers that provides efficient
alternatives to traditional attention mechanisms. The library implements state-of-the-art
spectral transform methods including Fourier Neural Networks (FNet), Global Filter Networks
(GFNet), Adaptive Fourier Neural Operators (AFNO), spectral attention mechanisms, and
wavelet-based transformers.

Key features include:
- Modular component architecture with registry system
- Mathematical rigor with proper complex number handling
- Comprehensive spectral transform implementations (FFT, DCT, DWT, Hadamard)
- Memory-efficient alternatives to quadratic attention
- YAML-based configuration system for easy experimentation

Attributes
----------
__version__ : str
    Current version of the spectrans library.

Functions
---------
create_component(category, name, **kwargs)
    Create a registered component instance.
get_component(category, name)
    Retrieve a component class from the registry.
list_components(category)
    List available components in a category.
register_component(category, name)
    Decorator for registering new components.

Classes
-------
SpectralComponent
    Abstract base class for all spectral components.
SpectralTransform
    Interface for spectral transform operations.
MixingLayer
    Base class for token mixing layers.
AttentionLayer
    Base class for spectral attention mechanisms.
TransformerBlock
    Base class for complete transformer blocks.
BaseModel
    Base class for spectral transformer models.

Examples
--------
Basic usage with the component registry:

>>> import spectrans
>>> # List available transforms
>>> spectrans.list_components('transform')
['fourier', 'cosine', 'hadamard', 'wavelet']

>>> # Create and use a Fourier transform
>>> fft = spectrans.create_component('transform', 'fourier')
>>> output = fft.transform(input_tensor)

Working with base classes:

>>> from spectrans import SpectralComponent, MixingLayer
>>> # Use base classes for custom implementations
>>> class CustomMixing(MixingLayer):
...     def forward(self, x):
...         return self.custom_transform(x)

Notes
-----
The library implements the mathematical formulations from several key papers:

1. **FNet**: Uses 2D Fourier transforms for token mixing with O(n log n) complexity
2. **GFNet**: Applies learnable complex filters in frequency domain  
3. **AFNO**: Implements mode-truncated Fourier operators with MLPs
4. **Spectral Attention**: Uses Random Fourier Features for kernel approximation

All transforms maintain mathematical properties such as orthogonality (DCT, Hadamard)
or unitarity (FFT) where applicable. Complex number operations are handled with
proper numerical stability and type safety.

See Also
--------
spectrans.core : Core interfaces and base classes
spectrans.transforms : Spectral transform implementations
spectrans.utils : Utility functions for complex operations and initialization
"""

__version__ = "0.1.0"

# Import core components
from .core.base import (
    AttentionLayer,
    BaseModel,
    MixingLayer,
    SpectralComponent,
    SpectralTransform,
    TransformerBlock,
)
from .core.registry import (
    create_component,
    get_component,
    list_components,
    register_component,
    registry,
)
from .core.types import (
    ActivationType,
    ComponentType,
    ConfigDict,
    ModelType,
    NormType,
    Tensor,
    TransformType,
    WaveletType,
)

# Public API
__all__: list[str] = [
    "ActivationType",
    "AttentionLayer",
    "BaseModel",
    "ComponentType",
    "ConfigDict",
    "MixingLayer",
    "ModelType",
    "NormType",
    # Base classes
    "SpectralComponent",
    "SpectralTransform",
    # Type exports
    "Tensor",
    "TransformType",
    "TransformerBlock",
    "WaveletType",
    # Version
    "__version__",
    "create_component",
    "get_component",
    "list_components",
    "register_component",
    # Registry functions
    "registry",
]
