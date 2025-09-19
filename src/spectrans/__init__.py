r"""Modular spectral transformer implementations in PyTorch.

Spectrans is a library for spectral transformers that provides mathematically rigorous
alternatives to traditional attention mechanisms. The library implements spectral transform
methods including Fourier Neural Networks (FNet), Global Filter Networks (GFNet), Adaptive
Fourier Neural Operators (AFNO), spectral attention mechanisms, and wavelet-based transformers.

Key features include modular component architecture with registry system, mathematical rigor
with proper complex number handling, spectral transform implementations (FFT, DCT, DWT, Hadamard),
linear and log-linear complexity alternatives to quadratic attention, and YAML-based configuration
system for easy experimentation.

Modules
-------
blocks
    Transformer blocks combining mixing/attention with feedforward networks.
config
    Configuration system for models, layers, and components.
core
    Core interfaces, base classes, and registry system.
kernels
    Kernel functions for spectral attention mechanisms.
layers
    Layer implementations including mixing, attention, and operators.
models
    Model implementations for various spectral transformers.
transforms
    Spectral transform implementations (FFT, DCT, DWT, Hadamard).
utils
    Utility functions for complex operations and initialization.

Classes
-------
AttentionLayer
    Base class for spectral attention mechanisms.
BaseModel
    Base class for spectral transformer models.
MixingLayer
    Base class for token mixing layers.
SpectralComponent
    Abstract base class for all spectral components.
TransformerBlock
    Base class for transformer blocks.

Functions
---------
create_component
    Create a registered component instance.
get_component
    Retrieve a component class from the registry.
list_components
    List available components in a category.
register_component
    Decorator for registering new components.

Attributes
----------
__version__ : str
    Current version of the spectrans library.
registry : ComponentRegistry
    Global component registry instance.

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

1. **FNet**: Uses 2D Fourier transforms for token mixing with $O(n \log n)$ complexity
2. **GFNet**: Applies learnable complex filters in frequency domain
3. **AFNO**: Implements mode-truncated Fourier operators with MLPs
4. **Spectral Attention**: Uses Random Fourier Features for kernel approximation

All transforms maintain mathematical properties such as orthogonality (DCT, Hadamard)
or unitarity (FFT) where applicable. Complex number operations are handled with
proper numerical stability and type safety.

References
----------
James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon. 2022.
FNet: Mixing tokens with Fourier transforms. In Proceedings of the 2022 Conference
of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (NAACL-HLT), pages 4296-4313, Seattle.

Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou. 2021.
Global filter networks for image classification. In Advances in Neural
Information Processing Systems 34 (NeurIPS 2021), pages 980-993.

John Guibas, Morteza Mardani, Zongyi Li, Andrew Tao, Anima Anandkumar, and
Bryan Catanzaro. 2022. Adaptive Fourier neural operators: Efficient token
mixers for transformers. In Proceedings of the International Conference on
Learning Representations (ICLR).

See Also
--------
[`spectrans.core`][] : Core interfaces and base classes.
[`spectrans.transforms`][] : Spectral transform implementations.
[`spectrans.layers`][] : Layer implementations for spectral transformers.
[`spectrans.models`][] : Model implementations.
[`spectrans.utils`][] : Utility functions for complex operations and initialization.
"""

__version__ = "0.1.0"

# Import core components
from .core.base import (
    AttentionLayer,
    BaseModel,
    SpectralComponent,
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
from .layers.mixing.base import MixingLayer

__all__ = [
    "ActivationType",
    "AttentionLayer",
    "BaseModel",
    "ComponentType",
    "ConfigDict",
    "MixingLayer",
    "ModelType",
    "NormType",
    "SpectralComponent",
    "Tensor",
    "TransformType",
    "TransformerBlock",
    "WaveletType",
    "__version__",
    "create_component",
    "get_component",
    "list_components",
    "register_component",
    "registry",
]
