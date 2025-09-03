r"""Neural operator implementations for function space mappings.

This module provides neural operators that learn mappings between infinite-dimensional
function spaces rather than between finite-dimensional vectors. These operators are
particularly effective for learning solution operators for partial differential equations
and other continuous transformations.

Neural operators parameterize integral kernels in the Fourier domain, enabling efficient
computation of global dependencies while maintaining resolution-invariant properties.
The key advantage is the ability to discretize functions at different resolutions
during training and evaluation without retraining.

Classes
-------
FourierNeuralOperator
    Base FNO layer implementing kernel learning in Fourier space.
SpectralConv1d
    1D spectral convolution with learnable complex weights.
SpectralConv2d
    2D spectral convolution for spatial data processing.
FNOBlock
    Complete FNO block with normalization and feedforward components.

Examples
--------
Basic Fourier neural operator:

>>> import torch
>>> from spectrans.layers.operators import FourierNeuralOperator
>>> fno = FourierNeuralOperator(hidden_dim=64, modes=16)
>>> x = torch.randn(32, 128, 64)
>>> output = fno(x)

Spectral convolution for 2D problems:

>>> from spectrans.layers.operators import SpectralConv2d
>>> conv2d = SpectralConv2d(in_channels=3, out_channels=64, modes=(32, 32))
>>> spatial_data = torch.randn(32, 3, 256, 256)
>>> features = conv2d(spatial_data)

Complete FNO block with residual connections:

>>> from spectrans.layers.operators import FNOBlock
>>> block = FNOBlock(hidden_dim=64, modes=16, mlp_ratio=2.0)
>>> processed = block(x)

Notes
-----
Mathematical Foundation:

The Fourier Neural Operator learns to approximate the solution operator
:math:`\mathcal{G}: \mathcal{A} \rightarrow \mathcal{U}` that maps from input function
space :math:`\mathcal{A}` to output function space :math:`\mathcal{U}`.

For input function :math:`v: \Omega \rightarrow \mathbb{R}^{d_v}`, the FNO layer computes:

.. math::
    v_{l+1}(x) = \sigma\left(\mathbf{W} v_l(x) + \mathcal{K}_l(v_l)(x) + \mathbf{b}\right)

The kernel operator :math:`\mathcal{K}_l` is parameterized in Fourier space:

.. math::
    \mathcal{F}[\mathcal{K}_l(v)](k) = \mathbf{R}_l(k) \cdot \mathcal{F}[v](k)

where :math:`\mathbf{R}_l(k) \in \mathbb{C}^{d \times d}` are learnable complex weights
and :math:`\mathcal{F}` denotes the Fourier transform.

Spectral convolution applies this kernel efficiently:

1. Transform input to Fourier domain: :math:`\hat{v} = \mathcal{F}[v]`
2. Apply learned kernel: :math:`\hat{u} = \mathbf{R} \cdot \hat{v}`
3. Transform back to spatial domain: :math:`u = \mathcal{F}^{-1}[\hat{u}]`

Computational Properties:

- Time complexity: :math:`O(N d \log N + k d^2)` where :math:`k` is number of retained modes
- Space complexity: :math:`O(k d^2)` for learnable parameters
- Resolution invariance: Same weights work for different discretizations

The mode truncation (keeping only low-frequency modes) is crucial for:

- Computational efficiency: Reduces from :math:`O(N^2)` to :math:`O(N \log N)`
- Generalization: High-frequency noise is filtered out
- Stability: Avoids overfitting to discretization artifacts

See Also
--------
spectrans.transforms.fourier : Underlying FFT implementations
spectrans.layers.mixing.afno : AFNO layers using similar principles
spectrans.utils.complex : Complex tensor operations
"""

from .fno import FNOBlock, FourierNeuralOperator, SpectralConv1d, SpectralConv2d

__all__ = [
    "FNOBlock",
    "FourierNeuralOperator",
    "SpectralConv1d",
    "SpectralConv2d",
]