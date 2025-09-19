r"""Neural operator implementations for function space mappings.

Provides neural operators that learn mappings between infinite-dimensional function
spaces rather than between finite-dimensional vectors. These operators are useful for
learning solution operators for partial differential equations and other continuous
transformations.

Neural operators parameterize integral kernels in the Fourier domain, computing global
dependencies while maintaining resolution-invariant properties. Functions can be
discretized at different resolutions during training and evaluation without retraining.

Modules
-------
fno
    Fourier Neural Operator implementations.

Classes
-------
FourierNeuralOperator
    Base FNO layer implementing kernel learning in Fourier space.
SpectralConv1d
    1D spectral convolution with learnable complex weights.
SpectralConv2d
    2D spectral convolution for spatial data processing.
FNOBlock
    FNO block with normalization and feedforward components.

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

FNO block with residual connections:

>>> from spectrans.layers.operators import FNOBlock
>>> block = FNOBlock(hidden_dim=64, modes=16, mlp_ratio=2.0)
>>> processed = block(x)

Notes
-----
Mathematical Foundation:

The Fourier Neural Operator learns to approximate the solution operator
$\mathcal{G}: \mathcal{A} \rightarrow \mathcal{U}$ that maps from input function
space $\mathcal{A}$ to output function space $\mathcal{U}$.

For input function $\mathbf{v}: \Omega \rightarrow \mathbb{R}^{d_v}$, the FNO layer computes:

$$
\mathbf{v}_{l+1}(x) = \sigma\left(\mathbf{W} \mathbf{v}_l(x) + \mathcal{K}_l(\mathbf{v}_l)(x) + \mathbf{b}\right)
$$

The kernel operator $\mathcal{K}_l$ is parameterized in Fourier space:

$$
\mathcal{F}[\mathcal{K}_l(\mathbf{v})](k) = \mathbf{R}_l(k) \cdot \mathcal{F}[\mathbf{v}](k)
$$

where $\mathbf{R}_l(k) \in \mathbb{C}^{d \times d}$ are learnable complex weights and
$\mathcal{F}$ denotes the Fourier transform.

Spectral convolution applies this kernel by transforming input to Fourier domain
$\hat{\mathbf{v}} = \mathcal{F}[\mathbf{v}]$, applying learned kernel $\hat{\mathbf{u}} = \mathbf{R} \cdot \hat{\mathbf{v}}$,
and transforming back to spatial domain $\mathbf{u} = \mathcal{F}^{-1}[\hat{\mathbf{u}}]$.

Time complexity is $O(N d \log N + k d^2)$ where $k$ is number of retained modes. Space
complexity is $O(k d^2)$ for learnable parameters. Resolution invariance allows the same
weights to work for different discretizations.

Mode truncation keeping only low-frequency modes reduces computational cost from $O(N^2)$
to $O(N \log N)$, filters out high-frequency noise for generalization, and avoids overfitting
to discretization artifacts for stability.

References
----------
Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya,
Andrew Stuart, and Anima Anandkumar. 2021. Fourier neural operator for parametric partial
differential equations. In Proceedings of the International Conference on Learning
Representations (ICLR).

See Also
--------
[`spectrans.transforms.fourier`][] : Underlying FFT implementations
[`spectrans.layers.mixing.afno`][] : AFNO layers using similar principles
[`spectrans.utils.complex`][] : Complex tensor operations
"""

from .fno import FNOBlock, FourierNeuralOperator, SpectralConv1d, SpectralConv2d

# Public API - alphabetically sorted
__all__ = [
    "FNOBlock",
    "FourierNeuralOperator",
    "SpectralConv1d",
    "SpectralConv2d",
]
