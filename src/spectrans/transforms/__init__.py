"""Spectral transform implementations."""

from .base import (
    AdaptiveTransform,
    MultiResolutionTransform,
    NeuralSpectralTransform,
    OrthogonalTransform,
    SpectralTransform,
    UnitaryTransform,
)
from .cosine import DCT, DCT2D, DST, MDCT
from .fourier import FFT1D, FFT2D, RFFT, RFFT2D, SpectralPooling
from .hadamard import (
    HadamardTransform,
    HadamardTransform2D,
    SequencyHadamardTransform,
    SlantTransform,
)
from .wavelet import DWT1D, DWT2D

__all__: list[str] = [
    # Cosine transforms
    "DCT",
    "DCT2D",
    "DST",
    # Wavelet transforms
    "DWT1D",
    "DWT2D",
    # Fourier transforms
    "FFT1D",
    "FFT2D",
    "MDCT",
    "RFFT",
    "RFFT2D",
    "AdaptiveTransform",
    # Hadamard transforms
    "HadamardTransform",
    "HadamardTransform2D",
    "MultiResolutionTransform",
    "NeuralSpectralTransform",
    "OrthogonalTransform",
    "SequencyHadamardTransform",
    "SlantTransform",
    "SpectralPooling",
    # Base classes
    "SpectralTransform",
    "UnitaryTransform",
]
