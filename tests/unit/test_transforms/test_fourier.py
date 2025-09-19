"""Unit tests for Fourier-based spectral transforms."""

import pytest
import torch

from spectrans.transforms import (
    FFT1D,
    FFT2D,
    RFFT,
)


class TestFFTTransforms:
    """Test FFT-based transforms."""

    def test_fft1d_forward_inverse(self, random_tensor):
        """Test FFT1D forward and inverse transforms."""
        transform = FFT1D(norm="ortho")

        # Forward transform
        freq = transform.transform(random_tensor)
        assert freq.dtype == torch.complex64 or freq.dtype == torch.complex128
        assert freq.shape == random_tensor.shape

        # Inverse transform
        reconstructed = transform.inverse_transform(freq)

        # Check reconstruction
        torch.testing.assert_close(reconstructed.real, random_tensor, rtol=1e-4, atol=1e-6)

    def test_fft2d_forward_inverse(self, random_tensor):
        """Test FFT2D forward and inverse transforms."""
        transform = FFT2D(norm="ortho")

        # Forward transform
        freq = transform.transform(random_tensor)
        assert freq.dtype == torch.complex64 or freq.dtype == torch.complex128
        assert freq.shape == random_tensor.shape

        # Inverse transform
        reconstructed = transform.inverse_transform(freq)

        # Check reconstruction
        torch.testing.assert_close(reconstructed.real, random_tensor, rtol=1e-4, atol=1e-6)

    def test_rfft_real_input(self, random_tensor):
        """Test RFFT with real input."""
        transform = RFFT(norm="ortho")

        # Forward transform
        freq = transform.transform(random_tensor)

        # Check output size (approximately half due to symmetry)
        expected_size = random_tensor.shape[-1] // 2 + 1
        assert freq.shape[-1] == expected_size

        # Inverse transform
        reconstructed = transform.inverse_transform(freq, n=random_tensor.shape[-1])

        # Check reconstruction
        torch.testing.assert_close(reconstructed, random_tensor, rtol=1e-4, atol=1e-6)

    def test_fft_parseval_theorem(self, random_tensor):
        """Test Parseval's theorem (energy conservation)."""
        transform = FFT1D(norm="ortho")

        # Compute energy in time domain
        energy_time = torch.sum(torch.abs(random_tensor) ** 2)

        # Compute energy in frequency domain
        freq = transform.transform(random_tensor)
        energy_freq = torch.sum(torch.abs(freq) ** 2)

        # Check energy conservation (Parseval's theorem)
        torch.testing.assert_close(energy_time, energy_freq, rtol=1e-4, atol=1e-6)

    def test_fft2d_separability(self, device):
        """Test that 2D FFT is separable (can be computed as two 1D FFTs)."""
        # Create a 2D tensor
        x = torch.randn(64, 64, device=device)

        # Method 1: Direct 2D FFT
        transform_2d = FFT2D(norm="ortho")
        result_2d = transform_2d.transform(x.unsqueeze(0)).squeeze(0)

        # Method 2: Two 1D FFTs
        transform_1d = FFT1D(norm="ortho")
        # FFT along first dimension
        temp = transform_1d.transform(x, dim=0)
        # FFT along second dimension
        result_1d = transform_1d.transform(temp, dim=1)

        # Results should be identical (within numerical precision)
        torch.testing.assert_close(result_2d, result_1d, rtol=1e-5, atol=1e-7)

    def test_rfft_hermitian_symmetry(self, device):
        """Test that RFFT output has Hermitian symmetry properties."""
        x = torch.randn(128, device=device)

        transform = RFFT(norm="ortho")
        freq = transform.transform(x.unsqueeze(0)).squeeze(0)

        # DC component should be real
        assert torch.abs(freq[0].imag) < 1e-7

        # Nyquist frequency should be real (if even length)
        if x.shape[0] % 2 == 0:
            assert torch.abs(freq[-1].imag) < 1e-7

    @pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
    def test_fft_normalization_modes(self, norm, device):
        """Test different normalization modes for FFT."""
        x = torch.randn(32, device=device)

        transform = FFT1D(norm=norm)
        freq = transform.transform(x)
        reconstructed = transform.inverse_transform(freq)

        # Should reconstruct perfectly regardless of normalization
        torch.testing.assert_close(reconstructed.real, x, rtol=1e-5, atol=1e-7)

    def test_fft_gradient_flow(self, device):
        """Test that gradients flow through FFT operations."""
        x = torch.randn(32, 32, requires_grad=True, device=device)

        transform = FFT1D()

        # Forward pass
        freq = transform.transform(x)
        # Create a loss (use real part for real-valued loss)
        loss = torch.sum(torch.abs(freq) ** 2)

        # Backward pass
        loss.backward()

        # Check gradient exists and is non-zero
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_fft2d_gradient_flow(self, device):
        """Test gradient flow through 2D FFT."""
        x = torch.randn(2, 32, 32, requires_grad=True, device=device)

        transform = FFT2D()

        # Forward pass
        freq = transform.transform(x)
        loss = torch.sum(torch.abs(freq) ** 2)

        # Backward pass
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_rfft_gradient_flow(self, device):
        """Test gradient flow through RFFT."""
        x = torch.randn(2, 64, requires_grad=True, device=device)

        transform = RFFT()

        # Forward and inverse
        freq = transform.transform(x)
        reconstructed = transform.inverse_transform(freq, n=x.shape[-1])
        loss = torch.sum((reconstructed - x) ** 2)

        # Backward pass
        loss.backward()

        assert x.grad is not None

    @pytest.mark.parametrize("size", [16, 32, 64, 128, 256])
    def test_fft_different_sizes(self, size, device):
        """Test FFT with different input sizes."""
        x = torch.randn(size, device=device)

        transform = FFT1D(norm="ortho")
        freq = transform.transform(x)
        reconstructed = transform.inverse_transform(freq)

        torch.testing.assert_close(reconstructed.real, x, rtol=1e-4, atol=1e-6)

    def test_fft_batch_dimensions(self, device):
        """Test FFT with various batch dimensions."""
        # Test with different batch shapes
        shapes = [
            (32,),  # 1D
            (4, 32),  # 2D with batch
            (4, 8, 32),  # 3D with batch
            (2, 4, 8, 32),  # 4D with batch
        ]

        transform = FFT1D(norm="ortho")

        for shape in shapes:
            x = torch.randn(*shape, device=device)
            freq = transform.transform(x)
            assert freq.shape == x.shape

            reconstructed = transform.inverse_transform(freq)
            torch.testing.assert_close(reconstructed.real, x, rtol=1e-4, atol=1e-6)

    def test_fft2d_batch_dimensions(self, device):
        """Test 2D FFT with batch dimensions."""
        # Create input with batch dimension
        batch_size = 4
        height, width = 32, 32
        x = torch.randn(batch_size, height, width, device=device)

        transform = FFT2D(norm="ortho")
        freq = transform.transform(x)

        assert freq.shape == x.shape

        # Test reconstruction
        reconstructed = transform.inverse_transform(freq)
        torch.testing.assert_close(reconstructed.real, x, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
