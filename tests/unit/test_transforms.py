"""Unit tests for spectral transforms."""


import pytest
import torch

from spectrans.transforms import (
    DCT,
    DST,
    DWT1D,
    FFT1D,
    FFT2D,
    RFFT,
    HadamardTransform,
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
        torch.testing.assert_close(
            reconstructed.real, random_tensor,
            rtol=1e-4, atol=1e-6
        )

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
        torch.testing.assert_close(
            reconstructed.real, random_tensor,
            rtol=1e-4, atol=1e-6
        )

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
        torch.testing.assert_close(
            reconstructed, random_tensor,
            rtol=1e-4, atol=1e-6
        )

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


class TestCosineTransforms:
    """Test DCT and DST transforms."""

    def test_dct_forward_inverse(self, random_tensor):
        """Test DCT forward and inverse transforms."""
        transform = DCT(normalized=True)

        # Forward transform
        dct_coeffs = transform.transform(random_tensor)
        assert dct_coeffs.shape == random_tensor.shape
        assert dct_coeffs.dtype == random_tensor.dtype

        # Inverse transform
        reconstructed = transform.inverse_transform(dct_coeffs)

        # Check reconstruction - DCT precision scales with matrix size due to accumulated FP errors
        # Empirically measured: ~3x matrix size factor vs machine epsilon
        n = random_tensor.shape[-1]  # Get the transform size
        machine_eps = torch.finfo(random_tensor.dtype).eps
        expected_error_factor = max(10, n * 3.0)  # Scale with matrix size, minimum 10x
        
        torch.testing.assert_close(
            reconstructed, random_tensor,
            rtol=expected_error_factor * machine_eps,
            atol=expected_error_factor * machine_eps
        )

    def test_dct_mathematical_properties(self, device):
        """Test DCT mathematical properties."""
        transform = DCT(normalized=True)
        
        # Test orthogonality: DCT matrix should be orthogonal
        n = 32
        x = torch.eye(n, device=device)
        
        # Apply DCT to identity matrix to get DCT matrix columns
        dct_matrix_cols = transform.transform(x, dim=0)
        
        # Check orthogonality: DCT^T * DCT = I
        gram_matrix = torch.matmul(dct_matrix_cols.T, dct_matrix_cols)
        identity = torch.eye(n, device=device)
        
        # Orthogonality test - matrix multiplication accumulates errors
        # For n×n matrix multiply: expect ~sqrt(n) × machine_eps error growth
        machine_eps = torch.finfo(identity.dtype).eps
        expected_error_factor = max(10, n * 0.5)  # Conservative estimate
        
        torch.testing.assert_close(
            gram_matrix, identity,
            rtol=expected_error_factor * machine_eps,
            atol=expected_error_factor * machine_eps,
            msg="DCT matrix should be orthogonal"
        )

    def test_dct_energy_conservation(self, random_tensor):
        """Test that normalized DCT conserves energy (Parseval's theorem)."""
        transform = DCT(normalized=True)
        
        # Compute energy in original domain
        energy_original = torch.sum(random_tensor ** 2)
        
        # Transform and compute energy in DCT domain
        dct_coeffs = transform.transform(random_tensor)
        energy_dct = torch.sum(dct_coeffs ** 2)
        
        # Energy should be conserved for orthogonal transform
        torch.testing.assert_close(
            energy_original, energy_dct,
            rtol=1e-5, atol=1e-7,
            msg="DCT should conserve energy (Parseval's theorem)"
        )

    def test_dst_forward_inverse(self, random_tensor):
        """Test DST forward and inverse transforms."""
        transform = DST(normalized=True)

        # Forward transform
        dst_coeffs = transform.transform(random_tensor)
        assert dst_coeffs.shape == random_tensor.shape

        # Inverse transform
        reconstructed = transform.inverse_transform(dst_coeffs)

        # Check reconstruction - DST precision scales with matrix size due to accumulated FP errors
        # Empirically measured: ~3x matrix size factor vs machine epsilon (slightly worse than DCT)
        n = random_tensor.shape[-1]  # Get the transform size
        machine_eps = torch.finfo(random_tensor.dtype).eps
        expected_error_factor = max(15, n * 3.0)  # Scale with matrix size, minimum 15x
        
        torch.testing.assert_close(
            reconstructed, random_tensor,
            rtol=expected_error_factor * machine_eps,
            atol=expected_error_factor * machine_eps
        )

    def test_dst_mathematical_properties(self, device):
        """Test DST mathematical properties."""
        transform = DST(normalized=True)
        
        # Test orthogonality: DST matrix should be orthogonal
        n = 32
        x = torch.eye(n, device=device)
        
        # Apply DST to identity matrix to get DST matrix columns
        dst_matrix_cols = transform.transform(x, dim=0)
        
        # Check orthogonality: DST^T * DST = I
        gram_matrix = torch.matmul(dst_matrix_cols.T, dst_matrix_cols)
        identity = torch.eye(n, device=device)
        
        # Orthogonality test - same error characteristics as DCT
        machine_eps = torch.finfo(identity.dtype).eps
        expected_error_factor = max(10, n * 0.5)  # Conservative estimate
        
        torch.testing.assert_close(
            gram_matrix, identity,
            rtol=expected_error_factor * machine_eps,
            atol=expected_error_factor * machine_eps,
            msg="DST matrix should be orthogonal"
        )

    def test_dst_energy_conservation(self, random_tensor):
        """Test that normalized DST conserves energy."""
        transform = DST(normalized=True)
        
        # Compute energy in original domain
        energy_original = torch.sum(random_tensor ** 2)
        
        # Transform and compute energy in DST domain
        dst_coeffs = transform.transform(random_tensor)
        energy_dst = torch.sum(dst_coeffs ** 2)
        
        # Energy should be conserved for orthogonal transform
        torch.testing.assert_close(
            energy_original, energy_dst,
            rtol=1e-5, atol=1e-7,
            msg="DST should conserve energy"
        )

    def test_dct_orthogonality(self, hidden_dim, device):
        """Test DCT orthogonality property."""
        transform = DCT(normalized=True)

        # Create identity matrix
        eye = torch.eye(hidden_dim, device=device)

        # Apply DCT then inverse DCT
        dct_coeffs = transform.transform(eye, dim=0)
        reconstructed = transform.inverse_transform(dct_coeffs, dim=0)

        # Should get back identity (relaxed tolerance for numerical precision)
        torch.testing.assert_close(reconstructed, eye, rtol=1e-3, atol=1e-4)


class TestHadamardTransforms:
    """Test Hadamard transforms."""

    @pytest.mark.parametrize("size", [2, 4, 8, 16, 32, 64])
    def test_hadamard_power_of_2(self, size, device):
        """Test Hadamard transform for different power-of-2 sizes."""
        transform = HadamardTransform(normalized=True)

        # Create random tensor with power-of-2 size
        x = torch.randn(2, size, device=device)

        # Forward transform
        h_coeffs = transform.transform(x)
        assert h_coeffs.shape == x.shape

        # Inverse transform
        reconstructed = transform.inverse_transform(h_coeffs)

        # Check reconstruction
        torch.testing.assert_close(
            reconstructed, x,
            rtol=1e-4, atol=1e-6
        )

    def test_hadamard_non_power_of_2_raises(self, device):
        """Test that non-power-of-2 sizes raise error."""
        transform = HadamardTransform()

        # Non-power-of-2 size should raise error
        x = torch.randn(2, 7, device=device)

        with pytest.raises(ValueError, match="power of 2"):
            transform.transform(x)

    def test_hadamard_orthogonality(self, device):
        """Test Hadamard transform orthogonality."""
        transform = HadamardTransform(normalized=True)

        # Create identity matrix (power of 2 size)
        eye = torch.eye(16, device=device)

        # Apply Hadamard twice (self-inverse property)
        h1 = transform.transform(eye, dim=0)
        h2 = transform.transform(h1, dim=0)

        # Should get back identity
        torch.testing.assert_close(h2, eye, rtol=1e-5, atol=1e-7)


class TestWaveletTransforms:
    """Test wavelet transforms."""

    @pytest.mark.parametrize("wavelet", ["db1"])  # Only test Haar for now
    def test_dwt1d_decompose_reconstruct(self, wavelet, random_tensor):
        """Test DWT decomposition and reconstruction."""
        transform = DWT1D(wavelet=wavelet, levels=2)

        # Decompose
        approx, details = transform.decompose(random_tensor)

        # Check we have correct number of detail levels
        assert len(details) == 2

        # Reconstruct
        reconstructed = transform.reconstruct((approx, details))

        # Check reconstruction (looser tolerance for wavelets)
        torch.testing.assert_close(
            reconstructed[..., :random_tensor.shape[-1]],
            random_tensor,
            rtol=1e-3, atol=1e-5
        )

    def test_dwt1d_energy_preservation(self, random_tensor):
        """Test energy preservation in DWT (should be exact for orthogonal wavelets)."""
        transform = DWT1D(wavelet="db1", levels=1)

        # Decompose
        approx, details = transform.decompose(random_tensor)

        # Compute energies
        energy_original = torch.sum(random_tensor ** 2)
        energy_approx = torch.sum(approx ** 2)
        energy_detail = torch.sum(details[0] ** 2)

        # Total energy should be exactly preserved for orthogonal wavelets
        energy_total = energy_approx + energy_detail
        
        torch.testing.assert_close(
            energy_total, energy_original,
            rtol=1e-5, atol=1e-7,
            msg="DWT should preserve energy exactly for orthogonal wavelets"
        )

    def test_dwt1d_perfect_reconstruction_precision(self, device):
        """Test perfect reconstruction - wavelets should achieve near machine precision."""
        transform = DWT1D(wavelet="db1", levels=1)
        
        # Create test signal with known properties
        x = torch.randn(2, 3, 16, device=device)
        
        # Decompose and reconstruct
        approx, details = transform.decompose(x)
        reconstructed = transform.reconstruct((approx, details))
        
        # DWT should achieve ~2x machine epsilon precision (measured empirically)
        # This is the theoretical limit for this algorithm due to accumulated floating point errors
        machine_eps = torch.finfo(x.dtype).eps
        torch.testing.assert_close(
            reconstructed[..., :x.shape[-1]], x,
            rtol=5 * machine_eps,  # 5x machine epsilon relative tolerance
            atol=5 * machine_eps,  # 5x machine epsilon absolute tolerance  
            msg="DWT should achieve near machine precision perfect reconstruction"
        )

    def test_dwt1d_filter_bank_conditions(self):
        """Test that wavelet filter banks satisfy perfect reconstruction conditions."""
        transform = DWT1D(wavelet="db1")
        
        # For perfect reconstruction: g0(n) = h0(-n), g1(n) = h1(-n)
        h0 = transform.h0
        h1 = transform.h1
        g0 = transform.g0
        g1 = transform.g1
        
        # Check filter relationships
        torch.testing.assert_close(
            g0, torch.flip(h0, dims=[0]),
            rtol=1e-15, atol=1e-15,
            msg="g0 should be h0 time-reversed"
        )
        
        torch.testing.assert_close(
            g1, torch.flip(h1, dims=[0]),
            rtol=1e-15, atol=1e-15,
            msg="g1 should be h1 time-reversed"
        )

    def test_dwt2d_basic(self, device):
        """Test 2D DWT basic functionality."""
        from spectrans.transforms import DWT2D

        transform = DWT2D(wavelet="db1", levels=1, mode="zero")

        # Create 2D signal
        x = torch.randn(1, 64, 64, device=device)

        # Decompose
        ll, details = transform.decompose(x)

        # Check shapes
        assert ll.shape == (1, 32, 32)  # Downsampled by 2
        assert len(details) == 1
        lh, hl, hh = details[0]
        assert lh.shape == (1, 32, 32)
        assert hl.shape == (1, 32, 32)
        assert hh.shape == (1, 32, 32)

        # Reconstruct
        reconstructed = transform.reconstruct((ll, details))

        # Check reconstruction (approximate due to boundary effects)
        torch.testing.assert_close(
            reconstructed[..., :x.shape[-2], :x.shape[-1]],
            x,
            rtol=1e-2, atol=1e-4
        )


class TestTransformProperties:
    """Test general transform properties."""

    def test_transform_registration(self):
        """Test that transforms are properly registered."""
        from spectrans.core.registry import registry

        # Check FFT transforms are registered
        assert ("transform", "fft1d") in registry
        assert ("transform", "fft2d") in registry
        assert ("transform", "rfft") in registry

        # Check DCT/DST transforms are registered
        assert ("transform", "dct") in registry
        assert ("transform", "dst") in registry

        # Check Hadamard transforms are registered
        assert ("transform", "hadamard") in registry

        # Check wavelet transforms are registered
        assert ("transform", "dwt") in registry

    def test_transform_properties(self):
        """Test transform property flags."""
        # FFT is unitary
        fft = FFT1D()
        assert fft.is_unitary
        assert not fft.is_orthogonal  # Complex-valued

        # DCT is orthogonal
        dct = DCT()
        assert dct.is_orthogonal
        assert not dct.is_unitary  # Real-valued

        # Hadamard is orthogonal
        hadamard = HadamardTransform()
        assert hadamard.is_orthogonal
        assert not hadamard.is_unitary

    @pytest.mark.parametrize("transform_cls", [FFT1D, RFFT, DCT, DST, HadamardTransform])
    def test_transform_deterministic(self, transform_cls, random_tensor):
        """Test that transforms are deterministic."""
        transform = transform_cls()

        # Apply transform twice
        result1 = transform.transform(random_tensor)
        result2 = transform.transform(random_tensor)

        # Results should be identical
        torch.testing.assert_close(result1, result2, rtol=0, atol=0)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_tensor(self, device):
        """Test handling of empty tensors."""
        transform = FFT1D()

        # Empty tensor - FFT should handle it or raise error
        x = torch.empty(0, device=device)
        # PyTorch FFT raises RuntimeError for empty tensors
        with pytest.raises(RuntimeError):
            transform.transform(x)

    def test_single_element(self, device):
        """Test single-element tensors."""
        transform = DCT()

        x = torch.tensor([5.0], device=device)
        coeffs = transform.transform(x)
        reconstructed = transform.inverse_transform(coeffs)

        torch.testing.assert_close(reconstructed, x)

    def test_batch_processing(self, batch_size, sequence_length, hidden_dim, device):
        """Test batch processing of transforms."""
        transform = FFT1D()

        # Create batched input
        x = torch.randn(batch_size, sequence_length, hidden_dim, device=device)

        # Apply transform along last dimension
        freq = transform.transform(x, dim=-1)
        assert freq.shape == x.shape

        # Apply along middle dimension
        freq = transform.transform(x, dim=1)
        assert freq.shape == x.shape

        # Verify different batch elements are processed independently
        x_single = x[0:1]
        freq_single = transform.transform(x_single, dim=1)  # Same dimension as above
        torch.testing.assert_close(freq[0:1], freq_single, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
