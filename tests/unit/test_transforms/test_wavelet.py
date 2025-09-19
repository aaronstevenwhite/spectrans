"""Comprehensive tests for PyWavelets-compatible wavelet transforms.

Tests all wavelets for perfect reconstruction and compatibility with PyWavelets.
"""

import numpy as np
import pytest
import pywt
import torch

from spectrans.transforms.wavelet import DWT1D, DWT2D, get_wavelet_filters

# Test wavelets from different families
TEST_WAVELETS = [
    "db1",
    "db2",
    "db4",
    "db8",  # Daubechies
    "sym2",
    "sym4",
    "sym8",  # Symlets
    "coif1",
    "coif2",  # Coiflets
]


class TestWaveletFilters:
    """Test filter extraction from PyWavelets."""

    @pytest.mark.parametrize("wavelet", TEST_WAVELETS)
    def test_filter_extraction(self, wavelet):
        """Test that filters are correctly extracted."""
        dec_lo, dec_hi, rec_lo, rec_hi = get_wavelet_filters(wavelet)

        # Check filters are tensors
        assert isinstance(dec_lo, torch.Tensor)
        assert isinstance(dec_hi, torch.Tensor)
        assert isinstance(rec_lo, torch.Tensor)
        assert isinstance(rec_hi, torch.Tensor)

        # Check filter lengths match
        pywt_wavelet = pywt.Wavelet(wavelet)
        assert len(dec_lo) == len(pywt_wavelet.dec_lo)
        assert len(dec_hi) == len(pywt_wavelet.dec_hi)
        assert len(rec_lo) == len(pywt_wavelet.rec_lo)
        assert len(rec_hi) == len(pywt_wavelet.rec_hi)

        # Check values match (relaxed tolerance due to float64 -> float32 in transform)
        np.testing.assert_array_almost_equal(dec_lo.numpy(), pywt_wavelet.dec_lo, decimal=7)
        np.testing.assert_array_almost_equal(dec_hi.numpy(), pywt_wavelet.dec_hi, decimal=7)


class TestDWT1D:
    """Test 1D Discrete Wavelet Transform."""

    @pytest.mark.parametrize("wavelet", TEST_WAVELETS)
    def test_single_level_dwt(self, wavelet):
        """Test single-level DWT against PyWavelets."""
        torch.manual_seed(42)
        x = torch.randn(256)

        # Our implementation
        dwt = DWT1D(wavelet=wavelet, levels=1)
        cA, cD_list = dwt.decompose(x.unsqueeze(0))
        cA = cA.squeeze(0)
        cD = cD_list[0].squeeze(0)

        # PyWavelets reference
        cA_ref, cD_ref = pywt.dwt(x.numpy(), wavelet, mode="symmetric")
        cA_ref = torch.from_numpy(cA_ref).float()
        cD_ref = torch.from_numpy(cD_ref).float()

        # Check shapes match
        assert cA.shape == cA_ref.shape, f"cA shape mismatch: {cA.shape} vs {cA_ref.shape}"
        assert cD.shape == cD_ref.shape, f"cD shape mismatch: {cD.shape} vs {cD_ref.shape}"

        # Check values match (relaxed tolerance for numerical differences)
        torch.testing.assert_close(cA, cA_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(cD, cD_ref, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("wavelet", TEST_WAVELETS)
    def test_perfect_reconstruction(self, wavelet):
        """Test perfect reconstruction for single-level DWT."""
        torch.manual_seed(42)
        x = torch.randn(256)

        # Forward and inverse transform
        dwt = DWT1D(wavelet=wavelet, levels=1)
        coeffs = dwt.decompose(x.unsqueeze(0))
        x_rec = dwt.reconstruct(coeffs)
        x_rec = x_rec.squeeze(0)

        # Check reconstruction error
        error = torch.max(torch.abs(x - x_rec))
        assert error < 1e-6, f"Reconstruction error too large: {error}"

        # Also check RMS error
        rms_error = torch.sqrt(torch.mean((x - x_rec) ** 2))
        assert rms_error < 1e-7, f"RMS reconstruction error too large: {rms_error}"

    @pytest.mark.parametrize("wavelet", ["db2", "db4", "sym2"])
    @pytest.mark.parametrize("levels", [1, 2, 3])
    def test_multi_level_dwt(self, wavelet, levels):
        """Test multi-level DWT decomposition."""
        torch.manual_seed(42)
        x = torch.randn(256)

        # Our implementation
        dwt = DWT1D(wavelet=wavelet, levels=levels)
        cA, cD_list = dwt.decompose(x.unsqueeze(0))

        # Check we have correct number of detail levels
        assert len(cD_list) == levels

        # Check shapes are decreasing
        prev_len = x.shape[0]
        for i, cD in enumerate(cD_list):
            assert cD.shape[1] < prev_len, f"Level {i} detail should be smaller"
            prev_len = cD.shape[1]

        # Test reconstruction
        x_rec = dwt.reconstruct((cA, cD_list))
        x_rec = x_rec.squeeze(0)

        error = torch.max(torch.abs(x - x_rec))
        assert error < 1e-5, f"Multi-level reconstruction error too large: {error}"

    def test_gradient_flow(self):
        """Test that gradients flow through the transform."""
        torch.manual_seed(42)
        x = torch.randn(64, requires_grad=True)

        dwt = DWT1D(wavelet="db4", levels=2)

        # Forward pass
        cA, cD_list = dwt.decompose(x.unsqueeze(0))

        # Create a loss
        loss = cA.sum() + sum(cD.sum() for cD in cD_list)

        # Backward pass
        loss.backward()

        # Check gradient exists and is non-zero
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_batch_processing(self):
        """Test batched DWT processing."""
        torch.manual_seed(42)
        batch_size = 8
        signal_len = 128

        x = torch.randn(batch_size, signal_len)

        dwt = DWT1D(wavelet="db2", levels=2)
        cA, cD_list = dwt.decompose(x)

        # Check batch dimension preserved
        assert cA.shape[0] == batch_size
        for cD in cD_list:
            assert cD.shape[0] == batch_size

        # Test reconstruction
        x_rec = dwt.reconstruct((cA, cD_list))
        assert x_rec.shape == x.shape

        # Check reconstruction quality
        error = torch.max(torch.abs(x - x_rec))
        assert error < 1e-5


class TestDWT2D:
    """Test 2D Discrete Wavelet Transform."""

    @pytest.mark.parametrize("wavelet", ["db1", "db2", "sym2"])
    def test_single_level_dwt2d(self, wavelet):
        """Test single-level 2D DWT against PyWavelets."""
        torch.manual_seed(42)
        x = torch.randn(64, 64)

        # Our implementation
        dwt2d = DWT2D(wavelet=wavelet, levels=1)
        ll, bands = dwt2d.decompose(x.unsqueeze(0))
        ll = ll.squeeze(0)
        # bands[0] contains (HL, LH, HH) following PyWavelets convention
        hl, lh, hh = bands[0]
        hl = hl.squeeze(0)
        lh = lh.squeeze(0)
        hh = hh.squeeze(0)

        # PyWavelets reference
        coeffs_ref = pywt.dwt2(x.numpy(), wavelet, mode="symmetric")
        # PyWavelets returns (cA, (cH, cV, cD)) where:
        # cH = horizontal detail (HL), cV = vertical detail (LH), cD = diagonal (HH)
        ll_ref, (hl_ref, lh_ref, hh_ref) = coeffs_ref
        ll_ref = torch.from_numpy(ll_ref).float()
        hl_ref = torch.from_numpy(hl_ref).float()
        lh_ref = torch.from_numpy(lh_ref).float()
        hh_ref = torch.from_numpy(hh_ref).float()

        # Check shapes
        assert ll.shape == ll_ref.shape
        assert hl.shape == hl_ref.shape
        assert lh.shape == lh_ref.shape
        assert hh.shape == hh_ref.shape

        # Check values (relaxed tolerance for numerical stability)
        torch.testing.assert_close(ll, ll_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(hl, hl_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(lh, lh_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(hh, hh_ref, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("wavelet", TEST_WAVELETS[:3])  # Test fewer for 2D
    def test_perfect_reconstruction_2d(self, wavelet):
        """Test perfect reconstruction for 2D DWT."""
        torch.manual_seed(42)
        x = torch.randn(64, 64)

        # Forward and inverse transform
        dwt2d = DWT2D(wavelet=wavelet, levels=1)
        coeffs = dwt2d.decompose(x.unsqueeze(0))
        x_rec = dwt2d.reconstruct(coeffs)
        x_rec = x_rec.squeeze(0)

        # Check reconstruction
        error = torch.max(torch.abs(x - x_rec))
        assert error < 1e-5, f"2D reconstruction error too large: {error}"

    def test_multi_level_2d(self):
        """Test multi-level 2D DWT."""
        torch.manual_seed(42)
        x = torch.randn(128, 128)

        dwt2d = DWT2D(wavelet="db2", levels=3)
        ll, bands = dwt2d.decompose(x.unsqueeze(0))

        # Check we have 3 levels of detail bands
        assert len(bands) == 3

        # Test reconstruction
        x_rec = dwt2d.reconstruct((ll, bands))
        x_rec = x_rec.squeeze(0)

        error = torch.max(torch.abs(x - x_rec))
        assert error < 1e-4, f"Multi-level 2D reconstruction error: {error}"

    def test_batch_processing_2d(self):
        """Test batched 2D DWT processing."""
        torch.manual_seed(42)
        batch_size = 4
        height, width = 64, 64

        x = torch.randn(batch_size, height, width)

        dwt2d = DWT2D(wavelet="sym2", levels=2)
        ll, bands = dwt2d.decompose(x)

        # Check batch dimension
        assert ll.shape[0] == batch_size
        for lh, hl, hh in bands:
            assert lh.shape[0] == batch_size
            assert hl.shape[0] == batch_size
            assert hh.shape[0] == batch_size

        # Test reconstruction
        x_rec = dwt2d.reconstruct((ll, bands))
        assert x_rec.shape == x.shape

    def test_gradient_flow_2d(self):
        """Test that gradients flow through 2D transforms."""
        torch.manual_seed(42)
        x = torch.randn(2, 64, 64, requires_grad=True)

        dwt2d = DWT2D(wavelet="db2", levels=2)

        # Forward pass through decomposition
        ll, bands = dwt2d.decompose(x)

        # Create a loss using all coefficients
        loss = ll.sum()
        for hl, lh, hh in bands:
            loss = loss + hl.sum() + lh.sum() + hh.sum()

        # Backward pass
        loss.backward()

        # Check gradient exists and is non-zero
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

        # Test gradient through reconstruction
        x2 = torch.randn(2, 64, 64, requires_grad=True)
        ll2, bands2 = dwt2d.decompose(x2)
        x_rec = dwt2d.reconstruct((ll2, bands2))

        loss2 = x_rec.sum()
        loss2.backward()

        assert x2.grad is not None
        assert not torch.allclose(x2.grad, torch.zeros_like(x2.grad))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_power_of_two_signals(self):
        """Test with power-of-2 length signals."""
        for length in [32, 64, 128, 256, 512]:
            x = torch.randn(length)
            dwt = DWT1D(wavelet="db2", levels=2)
            coeffs = dwt.decompose(x.unsqueeze(0))
            x_rec = dwt.reconstruct(coeffs).squeeze(0)

            error = torch.max(torch.abs(x - x_rec))
            assert error < 1e-5

    def test_non_power_of_two_signals(self):
        """Test with arbitrary length signals."""
        for length in [100, 150, 200, 333]:
            x = torch.randn(length)
            dwt = DWT1D(wavelet="db4", levels=1)
            coeffs = dwt.decompose(x.unsqueeze(0))
            # Pass original length to handle edge cases with odd-length signals
            x_rec = dwt.reconstruct(coeffs, output_len=length).squeeze(0)

            error = torch.max(torch.abs(x - x_rec))
            assert error < 1e-5

    def test_small_signals(self):
        """Test with very small signals."""
        # Minimum signal length depends on wavelet filter length
        x = torch.randn(16)  # Small but valid for most wavelets
        dwt = DWT1D(wavelet="db1", levels=1)  # db1 has shortest filter
        coeffs = dwt.decompose(x.unsqueeze(0))
        x_rec = dwt.reconstruct(coeffs).squeeze(0)

        error = torch.max(torch.abs(x - x_rec))
        assert error < 1e-5

    def test_invalid_wavelet(self):
        """Test error handling for invalid wavelet names."""
        with pytest.raises(ValueError, match="Unsupported wavelet"):
            DWT1D(wavelet="invalid_wavelet")

    @pytest.mark.parametrize("device_name", ["cpu", "cuda"])
    def test_device_compatibility(self, device_name):
        """Test that wavelet transforms work on different devices."""
        if device_name == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device(device_name)
        x = torch.randn(128, device=device)

        dwt = DWT1D(wavelet="db4", levels=2)
        cA, cD_list = dwt.decompose(x.unsqueeze(0))

        # Check all outputs are on correct device
        assert cA.device.type == device.type
        for cD in cD_list:
            assert cD.device.type == device.type

        # Test reconstruction
        x_rec = dwt.reconstruct((cA, cD_list))
        assert x_rec.device.type == device.type

    def test_different_dtypes(self):
        """Test wavelet transforms with different data types."""
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(128, dtype=dtype)

            dwt = DWT1D(wavelet="db2", levels=1)
            cA, cD_list = dwt.decompose(x.unsqueeze(0))

            # Check dtype preserved
            assert cA.dtype == dtype
            for cD in cD_list:
                assert cD.dtype == dtype

            # Test reconstruction
            x_rec = dwt.reconstruct((cA, cD_list))
            assert x_rec.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
