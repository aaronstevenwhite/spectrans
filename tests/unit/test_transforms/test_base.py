"""Base test utilities and common test cases for transforms."""

import pytest
import torch

from spectrans.core.registry import registry


class TestTransformProperties:
    """Test general transform properties."""

    def test_transform_registration(self):
        """Test that transforms are properly registered."""
        # Check FFT transforms are registered
        assert ("transform", "fft1d") in registry
        assert ("transform", "fft2d") in registry
        assert ("transform", "rfft") in registry

        # Check DCT/DST transforms are registered
        assert ("transform", "dct") in registry
        assert ("transform", "dst") in registry

        # Check Hadamard transforms are registered
        assert ("transform", "hadamard") in registry

    def test_transform_properties(self):
        """Test transform property flags."""
        from spectrans.transforms import FFT1D, DCT, HadamardTransform
        
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

    def test_transform_deterministic(self, random_tensor):
        """Test that transforms are deterministic."""
        from spectrans.transforms import FFT1D, RFFT, DCT, DST, HadamardTransform
        
        transform_classes = [FFT1D, RFFT, DCT, DST, HadamardTransform]
        
        for transform_cls in transform_classes:
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
        from spectrans.transforms import FFT1D
        
        transform = FFT1D()

        # Empty tensor - FFT should handle it or raise error
        x = torch.empty(0, device=device)
        # PyTorch FFT raises RuntimeError for empty tensors
        with pytest.raises(RuntimeError):
            transform.transform(x)

    def test_single_element(self, device):
        """Test single-element tensors."""
        from spectrans.transforms import DCT
        
        transform = DCT()

        x = torch.tensor([5.0], device=device)
        coeffs = transform.transform(x)
        reconstructed = transform.inverse_transform(coeffs)

        torch.testing.assert_close(reconstructed, x)

    def test_batch_processing(self, batch_size, sequence_length, hidden_dim, device):
        """Test batch processing of transforms."""
        from spectrans.transforms import FFT1D
        
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