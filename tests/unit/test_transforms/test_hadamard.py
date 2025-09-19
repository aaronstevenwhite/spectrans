"""Unit tests for Hadamard transforms."""

import pytest
import torch

from spectrans.transforms import HadamardTransform


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
        torch.testing.assert_close(reconstructed, x, rtol=1e-4, atol=1e-6)

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

    def test_hadamard_self_inverse(self, device):
        """Test that Hadamard transform is its own inverse."""
        transform = HadamardTransform(normalized=True)

        x = torch.randn(32, device=device)

        # Apply transform twice
        h1 = transform.transform(x)
        h2 = transform.transform(h1)

        # Should get back original
        torch.testing.assert_close(h2, x, rtol=1e-5, atol=1e-7)

    def test_hadamard_energy_conservation(self, device):
        """Test energy conservation for normalized Hadamard transform."""
        transform = HadamardTransform(normalized=True)

        x = torch.randn(64, device=device)

        # Compute energy in original domain
        energy_original = torch.sum(x**2)

        # Transform and compute energy
        h_coeffs = transform.transform(x)
        energy_transformed = torch.sum(h_coeffs**2)

        # Energy should be conserved for orthogonal transform
        torch.testing.assert_close(
            energy_original,
            energy_transformed,
            rtol=1e-5,
            atol=1e-7,
            msg="Hadamard transform should conserve energy",
        )

    @pytest.mark.parametrize("normalized", [True, False])
    def test_hadamard_normalization_modes(self, normalized, device):
        """Test Hadamard transform with different normalization modes."""
        transform = HadamardTransform(normalized=normalized)

        x = torch.randn(32, device=device)

        # Forward and inverse
        h_coeffs = transform.transform(x)
        reconstructed = transform.inverse_transform(h_coeffs)

        # Should reconstruct perfectly
        torch.testing.assert_close(reconstructed, x, rtol=1e-4, atol=1e-6)

    def test_hadamard_gradient_flow(self, device):
        """Test that gradients flow through Hadamard transform."""
        x = torch.randn(2, 32, requires_grad=True, device=device)

        transform = HadamardTransform(normalized=True)

        # Forward pass
        h_coeffs = transform.transform(x)
        loss = torch.sum(h_coeffs**2)

        # Backward pass
        loss.backward()

        # Check gradient exists and is non-zero
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_hadamard_batch_dimensions(self, device):
        """Test Hadamard transform with various batch dimensions."""
        shapes = [
            (32,),  # 1D
            (4, 32),  # 2D with batch
            (4, 8, 32),  # 3D with batch
            (2, 4, 8, 32),  # 4D with batch
        ]

        transform = HadamardTransform(normalized=True)

        for shape in shapes:
            x = torch.randn(*shape, device=device)
            h_coeffs = transform.transform(x)
            assert h_coeffs.shape == x.shape

            reconstructed = transform.inverse_transform(h_coeffs)
            torch.testing.assert_close(reconstructed, x, rtol=1e-4, atol=1e-6)

    def test_hadamard_matrix_properties(self, device):
        """Test properties of Hadamard matrix."""
        transform = HadamardTransform(normalized=False)

        # Apply to identity to get Hadamard matrix columns
        n = 8
        eye = torch.eye(n, device=device)
        H = transform.transform(eye, dim=0)

        # Check that all entries are ±1 (for unnormalized)
        assert torch.all(torch.abs(H) == 1)

        # Check orthogonality: H^T H = n*I for unnormalized
        gram = torch.matmul(H.T, H)
        expected = n * torch.eye(n, device=device)
        torch.testing.assert_close(gram, expected, rtol=1e-5, atol=1e-7)

    def test_hadamard_recursive_structure(self, device):
        """Test that Hadamard matrices have recursive structure."""
        transform = HadamardTransform(normalized=False)

        # Get 2x2 Hadamard matrix
        H2 = transform.transform(torch.eye(2, device=device), dim=0)

        # Get 4x4 Hadamard matrix
        H4 = transform.transform(torch.eye(4, device=device), dim=0)

        # H4 should have structure: [[H2, H2], [H2, -H2]]
        # Check top-left block
        torch.testing.assert_close(H4[:2, :2], H2)
        # Check top-right block
        torch.testing.assert_close(H4[:2, 2:], H2)
        # Check bottom-left block
        torch.testing.assert_close(H4[2:, :2], H2)
        # Check bottom-right block
        torch.testing.assert_close(H4[2:, 2:], -H2)

    @pytest.mark.parametrize("dim", [0, 1, -1, -2])
    def test_hadamard_different_dimensions(self, dim, device):
        """Test Hadamard transform along different dimensions."""
        x = torch.randn(8, 16, 32, device=device)

        transform = HadamardTransform(normalized=True)

        # Transform along specified dimension
        h_coeffs = transform.transform(x, dim=dim)
        assert h_coeffs.shape == x.shape

        # Inverse transform
        reconstructed = transform.inverse_transform(h_coeffs, dim=dim)
        torch.testing.assert_close(reconstructed, x, rtol=1e-4, atol=1e-6)

    def test_hadamard_complex_input(self, device):
        """Test Hadamard transform with complex input."""
        real = torch.randn(32, device=device)
        imag = torch.randn(32, device=device)
        x = torch.complex(real, imag)

        transform = HadamardTransform(normalized=True)

        # Forward and inverse
        h_coeffs = transform.transform(x)
        reconstructed = transform.inverse_transform(h_coeffs)

        # Should work with complex numbers
        torch.testing.assert_close(reconstructed, x, rtol=1e-4, atol=1e-6)

    def test_hadamard_fast_implementation(self, device):
        """Test that fast Hadamard transform is efficient."""
        # Test that implementation is actually fast (uses recursion/butterfly)
        # This is more of a performance test
        sizes = [256, 512, 1024]

        transform = HadamardTransform(normalized=True)

        for size in sizes:
            x = torch.randn(size, device=device)

            # Just verify it completes quickly and correctly
            h_coeffs = transform.transform(x)
            reconstructed = transform.inverse_transform(h_coeffs)

            torch.testing.assert_close(reconstructed, x, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
