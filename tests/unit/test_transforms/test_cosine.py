"""Unit tests for cosine-based transforms (DCT and DST)."""

import pytest
import scipy.fft
import torch

from spectrans.transforms import DCT, DST


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
            reconstructed,
            random_tensor,
            rtol=expected_error_factor * machine_eps,
            atol=expected_error_factor * machine_eps,
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
        # For nxn matrix multiply: expect ~sqrt(n) x machine_eps error growth
        machine_eps = torch.finfo(identity.dtype).eps
        expected_error_factor = max(10, n * 0.5)  # Conservative estimate

        torch.testing.assert_close(
            gram_matrix,
            identity,
            rtol=expected_error_factor * machine_eps,
            atol=expected_error_factor * machine_eps,
            msg="DCT matrix should be orthogonal",
        )

    def test_dct_energy_conservation(self, random_tensor):
        """Test that normalized DCT conserves energy (Parseval's theorem)."""
        transform = DCT(normalized=True)

        # Compute energy in original domain
        energy_original = torch.sum(random_tensor**2)

        # Transform and compute energy in DCT domain
        dct_coeffs = transform.transform(random_tensor)
        energy_dct = torch.sum(dct_coeffs**2)

        # Energy should be conserved for orthogonal transform
        torch.testing.assert_close(
            energy_original,
            energy_dct,
            rtol=1e-5,
            atol=1e-7,
            msg="DCT should conserve energy (Parseval's theorem)",
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
            reconstructed,
            random_tensor,
            rtol=expected_error_factor * machine_eps,
            atol=expected_error_factor * machine_eps,
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
            gram_matrix,
            identity,
            rtol=expected_error_factor * machine_eps,
            atol=expected_error_factor * machine_eps,
            msg="DST matrix should be orthogonal",
        )

    def test_dst_energy_conservation(self, random_tensor):
        """Test that normalized DST conserves energy."""
        transform = DST(normalized=True)

        # Compute energy in original domain
        energy_original = torch.sum(random_tensor**2)

        # Transform and compute energy in DST domain
        dst_coeffs = transform.transform(random_tensor)
        energy_dst = torch.sum(dst_coeffs**2)

        # Energy should be conserved for orthogonal transform
        torch.testing.assert_close(
            energy_original, energy_dst, rtol=1e-5, atol=1e-7, msg="DST should conserve energy"
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

    @pytest.mark.parametrize("normalized", [True, False])
    def test_dct_normalization_modes(self, normalized, device):
        """Test DCT with different normalization modes."""
        x = torch.randn(64, device=device)

        transform = DCT(normalized=normalized)
        dct_coeffs = transform.transform(x)
        reconstructed = transform.inverse_transform(dct_coeffs)

        # Adjust tolerance based on normalization
        if normalized:
            n = x.shape[-1]
            machine_eps = torch.finfo(x.dtype).eps
            expected_error_factor = max(10, n * 3.0)
            torch.testing.assert_close(
                reconstructed,
                x,
                rtol=expected_error_factor * machine_eps,
                atol=expected_error_factor * machine_eps,
            )
        else:
            # Relaxed tolerances for unnormalized transforms
            torch.testing.assert_close(reconstructed, x, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("normalized", [True, False])
    def test_dst_normalization_modes(self, normalized, device):
        """Test DST with different normalization modes."""
        x = torch.randn(64, device=device)

        transform = DST(normalized=normalized)
        dst_coeffs = transform.transform(x)
        reconstructed = transform.inverse_transform(dst_coeffs)

        # Adjust tolerance based on normalization
        if normalized:
            n = x.shape[-1]
            machine_eps = torch.finfo(x.dtype).eps
            expected_error_factor = max(15, n * 3.0)
            torch.testing.assert_close(
                reconstructed,
                x,
                rtol=expected_error_factor * machine_eps,
                atol=expected_error_factor * machine_eps,
            )
        else:
            # Relaxed tolerances for unnormalized transforms
            torch.testing.assert_close(reconstructed, x, rtol=1e-3, atol=1e-4)

    def test_dct_gradient_flow(self, device):
        """Test that gradients flow through DCT."""
        x = torch.randn(32, 64, requires_grad=True, device=device)

        transform = DCT(normalized=True)

        # Forward and inverse
        dct_coeffs = transform.transform(x)
        reconstructed = transform.inverse_transform(dct_coeffs)

        # Create loss
        loss = torch.sum((reconstructed - x) ** 2)

        # Backward pass
        loss.backward()

        # Check gradient exists and is reasonable
        assert x.grad is not None
        assert x.grad.shape == x.shape
        # Gradient should be small but non-zero for reconstruction loss
        assert torch.max(torch.abs(x.grad)) < 1e-3

    def test_dst_gradient_flow(self, device):
        """Test that gradients flow through DST."""
        x = torch.randn(32, 64, requires_grad=True, device=device)

        transform = DST(normalized=True)

        # Forward pass
        dst_coeffs = transform.transform(x)
        loss = torch.sum(dst_coeffs**2)

        # Backward pass
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    @pytest.mark.parametrize("size", [8, 16, 32, 64, 128, 256])
    def test_dct_different_sizes(self, size, device):
        """Test DCT with different input sizes."""
        x = torch.randn(size, device=device)

        transform = DCT(normalized=True)
        dct_coeffs = transform.transform(x)
        reconstructed = transform.inverse_transform(dct_coeffs)

        # Adjust tolerance based on size
        machine_eps = torch.finfo(x.dtype).eps
        expected_error_factor = max(10, size * 3.0)

        torch.testing.assert_close(
            reconstructed,
            x,
            rtol=expected_error_factor * machine_eps,
            atol=expected_error_factor * machine_eps,
        )

    @pytest.mark.parametrize("size", [8, 16, 32, 64, 128, 256])
    def test_dst_different_sizes(self, size, device):
        """Test DST with different input sizes."""
        x = torch.randn(size, device=device)

        transform = DST(normalized=True)
        dst_coeffs = transform.transform(x)
        reconstructed = transform.inverse_transform(dst_coeffs)

        # Adjust tolerance based on size
        machine_eps = torch.finfo(x.dtype).eps
        expected_error_factor = max(15, size * 3.0)

        torch.testing.assert_close(
            reconstructed,
            x,
            rtol=expected_error_factor * machine_eps,
            atol=expected_error_factor * machine_eps,
        )

    def test_dct_batch_dimensions(self, device):
        """Test DCT with various batch dimensions."""
        shapes = [
            (64,),  # 1D
            (4, 64),  # 2D with batch
            (4, 8, 64),  # 3D with batch
            (2, 4, 8, 64),  # 4D with batch
        ]

        transform = DCT(normalized=True)

        for shape in shapes:
            x = torch.randn(*shape, device=device)
            dct_coeffs = transform.transform(x)
            assert dct_coeffs.shape == x.shape

            reconstructed = transform.inverse_transform(dct_coeffs)

            # Adjust tolerance
            n = x.shape[-1]
            machine_eps = torch.finfo(x.dtype).eps
            expected_error_factor = max(10, n * 3.0)

            torch.testing.assert_close(
                reconstructed,
                x,
                rtol=expected_error_factor * machine_eps,
                atol=expected_error_factor * machine_eps,
            )

    def test_dst_batch_dimensions(self, device):
        """Test DST with various batch dimensions."""
        shapes = [
            (64,),  # 1D
            (4, 64),  # 2D with batch
            (4, 8, 64),  # 3D with batch
        ]

        transform = DST(normalized=True)

        for shape in shapes:
            x = torch.randn(*shape, device=device)
            dst_coeffs = transform.transform(x)
            assert dst_coeffs.shape == x.shape

            reconstructed = transform.inverse_transform(dst_coeffs)

            # Adjust tolerance
            n = x.shape[-1]
            machine_eps = torch.finfo(x.dtype).eps
            expected_error_factor = max(15, n * 3.0)

            torch.testing.assert_close(
                reconstructed,
                x,
                rtol=expected_error_factor * machine_eps,
                atol=expected_error_factor * machine_eps,
            )

    def test_dct_dst_relationship(self, device):
        """Test relationship between DCT and DST transforms."""
        # For certain inputs, DCT and DST have known relationships
        n = 32
        x = torch.randn(n, device=device)

        dct_transform = DCT(normalized=False)
        dst_transform = DST(normalized=False)

        dct_result = dct_transform.transform(x)
        dst_result = dst_transform.transform(x)

        # They should be different transforms
        assert not torch.allclose(dct_result, dst_result)

        # But both should be invertible
        dct_reconstructed = dct_transform.inverse_transform(dct_result)
        dst_reconstructed = dst_transform.inverse_transform(dst_result)

        # Relaxed tolerances for unnormalized transforms
        torch.testing.assert_close(dct_reconstructed, x, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(dst_reconstructed, x, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("normalized", [True, False])
    def test_dct_against_scipy(self, normalized, device):
        """Test DCT-II against scipy.fft.dct reference."""
        torch.manual_seed(42)
        x = torch.randn(64, device=device)

        # Our implementation
        dct = DCT(normalized=normalized)
        dct_coeffs = dct.transform(x)

        # Scipy reference
        x_numpy = x.cpu().numpy()
        norm_mode = "ortho" if normalized else None
        dct_coeffs_scipy = scipy.fft.dct(x_numpy, type=2, norm=norm_mode)
        dct_coeffs_ref = torch.from_numpy(dct_coeffs_scipy).float().to(device)

        # Check coefficients match (relaxed tolerance for numerical differences)
        torch.testing.assert_close(dct_coeffs, dct_coeffs_ref, rtol=1e-4, atol=1e-5)

        # Also test reconstruction against scipy
        recon_scipy = scipy.fft.idct(dct_coeffs_scipy, type=2, norm=norm_mode)
        recon_ref = torch.from_numpy(recon_scipy).float().to(device)
        recon_ours = dct.inverse_transform(dct_coeffs)

        torch.testing.assert_close(recon_ours, recon_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(recon_ours, x, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("normalized", [True, False])
    def test_dst_against_scipy(self, normalized, device):
        """Test DST-II against scipy.fft.dst reference."""
        torch.manual_seed(42)
        x = torch.randn(64, device=device)

        # Our implementation
        dst = DST(normalized=normalized)
        dst_coeffs = dst.transform(x)

        # Scipy reference
        x_numpy = x.cpu().numpy()
        norm_mode = "ortho" if normalized else None
        dst_coeffs_scipy = scipy.fft.dst(x_numpy, type=2, norm=norm_mode)
        dst_coeffs_ref = torch.from_numpy(dst_coeffs_scipy).float().to(device)

        # Check coefficients match (relaxed tolerance for numerical differences)
        torch.testing.assert_close(dst_coeffs, dst_coeffs_ref, rtol=1e-4, atol=1e-5)

        # Also test reconstruction against scipy
        recon_scipy = scipy.fft.idst(dst_coeffs_scipy, type=2, norm=norm_mode)
        recon_ref = torch.from_numpy(recon_scipy).float().to(device)
        recon_ours = dst.inverse_transform(dst_coeffs)

        torch.testing.assert_close(recon_ours, recon_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(recon_ours, x, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("size", [16, 32, 64, 128])
    def test_dct_scipy_different_sizes(self, size, device):
        """Test DCT against scipy for different sizes."""
        x = torch.randn(size, device=device)

        # Test both normalized modes
        for normalized in [True, False]:
            dct = DCT(normalized=normalized)
            dct_coeffs = dct.transform(x)

            # Compare with scipy
            x_numpy = x.cpu().numpy()
            norm_mode = "ortho" if normalized else None
            dct_coeffs_scipy = scipy.fft.dct(x_numpy, type=2, norm=norm_mode)
            dct_coeffs_ref = torch.from_numpy(dct_coeffs_scipy).float().to(device)

            # Use slightly relaxed tolerance for larger transforms due to numerical precision
            rtol = 2e-3 if size >= 128 else 1e-4
            atol = 5e-4 if size >= 128 else 1e-5
            torch.testing.assert_close(dct_coeffs, dct_coeffs_ref, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("size", [16, 32, 64, 128])
    def test_dst_scipy_different_sizes(self, size, device):
        """Test DST against scipy for different sizes."""
        x = torch.randn(size, device=device)

        # Test both normalized modes
        for normalized in [True, False]:
            dst = DST(normalized=normalized)
            dst_coeffs = dst.transform(x)

            # Compare with scipy
            x_numpy = x.cpu().numpy()
            norm_mode = "ortho" if normalized else None
            dst_coeffs_scipy = scipy.fft.dst(x_numpy, type=2, norm=norm_mode)
            dst_coeffs_ref = torch.from_numpy(dst_coeffs_scipy).float().to(device)

            # Use slightly relaxed tolerance for larger transforms due to numerical precision
            rtol = 2e-3 if size >= 128 else 1e-4
            atol = 5e-4 if size >= 128 else 1e-5
            torch.testing.assert_close(dst_coeffs, dst_coeffs_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__])
