"""Unit tests for kernel functions and feature maps."""

import math

import pytest
import torch

from spectrans.kernels import (
    CosineKernel,
    FourierKernel,
    GaussianRFFKernel,
    LaplacianRFFKernel,
    LearnableSpectralKernel,
    OrthogonalRandomFeatures,
    PolynomialKernel,
    PolynomialSpectralKernel,
    RFFAttentionKernel,
    TruncatedSVDKernel,
)


class TestBasicKernels:
    """Test basic kernel functions."""
    
    def test_polynomial_kernel_shapes(self, random_tensor):
        """Test polynomial kernel output shapes."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        kernel = PolynomialKernel(degree=2, alpha=1.0, coef0=1.0)
        
        # Test compute
        output = kernel.compute(random_tensor, random_tensor)
        assert output.shape == (batch_size, seq_len, seq_len)
        
        # Test with different shapes
        y = torch.randn(batch_size, seq_len // 2, hidden_dim)
        output = kernel.compute(random_tensor, y)
        assert output.shape == (batch_size, seq_len, seq_len // 2)
    
    def test_polynomial_kernel_positive_definite(self, random_tensor):
        """Test that polynomial kernel produces positive definite matrices."""
        kernel = PolynomialKernel(degree=2, coef0=1.0)
        
        # Check positive definiteness for various inputs
        assert kernel.is_positive_definite(random_tensor)
        
    def test_cosine_kernel_shapes(self, random_tensor):
        """Test cosine kernel output shapes."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        kernel = CosineKernel()
        
        output = kernel.compute(random_tensor, random_tensor)
        assert output.shape == (batch_size, seq_len, seq_len)
        
        # Values should be in [-1, 1] for cosine similarity
        assert torch.all(output >= -1.001)  # Small tolerance
        assert torch.all(output <= 1.001)
        
    def test_cosine_kernel_self_similarity(self, random_tensor):
        """Test that cosine kernel gives 1 for self-similarity."""
        kernel = CosineKernel()
        
        # Normalized vectors should have self-similarity of 1
        x = F.normalize(random_tensor, dim=-1)
        K = kernel.compute(x, x)
        
        # Diagonal should be approximately 1
        diagonal = torch.diagonal(K, dim1=-2, dim2=-1)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-5)


class TestRFFKernels:
    """Test Random Fourier Features kernels."""
    
    def test_gaussian_rff_shapes(self, random_tensor):
        """Test Gaussian RFF kernel output shapes."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        num_features = 128
        
        kernel = GaussianRFFKernel(
            input_dim=hidden_dim,
            num_features=num_features,
            sigma=1.0,
            use_cos_sin=False,
        )
        
        # Test forward (feature map)
        features = kernel.forward(random_tensor)
        assert features.shape == (batch_size, seq_len, num_features)
        
        # Test with cos+sin features
        kernel_cos_sin = GaussianRFFKernel(
            input_dim=hidden_dim,
            num_features=num_features,
            sigma=1.0,
            use_cos_sin=True,
        )
        features = kernel_cos_sin.forward(random_tensor)
        assert features.shape == (batch_size, seq_len, num_features * 2)
        
    def test_gaussian_rff_kernel_approximation(self, random_tensor):
        """Test that RFF approximates the true Gaussian kernel."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        sigma = 1.0
        
        # Use many features for good approximation
        kernel = GaussianRFFKernel(
            input_dim=hidden_dim,
            num_features=1024,
            sigma=sigma,
            seed=42,
        )
        
        # Small inputs for testing
        x = random_tensor[:1, :10, :]  # (1, 10, hidden_dim)
        
        # Compute approximate kernel
        features_x = kernel.forward(x)
        K_approx = kernel.kernel_approximation(x, x)
        
        # Compute true kernel
        K_true = kernel.compute(x, x)
        
        # Check approximation quality
        # With 1024 features, should be reasonably close
        error = torch.abs(K_approx - K_true).mean()
        assert error < 0.1  # Reasonable tolerance
        
    def test_laplacian_rff_shapes(self, random_tensor):
        """Test Laplacian RFF kernel shapes."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        num_features = 128
        
        kernel = LaplacianRFFKernel(
            input_dim=hidden_dim,
            num_features=num_features,
            sigma=1.0,
        )
        
        features = kernel.forward(random_tensor)
        assert features.shape == (batch_size, seq_len, num_features)
        
    def test_orthogonal_random_features_shapes(self, random_tensor):
        """Test orthogonal random features shapes."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        num_features = 128
        
        # Test without Hadamard
        orf = OrthogonalRandomFeatures(
            input_dim=hidden_dim,
            num_features=num_features,
            kernel_type="gaussian",
            use_hadamard=False,
        )
        
        features = orf(random_tensor)
        assert features.shape == (batch_size, seq_len, num_features)
        
    def test_rff_attention_kernel_shapes(self, random_tensor):
        """Test RFF attention kernel shapes."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        num_features = 128
        
        kernel = RFFAttentionKernel(
            input_dim=hidden_dim,
            num_features=num_features,
            kernel_type="softmax",
            use_orthogonal=True,
        )
        
        features = kernel(random_tensor)
        assert features.shape == (batch_size, seq_len, num_features)
        
        # Features should be positive for softmax kernel
        assert torch.all(features >= 0)
    
    def test_rff_attention_kernel_types(self, random_tensor):
        """Test different RFF attention kernel types."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        num_features = 64
        
        # Test softmax kernel
        kernel_softmax = RFFAttentionKernel(
            input_dim=hidden_dim,
            num_features=num_features,
            kernel_type="softmax",
        )
        features_softmax = kernel_softmax(random_tensor)
        assert torch.all(features_softmax >= 0)  # Positive features
        
        # Test ReLU kernel
        kernel_relu = RFFAttentionKernel(
            input_dim=hidden_dim,
            num_features=num_features,
            kernel_type="relu",
        )
        features_relu = kernel_relu(random_tensor)
        assert torch.all(features_relu >= 0)  # ReLU is non-negative
        
        # Test ELU kernel
        kernel_elu = RFFAttentionKernel(
            input_dim=hidden_dim,
            num_features=num_features,
            kernel_type="elu",
        )
        features_elu = kernel_elu(random_tensor)
        assert features_elu.shape == (batch_size, seq_len, num_features)


class TestSpectralKernels:
    """Test spectral kernel functions."""
    
    def test_polynomial_spectral_kernel_shapes(self, random_tensor):
        """Test polynomial spectral kernel shapes."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        rank = 16
        
        kernel = PolynomialSpectralKernel(rank=rank, degree=2)
        
        # Test compute
        K = kernel.compute(random_tensor, random_tensor)
        assert K.shape == (batch_size, seq_len, seq_len)
        
        # Test attention computation
        attention = kernel.compute_attention(random_tensor, random_tensor)
        assert attention.shape == (batch_size, seq_len, seq_len)
        
        # Check attention normalization
        row_sums = attention.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
        
    def test_truncated_svd_kernel_shapes(self, random_tensor):
        """Test truncated SVD kernel shapes."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        rank = 8
        
        kernel = TruncatedSVDKernel(rank=rank, use_randomized=False)
        
        # Test with standard SVD
        K = kernel.compute(random_tensor, random_tensor)
        assert K.shape == (batch_size, seq_len, seq_len)
        
        # Test with randomized SVD
        kernel_rand = TruncatedSVDKernel(rank=rank, use_randomized=True)
        K_rand = kernel_rand.compute(random_tensor, random_tensor)
        assert K_rand.shape == (batch_size, seq_len, seq_len)
        
    def test_truncated_svd_approximation_quality(self):
        """Test SVD approximation quality."""
        # Create a low-rank matrix for testing
        n, d, r_true = 20, 30, 5
        U = torch.randn(n, r_true)
        V = torch.randn(d, r_true)
        X = torch.matmul(U, V.T)  # Rank-5 matrix
        
        # Add batch dimension
        X = X.unsqueeze(0)  # (1, n, d)
        
        # Approximate with rank-5 kernel
        kernel = TruncatedSVDKernel(rank=r_true, normalize=False)
        K_true = torch.matmul(X, X.transpose(-2, -1))
        K_approx = kernel.compute(X, X)
        
        # Should be very close for exact rank
        error = torch.norm(K_true - K_approx) / torch.norm(K_true)
        assert error < 1e-5
        
    def test_learnable_spectral_kernel_shapes(self, random_tensor):
        """Test learnable spectral kernel shapes."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        rank = 16
        
        kernel = LearnableSpectralKernel(
            input_dim=hidden_dim,
            rank=rank,
            trainable_eigenvectors=True,
        )
        
        # Test compute
        K = kernel.compute(random_tensor, random_tensor)
        assert K.shape == (batch_size, seq_len, seq_len)
        
        # Test feature extraction
        features = kernel.extract_features(random_tensor)
        assert features.shape == (batch_size, seq_len, rank)
        
        # Test forward (nn.Module interface)
        output = kernel(random_tensor, random_tensor)
        assert output.shape == K.shape
        
        features_only = kernel(random_tensor)
        assert features_only.shape == features.shape
        
    def test_learnable_spectral_kernel_training(self, random_tensor):
        """Test that learnable spectral kernel parameters can be trained."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        kernel = LearnableSpectralKernel(
            input_dim=hidden_dim,
            rank=8,
            trainable_eigenvectors=True,
        )
        
        # Check parameters are registered
        params = list(kernel.parameters())
        assert len(params) > 0
        
        # Compute gradient
        K = kernel.compute(random_tensor, random_tensor)
        loss = K.mean()
        loss.backward()
        
        # Check gradients exist
        for param in params:
            assert param.grad is not None
            assert not torch.all(param.grad == 0)
            
    def test_learnable_spectral_kernel_orthogonalization(self, random_tensor):
        """Test eigenvector orthogonalization."""
        hidden_dim = random_tensor.shape[-1]
        kernel = LearnableSpectralKernel(
            input_dim=hidden_dim,
            rank=8,
            trainable_eigenvectors=True,
        )
        
        # Orthogonalize
        kernel.orthogonalize_eigenvectors()
        
        # Check orthogonality
        Q = kernel.eigenvectors
        QTQ = torch.matmul(Q.T, Q)
        I = torch.eye(Q.shape[1], device=Q.device)
        assert torch.allclose(QTQ, I, atol=1e-5)
        
    def test_fourier_kernel_shapes(self, random_tensor):
        """Test Fourier kernel shapes."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        rank = 32
        
        kernel = FourierKernel(
            rank=rank,
            input_dim=hidden_dim,
            learnable_filter=True,
            filter_type="gaussian",
        )
        
        K = kernel.compute(random_tensor, random_tensor)
        assert K.shape == (batch_size, seq_len, seq_len)
        
    def test_fourier_kernel_filter_types(self, random_tensor):
        """Test different Fourier kernel filter types."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        rank = 16
        
        # Test Gaussian filter
        kernel_gauss = FourierKernel(
            rank=rank,
            input_dim=hidden_dim,
            filter_type="gaussian",
            cutoff_freq=0.5,
        )
        K_gauss = kernel_gauss.compute(random_tensor, random_tensor)
        assert K_gauss.shape == (batch_size, seq_len, seq_len)
        
        # Test Butterworth filter
        kernel_butter = FourierKernel(
            rank=rank,
            input_dim=hidden_dim,
            filter_type="butterworth",
            cutoff_freq=0.5,
        )
        K_butter = kernel_butter.compute(random_tensor, random_tensor)
        assert K_butter.shape == (batch_size, seq_len, seq_len)
        
        # Test ideal filter
        kernel_ideal = FourierKernel(
            rank=rank,
            input_dim=hidden_dim,
            filter_type="ideal",
            cutoff_freq=0.5,
        )
        K_ideal = kernel_ideal.compute(random_tensor, random_tensor)
        assert K_ideal.shape == (batch_size, seq_len, seq_len)


class TestKernelProperties:
    """Test mathematical properties of kernels."""
    
    def test_kernel_symmetry(self, random_tensor):
        """Test that kernels produce symmetric matrices."""
        kernels = [
            PolynomialKernel(degree=2),
            CosineKernel(),
            PolynomialSpectralKernel(rank=8, degree=2),
        ]
        
        for kernel in kernels:
            K = kernel.compute(random_tensor, random_tensor)
            # Check symmetry
            assert torch.allclose(K, K.transpose(-2, -1), atol=1e-5)
            
    def test_kernel_positive_definiteness(self, random_tensor):
        """Test positive definiteness of kernel matrices."""
        # Use small tensor for numerical stability
        x = random_tensor[:1, :10, :]
        
        kernels = [
            PolynomialKernel(degree=2, coef0=1.0),
            CosineKernel(),
        ]
        
        for kernel in kernels:
            # Check using built-in method
            assert kernel.is_positive_definite(x)
            
            # Also check eigenvalues directly
            K = kernel.gram_matrix(x)
            eigenvalues = torch.linalg.eigvalsh(K)
            assert torch.all(eigenvalues > -1e-5)  # Allow small numerical errors
            
    def test_rff_approximation_convergence(self):
        """Test that RFF approximation improves with more features."""
        torch.manual_seed(42)
        
        # Create test data
        x = torch.randn(1, 20, 32)
        sigma = 1.0
        
        # Test with increasing number of features
        num_features_list = [32, 128, 512]
        errors = []
        
        for num_features in num_features_list:
            kernel = GaussianRFFKernel(
                input_dim=32,
                num_features=num_features,
                sigma=sigma,
                seed=42,
            )
            
            # Compute true and approximate kernels
            K_true = kernel.compute(x, x)
            K_approx = kernel.kernel_approximation(x, x)
            
            # Compute error
            error = torch.norm(K_true - K_approx) / torch.norm(K_true)
            errors.append(error.item())
            
        # Errors should decrease with more features
        assert errors[0] > errors[1] > errors[2]
        # With 512 features, should have reasonable approximation
        # Note: RFF approximation quality varies, allow higher tolerance
        assert errors[-1] < 0.3


class TestKernelComplexity:
    """Test computational complexity properties."""
    
    def test_complexity_info(self):
        """Test that kernels report complexity information."""
        kernels = [
            PolynomialKernel(),
            CosineKernel(),
            GaussianRFFKernel(input_dim=64, num_features=128),
            PolynomialSpectralKernel(rank=16),
            TruncatedSVDKernel(rank=8),
        ]
        
        for kernel in kernels:
            complexity = kernel.complexity
            assert isinstance(complexity, dict)
            assert 'time' in complexity
            assert 'space' in complexity
            assert isinstance(complexity['time'], str)
            assert isinstance(complexity['space'], str)


# Import F for normalization
import torch.nn.functional as F