"""Unit tests for Global Filter mixing layers."""

import pytest
import torch

from spectrans.layers.mixing.global_filter import (
    AdaptiveGlobalFilter,
    GlobalFilterMixing,
    GlobalFilterMixing2D,
)


class TestGlobalFilterMixing:
    """Test Global Filter Network mixing layers."""
    
    def test_global_filter_forward_shape(self, random_tensor):
        """Test that GlobalFilterMixing preserves shape."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len)
        
        output = mixer(random_tensor)
        
        assert output.shape == random_tensor.shape
        assert output.dtype == torch.float32
        
    def test_global_filter_2d_forward_shape(self, random_tensor):
        """Test that GlobalFilterMixing2D preserves shape."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = GlobalFilterMixing2D(hidden_dim=hidden_dim, sequence_length=seq_len)
        
        output = mixer(random_tensor)
        
        assert output.shape == random_tensor.shape
        assert output.dtype == torch.float32
        
    def test_adaptive_global_filter_forward_shape(self, random_tensor):
        """Test that AdaptiveGlobalFilter preserves shape."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = AdaptiveGlobalFilter(
            hidden_dim=hidden_dim, 
            sequence_length=seq_len,
            adaptive_initialization=True
        )
        
        output = mixer(random_tensor)
        
        assert output.shape == random_tensor.shape
        assert output.dtype == torch.float32
        
    def test_global_filter_activations(self, random_tensor):
        """Test different activation functions."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        
        activations = ["sigmoid", "tanh", "identity"]
        for activation in activations:
            mixer = GlobalFilterMixing(
                hidden_dim=hidden_dim,
                sequence_length=seq_len,
                activation=activation
            )
            output = mixer(random_tensor)
            assert output.shape == random_tensor.shape
            
    def test_global_filter_invalid_activation(self, random_tensor):
        """Test that invalid activation raises error."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        
        with pytest.raises(ValueError):
            GlobalFilterMixing(
                hidden_dim=hidden_dim,
                sequence_length=seq_len,
                activation="invalid_activation"
            )
            
    def test_global_filter_learnable_parameters(self, random_tensor):
        """Test that global filters have learnable parameters."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len)
        
        # Check that parameters exist and are learnable
        params = list(mixer.parameters())
        assert len(params) > 0
        
        # Should have real and imaginary filter parameters
        param_shapes = [p.shape for p in params]
        expected_shape = (seq_len, hidden_dim)
        assert expected_shape in param_shapes  # Real part
        assert expected_shape in param_shapes  # Imaginary part
        
    def test_global_filter_frequency_response(self, random_tensor):
        """Test frequency response analysis."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len)
        
        # Get frequency response
        response = mixer.get_filter_response()
        assert response.shape == (seq_len, hidden_dim)
        assert response.dtype in [torch.complex64, torch.complex128]
        
        # Analyze frequency response
        analysis = mixer.analyze_frequency_response()
        assert 'magnitude' in analysis
        assert 'phase' in analysis
        assert 'total_energy' in analysis
        
    def test_adaptive_global_filter_regularization(self, random_tensor):
        """Test regularization in adaptive global filter."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = AdaptiveGlobalFilter(
            hidden_dim=hidden_dim,
            sequence_length=seq_len,
            filter_regularization=0.01
        )
        
        # Test regularization loss
        reg_loss = mixer.get_regularization_loss()
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0.0
        
        # Test without regularization
        mixer_no_reg = AdaptiveGlobalFilter(
            hidden_dim=hidden_dim,
            sequence_length=seq_len,
            filter_regularization=0.0
        )
        reg_loss_none = mixer_no_reg.get_regularization_loss()
        assert reg_loss_none.item() == 0.0
        
    def test_global_filter_complexity_properties(self, random_tensor):
        """Test complexity and properties of global filters."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len)
        
        # Check complexity
        complexity = mixer.complexity
        assert 'time' in complexity
        assert 'space' in complexity
        assert 'parameters' in complexity
        assert 'log n' in complexity['time']
        
        # Check properties
        props = mixer.get_spectral_properties()
        assert props['frequency_domain'] is True
        assert props['learnable_filters'] is True
        assert props['selective_filtering'] is True
        assert props['complex_valued'] is True


class TestGlobalFilterGradients:
    """Test gradient computation for global filter layers."""
    
    def test_global_filter_gradients(self, random_tensor):
        """Test gradient computation for global filter mixing."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        random_tensor.requires_grad_(True)
        
        mixer = GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len)
        
        output = mixer(random_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check input gradients
        assert random_tensor.grad is not None
        assert random_tensor.grad.shape == random_tensor.shape
        
        # Check parameter gradients
        for name, param in mixer.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert param.grad.shape == param.shape
            
    def test_gradient_flow_through_complex_operations(self, random_tensor):
        """Test gradient flow through complex operations."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        random_tensor.requires_grad_(True)
        
        mixer = AdaptiveGlobalFilter(
            hidden_dim=hidden_dim,
            sequence_length=seq_len,
            filter_regularization=0.01
        )
        
        output = mixer(random_tensor)
        reg_loss = mixer.get_regularization_loss()
        total_loss = output.sum() + reg_loss
        total_loss.backward()
        
        # Verify gradients for input
        assert random_tensor.grad is not None
        
        # Verify gradients for filter parameters
        assert mixer.filter_real.grad is not None
        assert mixer.filter_imag.grad is not None
        
        # Gradients should be finite
        assert torch.isfinite(random_tensor.grad).all()
        assert torch.isfinite(mixer.filter_real.grad).all()
        assert torch.isfinite(mixer.filter_imag.grad).all()


class TestGlobalFilterEdgeCases:
    """Test edge cases for global filter layers."""
    
    def test_zero_input(self):
        """Test global filter with zero input."""
        hidden_dim = 64
        seq_len = 32
        batch_size = 4
        
        zero_input = torch.zeros(batch_size, seq_len, hidden_dim)
        mixer = GlobalFilterMixing(hidden_dim, seq_len)
        
        output = mixer(zero_input)
        assert output.shape == zero_input.shape
        # Output may not be zero due to learnable filters
        
    def test_single_element_sequences(self):
        """Test with sequence length 1."""
        hidden_dim = 32
        seq_len = 1
        batch_size = 2
        
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
        
        mixer = GlobalFilterMixing(hidden_dim, seq_len)
        output = mixer(input_tensor)
        assert output.shape == input_tensor.shape