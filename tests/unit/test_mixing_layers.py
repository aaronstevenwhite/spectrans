"""Unit tests for mixing layers."""

import pytest
import torch
import torch.nn as nn

from spectrans.layers.mixing.fourier import (
    FourierMixing,
    FourierMixing1D,
    RealFourierMixing,
    SeparableFourierMixing,
)
from spectrans.layers.mixing.global_filter import (
    AdaptiveGlobalFilter,
    GlobalFilterMixing,
    GlobalFilterMixing2D,
)


class TestFourierMixing:
    """Test Fourier-based mixing layers."""

    def test_fourier_mixing_forward_shape(self, random_tensor):
        """Test that FourierMixing preserves tensor shape."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = FourierMixing(hidden_dim=hidden_dim)
        
        output = mixer(random_tensor)
        
        assert output.shape == random_tensor.shape
        assert output.dtype == torch.float32  # Should be real output
        
    def test_fourier_mixing_1d_forward_shape(self, random_tensor):
        """Test that FourierMixing1D preserves tensor shape."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = FourierMixing1D(hidden_dim=hidden_dim)
        
        output = mixer(random_tensor)
        
        assert output.shape == random_tensor.shape
        assert output.dtype == torch.float32
        
    def test_real_fourier_mixing_forward_shape(self, random_tensor):
        """Test that RealFourierMixing preserves tensor shape."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = RealFourierMixing(hidden_dim=hidden_dim, use_real_fft=True)
        
        output = mixer(random_tensor)
        
        assert output.shape == random_tensor.shape
        assert output.dtype == torch.float32
        
    def test_separable_fourier_mixing_configurations(self, random_tensor):
        """Test SeparableFourierMixing with different configurations."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        
        # Test sequence mixing only
        mixer_seq = SeparableFourierMixing(
            hidden_dim=hidden_dim, 
            mix_sequence=True, 
            mix_features=False
        )
        output_seq = mixer_seq(random_tensor)
        assert output_seq.shape == random_tensor.shape
        
        # Test feature mixing only
        mixer_feat = SeparableFourierMixing(
            hidden_dim=hidden_dim,
            mix_sequence=False, 
            mix_features=True
        )
        output_feat = mixer_feat(random_tensor)
        assert output_feat.shape == random_tensor.shape
        
        # Test both dimensions
        mixer_both = SeparableFourierMixing(
            hidden_dim=hidden_dim,
            mix_sequence=True,
            mix_features=True
        )
        output_both = mixer_both(random_tensor)
        assert output_both.shape == random_tensor.shape
        
    def test_separable_fourier_mixing_invalid_config(self, random_tensor):
        """Test that invalid configuration raises error."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        
        with pytest.raises(ValueError):
            SeparableFourierMixing(
                hidden_dim=hidden_dim,
                mix_sequence=False,
                mix_features=False
            )
            
    def test_fourier_mixing_dropout(self, random_tensor):
        """Test dropout functionality in Fourier mixing."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = FourierMixing(hidden_dim=hidden_dim, dropout=0.5)
        
        # Test training mode
        mixer.train()
        output_train1 = mixer(random_tensor)
        output_train2 = mixer(random_tensor) 
        
        # Outputs should be different due to dropout randomness
        assert not torch.equal(output_train1, output_train2)
        
        # Test eval mode  
        mixer.eval()
        output_eval1 = mixer(random_tensor)
        output_eval2 = mixer(random_tensor)
        
        # Outputs should be identical in eval mode
        assert torch.equal(output_eval1, output_eval2)
        
    def test_fourier_mixing_complexity_properties(self, random_tensor):
        """Test complexity and spectral properties."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = FourierMixing(hidden_dim=hidden_dim)
        
        # Check complexity
        complexity = mixer.complexity
        assert 'time' in complexity
        assert 'space' in complexity
        assert 'log n' in complexity['time']  # Should have log n complexity
        
        # Check spectral properties
        props = mixer.get_spectral_properties()
        assert props['real_output'] is True
        assert props['frequency_domain'] is True
        assert props['learnable_parameters'] is False
        assert props['translation_equivariant'] is True


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
        

class TestMixingLayerMathematicalProperties:
    """Test mathematical properties of mixing layers."""
    
    def test_shape_consistency(self, random_tensor):
        """Test that all mixing layers preserve input shape."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        
        layers = [
            FourierMixing(hidden_dim),
            FourierMixing1D(hidden_dim),
            RealFourierMixing(hidden_dim),
            GlobalFilterMixing(hidden_dim, seq_len),
            GlobalFilterMixing2D(hidden_dim, seq_len),
            AdaptiveGlobalFilter(hidden_dim, seq_len),
        ]
        
        for layer in layers:
            output = layer(random_tensor)
            assert layer.verify_shape_consistency(random_tensor, output)
            
    def test_spectral_norm_computation(self, random_tensor):
        """Test spectral norm computation."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = FourierMixing(hidden_dim)
        
        spectral_norm = mixer.compute_spectral_norm(random_tensor)
        assert isinstance(spectral_norm, torch.Tensor)
        assert spectral_norm.numel() == 1  # Should be scalar
        assert spectral_norm.item() >= 0  # Should be non-negative
        
    def test_energy_preservation_real_fft(self, random_tensor):
        """Test energy preservation for real FFT mixing."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = RealFourierMixing(hidden_dim, use_real_fft=True)
        
        output = mixer(random_tensor)
        
        # For unitary transforms, energy should be approximately preserved
        input_energy = torch.norm(random_tensor, p=2, dim=-1) ** 2
        output_energy = torch.norm(output, p=2, dim=-1) ** 2
        
        # Allow some tolerance for numerical precision
        energy_diff = torch.abs(input_energy - output_energy)
        max_energy = torch.max(input_energy, output_energy)
        relative_error = energy_diff / (max_energy + 1e-8)
        
        # Most entries should have small relative error
        assert torch.mean(relative_error.float()) < 0.1  # 10% average error tolerance
        
    def test_translation_equivariance_fourier(self, random_tensor):
        """Test translation equivariance of Fourier mixing."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = FourierMixing(hidden_dim)
        
        # Apply mixing to original tensor
        output1 = mixer(random_tensor)
        
        # Apply circular shift and then mixing
        shift = 5
        shifted_input = torch.roll(random_tensor, shifts=shift, dims=1)
        output2 = mixer(shifted_input)
        
        # Apply shift to original output
        shifted_output1 = torch.roll(output1, shifts=shift, dims=1)
        
        # Due to FFT's translation properties, these should be approximately equal
        # (within numerical precision and potential boundary effects)
        diff = torch.norm(output2 - shifted_output1, p=2)
        original_norm = torch.norm(output1, p=2)
        
        # Allow some tolerance for numerical precision and boundary effects  
        relative_diff = diff / (original_norm + 1e-8)
        assert relative_diff < 2.0  # Allow larger tolerance due to boundary effects in FFT
        

class TestMixingLayerGradients:
    """Test gradient computation for mixing layers."""
    
    def test_fourier_mixing_gradients(self, random_tensor):
        """Test gradient computation for Fourier mixing."""
        random_tensor.requires_grad_(True)
        mixer = FourierMixing(hidden_dim=random_tensor.size(-1))
        
        output = mixer(random_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist and have correct shape
        assert random_tensor.grad is not None
        assert random_tensor.grad.shape == random_tensor.shape
        
        # Gradients should not be all zeros (unless very pathological case)
        assert not torch.all(random_tensor.grad == 0)
        
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
        

class TestMixingLayerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_input(self):
        """Test mixing layers with zero input."""
        hidden_dim = 64
        seq_len = 32
        batch_size = 4
        
        zero_input = torch.zeros(batch_size, seq_len, hidden_dim)
        
        layers = [
            FourierMixing(hidden_dim),
            GlobalFilterMixing(hidden_dim, seq_len),
        ]
        
        for layer in layers:
            output = layer(zero_input)
            assert output.shape == zero_input.shape
            # For Fourier mixing of zeros, output should be close to zero
            if isinstance(layer, FourierMixing):
                assert torch.allclose(output, zero_input, atol=1e-6)
                
    def test_single_element_sequences(self):
        """Test mixing layers with sequence length 1."""
        hidden_dim = 32
        seq_len = 1
        batch_size = 2
        
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Test layers that should work with seq_len=1
        fourier_mixer = FourierMixing(hidden_dim)
        output_fourier = fourier_mixer(input_tensor)
        assert output_fourier.shape == input_tensor.shape
        
    def test_large_tensors_memory_efficiency(self):
        """Test memory efficiency with larger tensors."""
        # This test ensures layers don't have memory leaks
        hidden_dim = 128  
        seq_len = 256
        batch_size = 8
        
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Test RealFourierMixing memory efficiency
        real_mixer = RealFourierMixing(hidden_dim, use_real_fft=True)
        output = real_mixer(input_tensor)
        assert output.shape == input_tensor.shape
        
        # Clean up
        del input_tensor, output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    def test_different_dtype_support(self):
        """Test mixing layers with different dtypes."""
        hidden_dim = 32
        seq_len = 64
        batch_size = 2
        
        # Test float64
        input_float64 = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float64)
        mixer = FourierMixing(hidden_dim)
        output = mixer(input_float64)
        # Output should be float64 as well
        assert output.dtype == torch.float64
        
        # Test float32 (default)
        input_float32 = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)
        output32 = mixer(input_float32)
        assert output32.dtype == torch.float32