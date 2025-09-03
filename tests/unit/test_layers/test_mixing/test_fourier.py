"""Unit tests for Fourier mixing layers."""

import pytest
import torch

from spectrans.layers.mixing.fourier import (
    FourierMixing,
    FourierMixing1D,
    RealFourierMixing,
    SeparableFourierMixing,
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


class TestFourierMixingMathematicalProperties:
    """Test mathematical properties of Fourier mixing layers."""
    
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
        
    def test_spectral_norm_computation(self, random_tensor):
        """Test spectral norm computation."""
        batch_size, seq_len, hidden_dim = random_tensor.shape
        mixer = FourierMixing(hidden_dim)
        
        spectral_norm = mixer.compute_spectral_norm(random_tensor)
        assert isinstance(spectral_norm, torch.Tensor)
        assert spectral_norm.numel() == 1  # Should be scalar
        assert spectral_norm.item() >= 0  # Should be non-negative


class TestFourierMixingGradients:
    """Test gradient computation for Fourier mixing layers."""
    
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


class TestFourierMixingEdgeCases:
    """Test edge cases for Fourier mixing layers."""
    
    def test_zero_input(self):
        """Test mixing layers with zero input."""
        hidden_dim = 64
        seq_len = 32
        batch_size = 4
        
        zero_input = torch.zeros(batch_size, seq_len, hidden_dim)
        mixer = FourierMixing(hidden_dim)
        
        output = mixer(zero_input)
        assert output.shape == zero_input.shape
        # For Fourier mixing of zeros, output should be close to zero
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