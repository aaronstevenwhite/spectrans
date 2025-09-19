"""Unit tests for Fourier Neural Operator (FNO) implementations."""

import pytest
import torch
import torch.nn as nn

from spectrans.layers.operators.fno import (
    FNOBlock,
    FourierNeuralOperator,
    SpectralConv1d,
    SpectralConv2d,
)


class TestSpectralConv1d:
    """Test 1D spectral convolution layer."""

    def test_spectral_conv1d_forward_shape(self):
        """Test output shape of SpectralConv1d."""
        batch_size = 4
        in_channels = 32
        out_channels = 64
        seq_len = 128
        modes = 16

        x = torch.randn(batch_size, in_channels, seq_len)
        conv = SpectralConv1d(in_channels, out_channels, modes)

        output = conv(x)

        assert output.shape == (batch_size, out_channels, seq_len)
        assert output.dtype == x.dtype

    def test_spectral_conv1d_mode_truncation(self):
        """Test mode truncation in spectral convolution."""
        batch_size = 2
        channels = 16
        seq_len = 64
        modes = 8  # Keep only first 8 modes

        x = torch.randn(batch_size, channels, seq_len)
        conv = SpectralConv1d(channels, channels, modes)

        output = conv(x)

        # Output should have same shape despite mode truncation
        assert output.shape == x.shape

    def test_spectral_conv1d_weights_initialization(self):
        """Test weight initialization."""
        in_channels = 32
        out_channels = 64
        modes = 16

        conv = SpectralConv1d(in_channels, out_channels, modes)

        # Check weight shape and properties
        assert conv.weights.shape == (in_channels, out_channels, modes, 2)

        # Weights should be initialized with small values
        assert conv.weights.abs().mean() < 1.0

    def test_spectral_conv1d_gradient_flow(self):
        """Test gradient flow through spectral convolution."""
        batch_size = 2
        channels = 16
        seq_len = 32
        modes = 8

        x = torch.randn(batch_size, channels, seq_len, requires_grad=True)
        conv = SpectralConv1d(channels, channels, modes)

        output = conv(x)
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        assert conv.weights.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(conv.weights.grad).all()


class TestSpectralConv2d:
    """Test 2D spectral convolution layer."""

    def test_spectral_conv2d_forward_shape(self):
        """Test output shape of SpectralConv2d."""
        batch_size = 2
        in_channels = 3
        out_channels = 64
        height = 32
        width = 32
        modes = (8, 8)

        x = torch.randn(batch_size, in_channels, height, width)
        conv = SpectralConv2d(in_channels, out_channels, modes)

        output = conv(x)

        assert output.shape == (batch_size, out_channels, height, width)
        assert output.dtype == x.dtype

    def test_spectral_conv2d_mode_truncation(self):
        """Test 2D mode truncation."""
        batch_size = 2
        channels = 16
        size = 64
        modes = (16, 16)  # Keep only first 16x16 modes

        x = torch.randn(batch_size, channels, size, size)
        conv = SpectralConv2d(channels, channels, modes)

        output = conv(x)

        assert output.shape == x.shape

    def test_spectral_conv2d_non_square(self):
        """Test with non-square inputs."""
        batch_size = 2
        channels = 8
        height = 32
        width = 64
        modes = (8, 16)

        x = torch.randn(batch_size, channels, height, width)
        conv = SpectralConv2d(channels, channels, modes)

        output = conv(x)

        assert output.shape == x.shape

    def test_spectral_conv2d_weights(self):
        """Test 2D weight properties."""
        in_channels = 16
        out_channels = 32
        modes = (8, 8)

        conv = SpectralConv2d(in_channels, out_channels, modes)

        assert conv.weights.shape == (in_channels, out_channels, 8, 8, 2)
        assert conv.modes1 == 8
        assert conv.modes2 == 8

    def test_spectral_conv2d_gradient_flow(self):
        """Test gradient flow through 2D spectral convolution."""
        batch_size = 2
        channels = 8
        height = 16
        width = 16
        modes = (4, 4)

        x = torch.randn(batch_size, channels, height, width, requires_grad=True)
        conv = SpectralConv2d(channels, channels, modes)

        output = conv(x)
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        assert conv.weights.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(conv.weights.grad).all()


class TestFourierNeuralOperator:
    """Test FourierNeuralOperator layer."""

    def test_fno_1d_forward_shape(self):
        """Test FNO with 1D data."""
        batch_size = 4
        seq_len = 128
        hidden_dim = 64
        modes = 16

        x = torch.randn(batch_size, seq_len, hidden_dim)
        fno = FourierNeuralOperator(hidden_dim=hidden_dim, modes=modes)

        output = fno(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_fno_2d_forward_shape(self):
        """Test FNO with 2D data."""
        batch_size = 2
        height = 32
        width = 32
        hidden_dim = 64
        modes = (8, 8)

        x = torch.randn(batch_size, height, width, hidden_dim)
        fno = FourierNeuralOperator(hidden_dim=hidden_dim, modes=modes)

        output = fno(x)

        assert output.shape == x.shape

    def test_fno_spectral_conv_only(self):
        """Test FNO with only spectral convolution."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 32
        modes = 16

        x = torch.randn(batch_size, seq_len, hidden_dim)
        fno = FourierNeuralOperator(
            hidden_dim=hidden_dim, modes=modes, use_spectral_conv=True, use_linear=False
        )

        assert fno.spectral_conv is not None
        assert fno.linear is None

        output = fno(x)
        assert output.shape == x.shape

    def test_fno_linear_only(self):
        """Test FNO with only linear transformation."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 32
        modes = 16

        x = torch.randn(batch_size, seq_len, hidden_dim)
        fno = FourierNeuralOperator(
            hidden_dim=hidden_dim, modes=modes, use_spectral_conv=False, use_linear=True
        )

        assert fno.spectral_conv is None
        assert fno.linear is not None

        output = fno(x)
        assert output.shape == x.shape

    def test_fno_both_transformations(self):
        """Test FNO with both spectral and linear transformations."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 32
        modes = 16

        x = torch.randn(batch_size, seq_len, hidden_dim)
        fno = FourierNeuralOperator(
            hidden_dim=hidden_dim, modes=modes, use_spectral_conv=True, use_linear=True
        )

        assert fno.spectral_conv is not None
        assert fno.linear is not None

        output = fno(x)
        assert output.shape == x.shape

    def test_fno_invalid_configuration(self):
        """Test that invalid configuration raises error."""
        with pytest.raises(ValueError):
            FourierNeuralOperator(
                hidden_dim=64, modes=16, use_spectral_conv=False, use_linear=False
            )

    def test_fno_activations(self):
        """Test different activation functions."""
        batch_size = 2
        seq_len = 32
        hidden_dim = 16
        modes = 8

        activations = ["gelu", "relu", "tanh", "silu"]

        for activation in activations:
            x = torch.randn(batch_size, seq_len, hidden_dim)
            fno = FourierNeuralOperator(hidden_dim=hidden_dim, modes=modes, activation=activation)

            output = fno(x)
            assert output.shape == x.shape


class TestFNOBlock:
    """Test complete FNO block with residual connections."""

    def test_fno_block_forward_shape(self):
        """Test FNO block output shape."""
        batch_size = 4
        seq_len = 128
        hidden_dim = 64
        modes = 16

        x = torch.randn(batch_size, seq_len, hidden_dim)
        block = FNOBlock(hidden_dim=hidden_dim, modes=modes)

        output = block(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_fno_block_with_ffn(self):
        """Test FNO block with feedforward network."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 32
        modes = 16
        mlp_ratio = 2.0

        x = torch.randn(batch_size, seq_len, hidden_dim)
        block = FNOBlock(hidden_dim=hidden_dim, modes=modes, mlp_ratio=mlp_ratio)

        assert block.ffn is not None
        assert block.norm2 is not None

        output = block(x)
        assert output.shape == x.shape

    def test_fno_block_without_ffn(self):
        """Test FNO block without feedforward network."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 32
        modes = 16

        x = torch.randn(batch_size, seq_len, hidden_dim)
        block = FNOBlock(hidden_dim=hidden_dim, modes=modes, mlp_ratio=0.0)  # No FFN

        assert block.ffn is None
        assert block.norm2 is None

        output = block(x)
        assert output.shape == x.shape

    def test_fno_block_dropout(self):
        """Test FNO block with dropout."""
        batch_size = 2
        seq_len = 32
        hidden_dim = 16
        modes = 8
        dropout = 0.5

        x = torch.randn(batch_size, seq_len, hidden_dim)
        block = FNOBlock(hidden_dim=hidden_dim, modes=modes, dropout=dropout)

        # Test training mode
        block.train()
        output1 = block(x)
        output2 = block(x)

        # Outputs should differ due to dropout
        assert not torch.allclose(output1, output2)

        # Test eval mode
        block.eval()
        output_eval1 = block(x)
        output_eval2 = block(x)

        # Outputs should be identical
        assert torch.allclose(output_eval1, output_eval2)

    def test_fno_block_norm_types(self):
        """Test different normalization types."""
        batch_size = 2
        seq_len = 32
        hidden_dim = 16
        modes = 8

        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Test layer norm
        block_ln = FNOBlock(hidden_dim=hidden_dim, modes=modes, norm_type="layernorm")
        assert isinstance(block_ln.norm1, nn.LayerNorm)
        output_ln = block_ln(x)
        assert output_ln.shape == x.shape

        # Test batch norm
        block_bn = FNOBlock(hidden_dim=hidden_dim, modes=modes, norm_type="batchnorm")
        assert isinstance(block_bn.norm1, nn.BatchNorm1d)
        output_bn = block_bn(x)
        assert output_bn.shape == x.shape

    def test_fno_block_invalid_norm(self):
        """Test invalid normalization type."""
        with pytest.raises(ValueError):
            FNOBlock(hidden_dim=64, modes=16, norm_type="invalid")


class TestFNOGradients:
    """Test gradient flow through FNO components."""

    def test_fno_gradient_flow(self):
        """Test gradient flow through FNO layer."""
        batch_size = 2
        seq_len = 32
        hidden_dim = 16
        modes = 8

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
        fno = FourierNeuralOperator(hidden_dim=hidden_dim, modes=modes)

        output = fno(x)
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Check parameter gradients
        for _name, param in fno.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()

    def test_fno_block_gradient_flow(self):
        """Test gradient flow through FNO block."""
        batch_size = 2
        seq_len = 32
        hidden_dim = 16
        modes = 8

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
        block = FNOBlock(hidden_dim=hidden_dim, modes=modes, mlp_ratio=2.0)

        output = block(x)
        loss = output.mean()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Check all parameter gradients
        for name, param in block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


class TestFNOEdgeCases:
    """Test edge cases for FNO implementations."""

    def test_fno_zero_input(self):
        """Test FNO with zero input."""
        batch_size = 2
        seq_len = 32
        hidden_dim = 16
        modes = 8

        x = torch.zeros(batch_size, seq_len, hidden_dim)
        fno = FourierNeuralOperator(hidden_dim=hidden_dim, modes=modes)

        output = fno(x)

        # Output may not be zero due to activation, but should be bounded
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_fno_single_batch(self):
        """Test with batch size 1."""
        batch_size = 1
        seq_len = 32
        hidden_dim = 16

        x = torch.randn(batch_size, seq_len, hidden_dim)
        fno = FourierNeuralOperator(hidden_dim=hidden_dim, modes=8)

        output = fno(x)
        assert output.shape == x.shape

    def test_fno_power_of_two_dimensions(self):
        """Test with power-of-two dimensions (optimal for FFT)."""
        batch_size = 2
        seq_len = 64  # Power of 2
        hidden_dim = 32  # Power of 2
        modes = 16

        x = torch.randn(batch_size, seq_len, hidden_dim)
        block = FNOBlock(hidden_dim=hidden_dim, modes=modes)

        output = block(x)
        assert output.shape == x.shape

    def test_fno_very_few_modes(self):
        """Test with very few Fourier modes."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 64
        modes = 2  # Very aggressive truncation

        x = torch.randn(batch_size, seq_len, hidden_dim)
        fno = FourierNeuralOperator(hidden_dim=hidden_dim, modes=modes)

        output = fno(x)

        # Should still produce valid output
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_fno_dtype_preservation(self):
        """Test dtype preservation."""
        batch_size = 2
        seq_len = 32
        hidden_dim = 16

        # Test float32
        x_f32 = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)
        fno = FourierNeuralOperator(hidden_dim=hidden_dim, modes=8)
        output_f32 = fno(x_f32)
        assert output_f32.dtype == torch.float32

        # Test float64
        x_f64 = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float64)
        output_f64 = fno(x_f64)
        assert output_f64.dtype == torch.float64
