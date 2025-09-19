"""Unit tests for AFNO (Adaptive Fourier Neural Operator) mixing layer."""

import torch

from spectrans.layers.mixing.afno import AFNOMixing


class TestAFNOMixing:
    """Test AFNO mixing layer basic functionality."""

    def test_afno_forward_shape(self):
        """Test that AFNO preserves tensor shape."""
        batch_size = 4
        seq_len = 128
        hidden_dim = 256

        x = torch.randn(batch_size, seq_len, hidden_dim)
        layer = AFNOMixing(hidden_dim=hidden_dim, max_sequence_length=seq_len)

        output = layer(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_afno_with_mode_truncation(self):
        """Test AFNO with custom mode truncation."""
        batch_size = 2
        seq_len = 256
        hidden_dim = 512

        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Keep only 25% of modes
        layer = AFNOMixing(
            hidden_dim=hidden_dim,
            max_sequence_length=seq_len,
            modes_seq=64,  # 25% of 256
            modes_hidden=128,  # 25% of 512
        )

        output = layer(x)
        assert output.shape == x.shape

    def test_afno_default_mode_selection(self):
        """Test AFNO with default mode selection."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 256

        x = torch.randn(batch_size, seq_len, hidden_dim)
        layer = AFNOMixing(hidden_dim=hidden_dim, max_sequence_length=seq_len)

        # Check default mode selection
        assert layer.modes_seq == seq_len // 2  # Default is half
        assert layer.modes_hidden == min(hidden_dim // 2, hidden_dim // 2 + 1)

        output = layer(x)
        assert output.shape == x.shape

    def test_afno_with_padding(self):
        """Test AFNO with sequences shorter than max_sequence_length."""
        batch_size = 2
        actual_seq_len = 100
        max_seq_len = 128
        hidden_dim = 256

        x = torch.randn(batch_size, actual_seq_len, hidden_dim)
        layer = AFNOMixing(hidden_dim=hidden_dim, max_sequence_length=max_seq_len)

        output = layer(x)

        # Output should match input shape (padding is internal)
        assert output.shape == x.shape

    def test_afno_mlp_ratio(self):
        """Test AFNO with different MLP expansion ratios."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 128

        for mlp_ratio in [1.0, 2.0, 4.0]:
            x = torch.randn(batch_size, seq_len, hidden_dim)
            layer = AFNOMixing(
                hidden_dim=hidden_dim, max_sequence_length=seq_len, mlp_ratio=mlp_ratio
            )

            output = layer(x)
            assert output.shape == x.shape

    def test_afno_activations(self):
        """Test AFNO with different activation functions."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 128

        activations = ["gelu", "relu", "silu", "tanh"]

        for activation in activations:
            x = torch.randn(batch_size, seq_len, hidden_dim)
            layer = AFNOMixing(
                hidden_dim=hidden_dim, max_sequence_length=seq_len, activation=activation
            )

            output = layer(x)
            assert output.shape == x.shape

    def test_afno_dropout(self):
        """Test AFNO with dropout."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 128
        dropout = 0.5

        x = torch.randn(batch_size, seq_len, hidden_dim)
        layer = AFNOMixing(hidden_dim=hidden_dim, max_sequence_length=seq_len, dropout=dropout)

        # Test training mode
        layer.train()
        output1 = layer(x)
        output2 = layer(x)

        # Outputs should differ due to dropout
        assert not torch.allclose(output1, output2)

        # Test eval mode
        layer.eval()
        output_eval1 = layer(x)
        output_eval2 = layer(x)

        # Outputs should be identical in eval mode
        assert torch.allclose(output_eval1, output_eval2)


class TestAFNOMixingGradients:
    """Test gradient flow through AFNO mixing layer."""

    def test_afno_gradient_flow(self):
        """Test that gradients flow through AFNO layer."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 128

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
        layer = AFNOMixing(hidden_dim=hidden_dim, max_sequence_length=seq_len)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Check input gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.all(x.grad == 0)

        # Check parameter gradients
        for name, param in layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.shape == param.shape
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_afno_gradient_stability(self):
        """Test gradient stability with different configurations."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 128

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        # Test with aggressive mode truncation
        layer = AFNOMixing(
            hidden_dim=hidden_dim,
            max_sequence_length=seq_len,
            modes_seq=8,  # Very few modes
            modes_hidden=16,
        )

        output = layer(x)
        loss = output.mean()  # Use mean to avoid large gradients
        loss.backward()

        # Gradients should be finite
        assert torch.isfinite(x.grad).all()
        for param in layer.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()


class TestAFNOMixingMathematicalProperties:
    """Test mathematical properties of AFNO mixing."""

    def test_afno_residual_connection(self):
        """Test that residual connection is properly applied."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 128

        x = torch.randn(batch_size, seq_len, hidden_dim)
        layer = AFNOMixing(hidden_dim=hidden_dim, max_sequence_length=seq_len)

        output = layer(x)

        # With residual connection, output should not be too far from input
        # (unless the Fourier transformation is very aggressive)
        relative_change = torch.norm(output - x) / torch.norm(x)
        assert relative_change < 10.0  # Reasonable bound

    def test_afno_spectral_processing(self):
        """Test spectral processing preserves basic properties."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 128

        # Create input with known frequency content
        t = torch.linspace(0, 2 * torch.pi, seq_len).unsqueeze(0).unsqueeze(-1)
        x = torch.sin(2 * t)  # Single frequency component
        x = x.expand(batch_size, seq_len, hidden_dim)

        layer = AFNOMixing(
            hidden_dim=hidden_dim,
            max_sequence_length=seq_len,
            modes_seq=32,  # Keep enough modes to preserve signal
        )

        output = layer(x)

        # Output should still be bounded (no explosion)
        assert torch.isfinite(output).all()
        assert output.abs().max() < 100  # Reasonable bound

    def test_afno_mode_truncation_effect(self):
        """Test effect of mode truncation on output."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 256

        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Layer with many modes (less truncation)
        layer_many = AFNOMixing(
            hidden_dim=hidden_dim,
            max_sequence_length=seq_len,
            modes_seq=seq_len // 2,
            modes_hidden=hidden_dim // 2,
        )

        # Layer with few modes (more truncation)
        layer_few = AFNOMixing(
            hidden_dim=hidden_dim, max_sequence_length=seq_len, modes_seq=8, modes_hidden=16
        )

        output_many = layer_many(x)
        output_few = layer_few(x)

        # Both should produce valid outputs
        assert output_many.shape == x.shape
        assert output_few.shape == x.shape

        # Output with fewer modes should be "smoother" (less high-frequency content)
        # This is hard to test directly, but we can check they're different
        assert not torch.allclose(output_many, output_few)


class TestAFNOMixingEdgeCases:
    """Test edge cases for AFNO mixing layer."""

    def test_afno_zero_input(self):
        """Test AFNO with zero input."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 128

        x = torch.zeros(batch_size, seq_len, hidden_dim)
        layer = AFNOMixing(hidden_dim=hidden_dim, max_sequence_length=seq_len)

        output = layer(x)

        # Output should be zeros (due to residual connection with zero transform)
        assert output.shape == x.shape
        # Due to layer norm, output might not be exactly zero
        assert output.abs().max() < 1.0  # Should be small

    def test_afno_single_batch(self):
        """Test AFNO with batch size 1."""
        batch_size = 1
        seq_len = 64
        hidden_dim = 128

        x = torch.randn(batch_size, seq_len, hidden_dim)
        layer = AFNOMixing(hidden_dim=hidden_dim, max_sequence_length=seq_len)

        output = layer(x)
        assert output.shape == x.shape

    def test_afno_power_of_two_dimensions(self):
        """Test AFNO with power-of-two dimensions (optimal for FFT)."""
        batch_size = 2
        seq_len = 128  # Power of 2
        hidden_dim = 256  # Power of 2

        x = torch.randn(batch_size, seq_len, hidden_dim)
        layer = AFNOMixing(hidden_dim=hidden_dim, max_sequence_length=seq_len)

        output = layer(x)
        assert output.shape == x.shape

    def test_afno_non_power_of_two_dimensions(self):
        """Test AFNO with non-power-of-two dimensions."""
        batch_size = 3
        seq_len = 100  # Not power of 2
        hidden_dim = 384  # Not power of 2

        x = torch.randn(batch_size, seq_len, hidden_dim)
        layer = AFNOMixing(
            hidden_dim=hidden_dim,
            max_sequence_length=150,  # Larger than actual sequence
        )

        output = layer(x)
        assert output.shape == x.shape

    def test_afno_dtype_preservation(self):
        """Test AFNO preserves input dtype."""
        batch_size = 2
        seq_len = 64
        hidden_dim = 128

        # Test float32
        x_f32 = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)
        layer = AFNOMixing(hidden_dim=hidden_dim, max_sequence_length=seq_len)
        output_f32 = layer(x_f32)
        assert output_f32.dtype == torch.float32

        # Test float64
        x_f64 = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float64)
        output_f64 = layer(x_f64)
        assert output_f64.dtype == torch.float64
