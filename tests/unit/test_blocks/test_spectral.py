"""Unit tests for spectral transformer blocks.

Tests for spectral transformer block implementations including FNet, GFNet,
AFNO, Spectral Attention, LST, Wavelet, and FNO blocks.
"""

import torch
import torch.nn as nn

from spectrans.blocks.spectral import (
    AFNOBlock,
    FNetBlock,
    FNO2DBlock,
    FNOBlock,
    GFNetBlock,
    LSTBlock,
    SpectralAttentionBlock,
    WaveletBlock,
)


class TestFNetBlock:
    """Test FNetBlock."""

    def test_forward_shape(self):
        """Test FNetBlock forward pass."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        block = FNetBlock(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=256,
            dropout=0.1,
        )

        output = block(x)
        assert output.shape == x.shape

    def test_configuration(self):
        """Test FNetBlock configuration options."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Test with different FFN dimensions
        block = FNetBlock(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=512,
            activation="relu",
            dropout=0.2,
            norm_eps=1e-6,
        )

        output = block(x)
        assert output.shape == x.shape


class TestGFNetBlock:
    """Test GFNetBlock."""

    def test_forward_shape(self):
        """Test GFNetBlock forward pass."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        block = GFNetBlock(
            hidden_dim=hidden_dim,
            sequence_length=seq_len,
            ffn_hidden_dim=256,
            dropout=0.1,
        )

        output = block(x)
        assert output.shape == x.shape

    def test_filter_activations(self):
        """Test GFNetBlock with different filter activations."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        for filter_activation in ["sigmoid", "tanh", "identity"]:
            block = GFNetBlock(
                hidden_dim=hidden_dim,
                sequence_length=seq_len,
                filter_activation=filter_activation,
                filter_init_std=0.02,
            )

            output = block(x)
            assert output.shape == x.shape


class TestAFNOBlock:
    """Test AFNOBlock."""

    def test_forward_shape(self):
        """Test AFNOBlock forward pass."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        block = AFNOBlock(
            hidden_dim=hidden_dim,
            sequence_length=seq_len,
            modes=32,
            mlp_hidden_dim=128,
            ffn_hidden_dim=256,
            dropout=0.1,
        )

        output = block(x)
        assert output.shape == x.shape

    def test_mode_truncation(self):
        """Test AFNOBlock with different mode truncation."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Test with different number of modes
        for modes in [16, 32, 64]:
            block = AFNOBlock(
                hidden_dim=hidden_dim,
                sequence_length=seq_len,
                modes=modes,
            )

            output = block(x)
            assert output.shape == x.shape


class TestSpectralAttentionBlock:
    """Test SpectralAttentionBlock."""

    def test_forward_shape(self):
        """Test SpectralAttentionBlock forward pass."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        block = SpectralAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_features=256,
            kernel_type="gaussian",
            ffn_hidden_dim=256,
            dropout=0.1,
        )

        output = block(x)
        assert output.shape == x.shape

    def test_kernel_types(self):
        """Test SpectralAttentionBlock with different kernel types."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        for kernel_type in ["gaussian", "softmax"]:
            block = SpectralAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=8,
                kernel_type=kernel_type,
            )

            output = block(x)
            assert output.shape == x.shape

    def test_head_configurations(self):
        """Test different head configurations."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        for num_heads in [1, 4, 8, 16]:
            if hidden_dim % num_heads == 0:  # Valid configuration
                block = SpectralAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_features=128,
                )

                output = block(x)
                assert output.shape == x.shape


class TestLSTBlock:
    """Test LSTBlock."""

    def test_forward_shape(self):
        """Test LSTBlock forward pass."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        block = LSTBlock(
            hidden_dim=hidden_dim,
            num_heads=8,
            transform_type="dct",
            use_scaling=True,
            ffn_hidden_dim=256,
            dropout=0.1,
        )

        output = block(x)
        assert output.shape == x.shape

    def test_transform_types(self):
        """Test LSTBlock with different transform types."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        for transform_type in ["dct", "dst", "hadamard"]:
            block = LSTBlock(
                hidden_dim=hidden_dim,
                num_heads=8,
                transform_type=transform_type,
            )

            output = block(x)
            assert output.shape == x.shape

    def test_scaling_options(self):
        """Test with and without learnable scaling."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        for use_scaling in [True, False]:
            block = LSTBlock(
                hidden_dim=hidden_dim,
                num_heads=8,
                use_scaling=use_scaling,
            )

            output = block(x)
            assert output.shape == x.shape


class TestWaveletBlock:
    """Test WaveletBlock."""

    def test_forward_shape(self):
        """Test WaveletBlock forward pass."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        block = WaveletBlock(
            hidden_dim=hidden_dim,
            wavelet="db4",
            levels=2,
            ffn_hidden_dim=256,
            dropout=0.1,
        )

        output = block(x)
        assert output.shape == x.shape

    def test_wavelet_types(self):
        """Test WaveletBlock with different wavelet types."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        for wavelet in ["db1", "db2", "db4", "sym3"]:
            block = WaveletBlock(
                hidden_dim=hidden_dim,
                wavelet=wavelet,
                levels=2,
            )

            output = block(x)
            assert output.shape == x.shape

    def test_decomposition_levels(self):
        """Test different decomposition levels."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        for levels in [1, 2, 3]:
            block = WaveletBlock(
                hidden_dim=hidden_dim,
                wavelet="db4",
                levels=levels,
            )

            output = block(x)
            assert output.shape == x.shape


class TestFNOBlock:
    """Test FNOBlock."""

    def test_forward_shape(self):
        """Test FNOBlock forward pass."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        block = FNOBlock(
            hidden_dim=hidden_dim,
            modes=16,
            num_layers=1,
            ffn_hidden_dim=256,
            dropout=0.1,
        )

        output = block(x)
        assert output.shape == x.shape

    def test_multiple_layers(self):
        """Test FNOBlock with multiple FNO layers."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        for num_layers in [1, 2, 3]:
            block = FNOBlock(
                hidden_dim=hidden_dim,
                modes=16,
                num_layers=num_layers,
            )

            output = block(x)
            assert output.shape == x.shape

    def test_mode_configurations(self):
        """Test different mode configurations."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        for modes in [8, 16, 32]:
            block = FNOBlock(
                hidden_dim=hidden_dim,
                modes=modes,
            )

            output = block(x)
            assert output.shape == x.shape


class TestFNO2DBlock:
    """Test FNO2DBlock."""

    def test_forward_shape(self):
        """Test FNO2DBlock forward pass."""
        batch_size, height, width, hidden_dim = 2, 32, 32, 64
        x = torch.randn(batch_size, height, width, hidden_dim)

        block = FNO2DBlock(
            hidden_dim=hidden_dim,
            modes_h=8,
            modes_w=8,
            num_layers=1,
            ffn_hidden_dim=256,
            dropout=0.1,
        )

        output = block(x)
        assert output.shape == x.shape

    def test_asymmetric_modes(self):
        """Test with different modes for height and width."""
        batch_size, height, width, hidden_dim = 2, 64, 32, 64
        x = torch.randn(batch_size, height, width, hidden_dim)

        block = FNO2DBlock(
            hidden_dim=hidden_dim,
            modes_h=16,
            modes_w=8,
        )

        output = block(x)
        assert output.shape == x.shape


class TestGradientFlow:
    """Test gradient flow through spectral blocks."""

    def test_gradient_flow_all_blocks(self):
        """Test gradient flow through all spectral blocks."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        blocks_to_test = [
            FNetBlock(hidden_dim=hidden_dim),
            GFNetBlock(hidden_dim=hidden_dim, sequence_length=seq_len),
            AFNOBlock(hidden_dim=hidden_dim, sequence_length=seq_len),
            SpectralAttentionBlock(hidden_dim=hidden_dim, num_heads=8),
            LSTBlock(hidden_dim=hidden_dim, num_heads=8),
            WaveletBlock(hidden_dim=hidden_dim),
            FNOBlock(hidden_dim=hidden_dim),
        ]

        for block in blocks_to_test:
            x_copy = x.clone().detach().requires_grad_(True)
            output = block(x_copy)
            loss = output.sum()
            loss.backward()

            assert x_copy.grad is not None
            assert not torch.isnan(x_copy.grad).any()
            assert not torch.isinf(x_copy.grad).any()

    def test_gradient_flow_2d(self):
        """Test gradient flow through 2D blocks."""
        batch_size, height, width, hidden_dim = 2, 32, 32, 64
        x = torch.randn(batch_size, height, width, hidden_dim, requires_grad=True)

        block = FNO2DBlock(
            hidden_dim=hidden_dim,
            modes_h=8,
            modes_w=8,
        )

        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestBlockComposition:
    """Test composing spectral blocks."""

    def test_sequential_spectral_blocks(self):
        """Test sequential composition of spectral blocks."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Create a sequence of different spectral blocks
        blocks = nn.Sequential(
            FNetBlock(hidden_dim=hidden_dim),
            GFNetBlock(hidden_dim=hidden_dim, sequence_length=seq_len),
            AFNOBlock(hidden_dim=hidden_dim, sequence_length=seq_len),
        )

        output = blocks(x)
        assert output.shape == x.shape

    def test_mixed_spectral_blocks(self):
        """Test mixing different spectral strategies."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Alternate between frequency and attention-based blocks
        blocks = nn.ModuleList(
            [
                FNetBlock(hidden_dim=hidden_dim),
                SpectralAttentionBlock(hidden_dim=hidden_dim, num_heads=8),
                WaveletBlock(hidden_dim=hidden_dim, levels=2),
                LSTBlock(hidden_dim=hidden_dim, num_heads=8),
            ]
        )

        h = x
        for block in blocks:
            h = block(h)
            assert h.shape == x.shape

        assert h.shape == x.shape
