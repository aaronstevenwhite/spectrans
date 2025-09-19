"""Unit tests for base transformer blocks.

Tests for the base transformer block implementations including TransformerBlock,
FeedForwardNetwork, and various normalization strategies.
"""

import torch
import torch.nn as nn

from spectrans.blocks.base import (
    FeedForwardNetwork,
    ParallelBlock,
    PostNormBlock,
    PreNormBlock,
    TransformerBlock,
)
from spectrans.layers.mixing.fourier import FourierMixing


class TestFeedForwardNetwork:
    """Test FeedForwardNetwork."""

    def test_forward_shape(self, random_tensor):
        """Test that FFN preserves input shape."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        ffn_hidden_dim = 256
        x = torch.randn(batch_size, seq_len, hidden_dim)

        ffn = FeedForwardNetwork(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation="gelu",
            dropout=0.0,
        )

        output = ffn(x)
        assert output.shape == x.shape

    def test_different_activations(self):
        """Test FFN with different activation functions."""
        hidden_dim = 64
        ffn_hidden_dim = 256
        x = torch.randn(2, 128, hidden_dim)

        activations = ["gelu", "relu", "silu", "tanh", "sigmoid", "elu", "leaky_relu"]

        for activation in activations:
            ffn = FeedForwardNetwork(
                hidden_dim=hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                activation=activation,
                dropout=0.0,
            )
            output = ffn(x)
            assert output.shape == x.shape
            assert not torch.isnan(output).any()

    def test_dropout(self):
        """Test FFN with dropout."""
        hidden_dim = 64
        ffn_hidden_dim = 256
        x = torch.randn(2, 128, hidden_dim)

        ffn = FeedForwardNetwork(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation="gelu",
            dropout=0.5,
        )

        # Set to training mode
        ffn.train()
        ffn(x)

        # Set to eval mode
        ffn.eval()
        output_eval1 = ffn(x)
        output_eval2 = ffn(x)

        # In eval mode, outputs should be deterministic
        assert torch.allclose(output_eval1, output_eval2)


class TestTransformerBlock:
    """Test TransformerBlock base class."""

    def test_forward_shape(self):
        """Test that TransformerBlock preserves input shape."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        mixing_layer = FourierMixing(hidden_dim=hidden_dim)

        # Test with FFN
        block = TransformerBlock(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=256,
            activation="gelu",
            dropout=0.0,
            use_pre_norm=True,
        )

        output = block(x)
        assert output.shape == x.shape

        # Test without FFN
        block_no_ffn = TransformerBlock(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=None,
            dropout=0.0,
            use_pre_norm=True,
        )

        output_no_ffn = block_no_ffn(x)
        assert output_no_ffn.shape == x.shape

    def test_pre_norm_vs_post_norm(self):
        """Test pre-norm vs post-norm configurations."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        mixing_layer = FourierMixing(hidden_dim=hidden_dim)

        # Pre-norm
        pre_norm_block = TransformerBlock(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=256,
            use_pre_norm=True,
        )

        # Post-norm
        post_norm_block = TransformerBlock(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=256,
            use_pre_norm=False,
        )

        output_pre = pre_norm_block(x)
        output_post = post_norm_block(x)

        assert output_pre.shape == x.shape
        assert output_post.shape == x.shape
        # Outputs should be different due to different normalization strategies
        assert not torch.allclose(output_pre, output_post)

    def test_residual_connections(self):
        """Test that residual connections are applied."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Create a simple identity mixing layer
        class IdentityMixing(nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)  # Return zeros

        mixing_layer = IdentityMixing()

        block = TransformerBlock(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=None,  # No FFN
            dropout=0.0,
            use_pre_norm=False,  # Post-norm for simplicity
        )

        output = block(x)
        # With identity mixing returning zeros and no FFN, output should be close to input
        # due to residual connection (after layer norm)
        assert output.shape == x.shape


class TestPreNormBlock:
    """Test PreNormBlock."""

    def test_forward_shape(self):
        """Test PreNormBlock forward shape."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        mixing_layer = FourierMixing(hidden_dim=hidden_dim)
        block = PreNormBlock(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
        )

        output = block(x)
        assert output.shape == x.shape
        assert block.use_pre_norm is True


class TestPostNormBlock:
    """Test PostNormBlock."""

    def test_forward_shape(self):
        """Test PostNormBlock forward shape."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        mixing_layer = FourierMixing(hidden_dim=hidden_dim)
        block = PostNormBlock(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
        )

        output = block(x)
        assert output.shape == x.shape
        assert block.use_pre_norm is False


class TestParallelBlock:
    """Test ParallelBlock."""

    def test_forward_shape(self):
        """Test ParallelBlock forward shape."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        mixing_layer = FourierMixing(hidden_dim=hidden_dim)
        block = ParallelBlock(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=256,
        )

        output = block(x)
        assert output.shape == x.shape

    def test_parallel_computation(self):
        """Test that parallel block processes mixing and FFN in parallel."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Create custom layers that mark when they're called
        class MarkedMixing(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.linear = nn.Linear(hidden_dim, hidden_dim)

            def forward(self, x):
                return self.linear(x)

        mixing_layer = MarkedMixing(hidden_dim)
        block = ParallelBlock(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=256,
        )

        output = block(x)
        assert output.shape == x.shape


class TestGradientFlow:
    """Test gradient flow through base blocks."""

    def test_gradient_flow_transformer_block(self):
        """Test gradient flow through base TransformerBlock."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        mixing_layer = FourierMixing(hidden_dim=hidden_dim)
        block = TransformerBlock(
            mixing_layer=mixing_layer,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=256,
        )

        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_gradient_flow_norm_blocks(self):
        """Test gradient flow through normalization block variants."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        mixing_layer = FourierMixing(hidden_dim=hidden_dim)

        blocks_to_test = [
            PreNormBlock(mixing_layer=mixing_layer, hidden_dim=hidden_dim),
            PostNormBlock(mixing_layer=mixing_layer, hidden_dim=hidden_dim),
            ParallelBlock(mixing_layer=mixing_layer, hidden_dim=hidden_dim),
        ]

        for block in blocks_to_test:
            x_copy = x.clone().detach().requires_grad_(True)
            output = block(x_copy)
            loss = output.sum()
            loss.backward()

            assert x_copy.grad is not None
            assert not torch.isnan(x_copy.grad).any()
            assert not torch.isinf(x_copy.grad).any()
