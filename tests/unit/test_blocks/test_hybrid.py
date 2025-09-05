"""Unit tests for hybrid transformer blocks.

Tests for hybrid transformer block implementations that combine different
mixing strategies including alternating, adaptive, multiscale, and cascade blocks.
"""

import torch
import torch.nn as nn

from spectrans.blocks.hybrid import (
    AdaptiveBlock,
    AlternatingBlock,
    CascadeBlock,
    MultiscaleBlock,
)
from spectrans.layers.attention.spectral import SpectralAttention
from spectrans.layers.mixing.fourier import FourierMixing
from spectrans.layers.mixing.global_filter import GlobalFilterMixing


class TestAlternatingBlock:
    """Test AlternatingBlock."""

    def test_forward_shape(self):
        """Test AlternatingBlock forward pass."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layer1 = FourierMixing(hidden_dim=hidden_dim)
        layer2 = GlobalFilterMixing(
            hidden_dim=hidden_dim,
            sequence_length=seq_len,
        )

        block = AlternatingBlock(
            layer1=layer1,
            layer2=layer2,
            hidden_dim=hidden_dim,
            use_layer1=True,
        )

        output = block(x)
        assert output.shape == x.shape

    def test_layer_switching(self):
        """Test switching between layers."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layer1 = FourierMixing(hidden_dim=hidden_dim)
        layer2 = GlobalFilterMixing(
            hidden_dim=hidden_dim,
            sequence_length=seq_len,
        )

        block = AlternatingBlock(
            layer1=layer1,
            layer2=layer2,
            hidden_dim=hidden_dim,
            use_layer1=True,
        )

        # Test with layer1
        output1 = block(x)
        assert output1.shape == x.shape

        # Switch to layer2
        block.set_layer(use_layer1=False)
        output2 = block(x)
        assert output2.shape == x.shape

        # Outputs should be different
        assert not torch.allclose(output1, output2, rtol=1e-3)

    def test_alternating_pattern(self):
        """Test creating alternating pattern of blocks."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layer1 = FourierMixing(hidden_dim=hidden_dim)
        layer2 = SpectralAttention(hidden_dim=hidden_dim, num_heads=8)

        # Create alternating blocks
        blocks = nn.ModuleList([
            AlternatingBlock(
                layer1=layer1,
                layer2=layer2,
                hidden_dim=hidden_dim,
                use_layer1=(i % 2 == 0),
            )
            for i in range(4)
        ])

        h = x
        for block in blocks:
            h = block(h)

        assert h.shape == x.shape


class TestAdaptiveBlock:
    """Test AdaptiveBlock."""

    def test_forward_shape_soft_gating(self):
        """Test AdaptiveBlock with soft gating."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len),
        ]

        block = AdaptiveBlock(
            layers=layers,
            hidden_dim=hidden_dim,
            gate_type="soft",
        )

        output = block(x)
        assert output.shape == x.shape

    def test_forward_shape_hard_gating(self):
        """Test AdaptiveBlock with hard gating."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len),
        ]

        block = AdaptiveBlock(
            layers=layers,
            hidden_dim=hidden_dim,
            gate_type="hard",
        )

        output = block(x)
        assert output.shape == x.shape

    def test_multiple_layers(self):
        """Test adaptive selection with multiple layers."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len),
            SpectralAttention(hidden_dim=hidden_dim, num_heads=8),
        ]

        block = AdaptiveBlock(
            layers=layers,
            hidden_dim=hidden_dim,
            gate_type="soft",
        )

        output = block(x)
        assert output.shape == x.shape

    def test_gate_initialization(self):
        """Test that gates are properly initialized."""
        hidden_dim = 64

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=128),
        ]

        block = AdaptiveBlock(
            layers=layers,
            hidden_dim=hidden_dim,
            gate_type="soft",
        )

        # Check gate parameters
        assert block.gate.weight.shape == (len(layers), hidden_dim)
        assert block.gate.bias.shape == (len(layers),)


class TestMultiscaleBlock:
    """Test MultiscaleBlock."""

    def test_forward_shape_add_fusion(self):
        """Test MultiscaleBlock with additive fusion."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len),
        ]

        block = MultiscaleBlock(
            layers=layers,
            hidden_dim=hidden_dim,
            fusion_type="add",
        )

        output = block(x)
        assert output.shape == x.shape

    def test_forward_shape_weighted_fusion(self):
        """Test MultiscaleBlock with weighted fusion."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len),
        ]

        block = MultiscaleBlock(
            layers=layers,
            hidden_dim=hidden_dim,
            fusion_type="weighted",
        )

        output = block(x)
        assert output.shape == x.shape

        # Check that fusion weights exist
        assert hasattr(block, "fusion_weights")
        assert block.fusion_weights.shape == (len(layers),)

    def test_forward_shape_concat_fusion(self):
        """Test MultiscaleBlock with concatenation fusion."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len),
        ]

        block = MultiscaleBlock(
            layers=layers,
            hidden_dim=hidden_dim,
            fusion_type="concat",
        )

        output = block(x)
        assert output.shape == x.shape

        # Check that fusion projection exists
        assert hasattr(block, "fusion_proj")
        assert block.fusion_proj.in_features == hidden_dim * len(layers)
        assert block.fusion_proj.out_features == hidden_dim

    def test_multiple_scales(self):
        """Test with more than two scales."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len),
            SpectralAttention(hidden_dim=hidden_dim, num_heads=8),
        ]

        for fusion_type in ["add", "weighted", "concat"]:
            block = MultiscaleBlock(
                layers=layers,
                hidden_dim=hidden_dim,
                fusion_type=fusion_type,
            )

            output = block(x)
            assert output.shape == x.shape


class TestCascadeBlock:
    """Test CascadeBlock."""

    def test_forward_shape(self):
        """Test CascadeBlock forward pass."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len),
        ]

        block = CascadeBlock(
            layers=layers,
            hidden_dim=hidden_dim,
            share_norm=False,
        )

        output = block(x)
        assert output.shape == x.shape

    def test_shared_norm(self):
        """Test CascadeBlock with shared normalization."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len),
        ]

        block = CascadeBlock(
            layers=layers,
            hidden_dim=hidden_dim,
            share_norm=True,
        )

        output = block(x)
        assert output.shape == x.shape

        # Check that norms are shared (same object)
        assert all(norm is block.norm1 for norm in block.norms)

    def test_separate_norms(self):
        """Test CascadeBlock with separate normalizations."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len),
        ]

        block = CascadeBlock(
            layers=layers,
            hidden_dim=hidden_dim,
            share_norm=False,
        )

        output = block(x)
        assert output.shape == x.shape

        # Check that norms are different objects
        assert len({id(norm) for norm in block.norms}) == len(layers)

    def test_cascading_multiple_layers(self):
        """Test cascading more than two layers."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len),
            SpectralAttention(hidden_dim=hidden_dim, num_heads=8),
        ]

        block = CascadeBlock(
            layers=layers,
            hidden_dim=hidden_dim,
            share_norm=False,
        )

        output = block(x)
        assert output.shape == x.shape


class TestGradientFlow:
    """Test gradient flow through hybrid blocks."""

    def test_gradient_flow_all_hybrid_blocks(self):
        """Test gradient flow through all hybrid block types."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len),
        ]

        blocks_to_test = [
            AlternatingBlock(layer1=layers[0], layer2=layers[1], hidden_dim=hidden_dim),
            AdaptiveBlock(layers=layers, hidden_dim=hidden_dim, gate_type="soft"),
            AdaptiveBlock(layers=layers, hidden_dim=hidden_dim, gate_type="hard"),
            MultiscaleBlock(layers=layers, hidden_dim=hidden_dim, fusion_type="add"),
            MultiscaleBlock(layers=layers, hidden_dim=hidden_dim, fusion_type="weighted"),
            MultiscaleBlock(layers=layers, hidden_dim=hidden_dim, fusion_type="concat"),
            CascadeBlock(layers=layers, hidden_dim=hidden_dim, share_norm=False),
            CascadeBlock(layers=layers, hidden_dim=hidden_dim, share_norm=True),
        ]

        for block in blocks_to_test:
            x_copy = x.clone().detach().requires_grad_(True)
            output = block(x_copy)
            loss = output.sum()
            loss.backward()

            assert x_copy.grad is not None
            assert not torch.isnan(x_copy.grad).any()
            assert not torch.isinf(x_copy.grad).any()




class TestHybridComposition:
    """Test composing hybrid blocks."""

    def test_sequential_hybrid_blocks(self):
        """Test sequential composition of hybrid blocks."""
        batch_size, seq_len, hidden_dim = 2, 128, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)

        layers = [
            FourierMixing(hidden_dim=hidden_dim),
            GlobalFilterMixing(hidden_dim=hidden_dim, sequence_length=seq_len),
        ]

        # Create a sequence of different hybrid blocks
        blocks = nn.Sequential(
            AlternatingBlock(layer1=layers[0], layer2=layers[1], hidden_dim=hidden_dim),
            MultiscaleBlock(layers=layers, hidden_dim=hidden_dim, fusion_type="add"),
            CascadeBlock(layers=layers, hidden_dim=hidden_dim),
        )

        output = blocks(x)
        assert output.shape == x.shape
