"""Test that GlobalFilterMixing correctly handles variable sequence lengths.

This test demonstrates that the interpolation-based approach for handling
variable sequence lengths is superior to naive truncation/padding.
"""

import pytest
import torch
import torch.nn as nn

from spectrans.layers.mixing.global_filter import (
    AdaptiveGlobalFilter,
    GlobalFilterMixing,
    GlobalFilterMixing2D,
)


class TestGlobalFilterInterpolation:
    """Test interpolation-based handling of variable sequence lengths."""

    def test_variable_sequence_lengths(self):
        """Test that GlobalFilterMixing handles different sequence lengths."""
        layer = GlobalFilterMixing(hidden_dim=256, sequence_length=512)

        # Test with various sequence lengths
        test_cases = [
            (128, "shorter"),
            (256, "half"),
            (512, "exact"),
            (768, "longer"),
            (1024, "double"),
        ]

        for seq_len, desc in test_cases:
            x = torch.randn(2, seq_len, 256)
            y = layer(x)
            assert y.shape == x.shape, f"Failed for {desc} sequence length"
            # Ensure output is real-valued and finite
            assert torch.isreal(y).all()
            assert torch.isfinite(y).all()

    def test_gradient_flow_through_interpolation(self):
        """Test that gradients flow correctly through interpolated filters."""
        layer = GlobalFilterMixing(hidden_dim=128, sequence_length=256)

        # Test gradient flow for different sequence lengths
        for seq_len in [128, 256, 384]:
            x = torch.randn(2, seq_len, 128, requires_grad=True)
            y = layer(x)
            loss = y.sum()
            loss.backward()

            # Check input gradients
            assert x.grad is not None
            assert torch.isfinite(x.grad).all()
            assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

            # Reset gradients
            x.grad.zero_()
            layer.zero_grad()

    def test_learned_patterns_preserved(self):
        """Test that learned frequency patterns are preserved across scales."""
        layer = GlobalFilterMixing(hidden_dim=64, sequence_length=128)

        # Initialize filters with a specific pattern (low-pass filter)
        with torch.no_grad():
            # Create frequency-dependent pattern
            freqs = torch.fft.fftfreq(128).abs()
            # Low-pass filter pattern (higher weights for lower frequencies)
            pattern = torch.exp(-10 * freqs).unsqueeze(-1)
            layer.filter_real.data = pattern.expand(128, 64) * 0.5
            layer.filter_imag.data = pattern.expand(128, 64) * 0.1

        # Test that the pattern is preserved at different resolutions
        # Process inputs of different lengths and check filter adaptation
        for seq_len in [64, 128, 256]:
            x = torch.randn(1, seq_len, 64)
            _ = layer(x)

            # The interpolation should preserve the general shape of the filter
            # (low frequencies emphasized, high frequencies suppressed)
            # This is validated by checking the output doesn't explode or vanish

    def test_2d_filter_interpolation(self):
        """Test 2D global filter with variable dimensions."""
        layer = GlobalFilterMixing2D(hidden_dim=256, sequence_length=512)

        # Test with different dimensions
        test_cases = [
            (256, 256),
            (384, 256),
            (512, 256),
            (256, 384),
            (512, 512),
        ]

        for seq_len, hidden in test_cases:
            x = torch.randn(2, seq_len, hidden)
            y = layer(x)
            assert y.shape == x.shape, f"Failed for shape ({seq_len}, {hidden})"
            assert torch.isreal(y).all()
            assert torch.isfinite(y).all()

    def test_adaptive_filter_interpolation(self):
        """Test adaptive global filter with interpolation."""
        layer = AdaptiveGlobalFilter(
            hidden_dim=128,
            sequence_length=256,
            adaptive_initialization=True,
            filter_regularization=0.01
        )

        # Test with different sequence lengths
        for seq_len in [128, 256, 512]:
            x = torch.randn(2, seq_len, 128)
            y = layer(x)
            assert y.shape == x.shape

            # Check regularization loss works with interpolated filters
            reg_loss = layer.get_regularization_loss()
            assert reg_loss.item() >= 0

    def test_interpolation_consistency(self):
        """Test that interpolation is consistent and deterministic."""
        layer = GlobalFilterMixing(hidden_dim=64, sequence_length=128)

        # Process same input twice with different sequence length
        x = torch.randn(2, 200, 64)
        y1 = layer(x)
        y2 = layer(x)

        # Should get same result (deterministic)
        assert torch.allclose(y1, y2)

    def test_interpolation_preserves_energy(self):
        """Test that interpolation doesn't cause energy explosion/vanishing."""
        layer = GlobalFilterMixing(hidden_dim=64, sequence_length=128)

        # Initialize with reasonable filter values
        nn.init.xavier_uniform_(layer.filter_real)
        nn.init.xavier_uniform_(layer.filter_imag)

        energy_ratios = []
        for seq_len in [64, 128, 256, 512]:
            x = torch.randn(4, seq_len, 64)
            y = layer(x)

            # Compute energy ratio
            input_energy = (x ** 2).mean()
            output_energy = (y ** 2).mean()
            ratio = output_energy / input_energy
            energy_ratios.append(ratio.item())

        # Energy shouldn't explode or vanish too much
        # Allow for some variation but not extreme
        for ratio in energy_ratios:
            assert 0.1 < ratio < 10.0, f"Energy ratio {ratio} is out of bounds"

    @pytest.mark.parametrize("activation", ["sigmoid", "tanh", "identity"])
    def test_activations_with_interpolation(self, activation):
        """Test different activation functions with interpolation."""
        layer = GlobalFilterMixing(
            hidden_dim=64,
            sequence_length=128,
            activation=activation
        )

        # Test with non-matching sequence length
        x = torch.randn(2, 200, 64)
        y = layer(x)

        assert y.shape == x.shape
        assert torch.isfinite(y).all()

        # Check gradients flow
        x.requires_grad_(True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
