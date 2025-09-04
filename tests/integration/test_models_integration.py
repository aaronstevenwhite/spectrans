"""Integration tests for spectral transformer models.

This module contains integration tests that verify the complete functionality
of spectral transformer models, including configuration loading, registry
integration, and end-to-end training simulations.
"""

import pytest
import torch

from spectrans.config.models import FNetModelConfig, GFNetModelConfig
from spectrans.core.registry import registry
from spectrans.models.afno import AFNOEncoder, AFNOModel
from spectrans.models.fnet import FNet, FNetEncoder
from spectrans.models.gfnet import GFNet, GFNetEncoder


class TestModelRegistry:
    """Test model registration and retrieval from registry."""

    def test_models_registered(self):
        """Test that all models are properly registered."""
        # Check that models are registered
        assert "fnet" in registry.list("model")
        assert "fnet_encoder" in registry.list("model")
        assert "gfnet" in registry.list("model")
        assert "gfnet_encoder" in registry.list("model")
        assert "afno" in registry.list("model")
        assert "afno_encoder" in registry.list("model")

    def test_create_models_from_registry(self):
        """Test creating models from registry."""
        # Create FNet from registry
        fnet = registry.get("model", "fnet")(
            hidden_dim=128,
            num_layers=2,
            max_sequence_length=256,
        )
        assert isinstance(fnet, FNet)
        assert len(fnet.blocks) == 2

        # Create GFNet from registry
        gfnet = registry.get("model", "gfnet")(
            hidden_dim=128,
            num_layers=2,
            max_sequence_length=256,
        )
        assert isinstance(gfnet, GFNet)

        # Create AFNO from registry
        afno = registry.get("model", "afno")(
            hidden_dim=128,
            num_layers=2,
            max_sequence_length=256,
        )
        assert isinstance(afno, AFNOModel)


class TestModelConfiguration:
    """Test model configuration and instantiation."""

    def test_fnet_from_config(self):
        """Test creating FNet from configuration."""
        config = FNetModelConfig(
            hidden_dim=256,
            num_layers=4,
            sequence_length=512,
            dropout=0.1,
        )

        model = FNet.from_config(config)
        assert model.hidden_dim == 256
        assert model.num_layers == 4
        assert model.max_sequence_length == 512
        assert len(model.blocks) == 4

    def test_gfnet_from_config(self):
        """Test creating GFNet from configuration."""
        config = GFNetModelConfig(
            hidden_dim=256,
            num_layers=4,
            sequence_length=512,
            dropout=0.1,
        )

        model = GFNet.from_config(config)
        assert model.hidden_dim == 256
        assert model.num_layers == 4
        assert model.max_sequence_length == 512
        assert len(model.blocks) == 4


class TestModelComparison:
    """Test comparing different model architectures."""

    @pytest.fixture
    def common_config(self):
        """Common configuration for all models."""
        return {
            "vocab_size": 1000,
            "hidden_dim": 128,
            "num_layers": 2,
            "max_sequence_length": 256,
            "num_classes": 10,
            "dropout": 0.1,
        }

    def test_model_output_shapes(self, common_config):
        """Test that all models produce correct output shapes."""
        batch_size = 4
        seq_length = 256

        # Create models
        fnet = FNet(**common_config)
        gfnet = GFNet(**common_config)
        afno = AFNOModel(
            **common_config,
            modes_seq=64,
            modes_hidden=32,
        )

        # Create input
        input_ids = torch.randint(0, common_config["vocab_size"], (batch_size, seq_length))

        # Test outputs
        fnet_output = fnet(input_ids=input_ids)
        gfnet_output = gfnet(input_ids=input_ids)
        afno_output = afno(input_ids=input_ids)

        # All should produce same shape for classification
        expected_shape = (batch_size, common_config["num_classes"])
        assert fnet_output.shape == expected_shape
        assert gfnet_output.shape == expected_shape
        assert afno_output.shape == expected_shape

    def test_encoder_output_shapes(self):
        """Test that all encoders produce correct output shapes."""
        batch_size = 4
        seq_length = 256
        hidden_dim = 128

        # Create encoders
        fnet_encoder = FNetEncoder(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
        )
        gfnet_encoder = GFNetEncoder(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
        )
        afno_encoder = AFNOEncoder(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            modes_seq=64,
            modes_hidden=32,
        )

        # Create input
        inputs = torch.randn(batch_size, seq_length, hidden_dim)

        # Test outputs
        fnet_output = fnet_encoder(inputs_embeds=inputs)
        gfnet_output = gfnet_encoder(inputs_embeds=inputs)
        afno_output = afno_encoder(inputs_embeds=inputs)

        # All should preserve input shape
        expected_shape = (batch_size, seq_length, hidden_dim)
        assert fnet_output.shape == expected_shape
        assert gfnet_output.shape == expected_shape
        assert afno_output.shape == expected_shape


class TestEndToEndTraining:
    """Test end-to-end training scenarios."""

    def test_classification_training_loop(self):
        """Test a simple training loop for classification."""
        # Setup
        batch_size = 8
        seq_length = 128
        vocab_size = 100
        hidden_dim = 64
        num_classes = 5

        # Create model
        model = FNet(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            num_classes=num_classes,
        )

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Training step
        model.train()
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        labels = torch.randint(0, num_classes, (batch_size,))

        # Forward pass
        logits = model(input_ids=input_ids)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Check that loss is finite
        assert torch.isfinite(loss).all()

        # Evaluation
        model.eval()
        with torch.no_grad():
            eval_logits = model(input_ids=input_ids)
            eval_loss = torch.nn.functional.cross_entropy(eval_logits, labels)
            assert torch.isfinite(eval_loss).all()

    def test_gradient_checkpointing(self):
        """Test models with gradient checkpointing enabled."""
        # Create model with gradient checkpointing
        model = AFNOModel(
            hidden_dim=128,
            num_layers=4,
            max_sequence_length=256,
            modes_seq=32,
            modes_hidden=32,
            gradient_checkpointing=True,
        )

        # Training forward pass
        model.train()
        inputs = torch.randn(2, 256, 128, requires_grad=True)
        output = model(inputs_embeds=inputs)

        # Should produce valid output
        assert output.shape == (2, 256, 128)  # No classification head

        # Gradient should flow
        loss = output.sum()
        loss.backward()
        assert inputs.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for mixed precision")
    def test_mixed_precision_compatibility(self):
        """Test model compatibility with mixed precision training."""
        device = torch.device("cuda")
        model = GFNet(
            hidden_dim=128,
            num_layers=2,
            max_sequence_length=128,
        ).to(device)

        # Test with automatic mixed precision
        inputs = torch.randn(2, 128, 128).to(device)

        with torch.cuda.amp.autocast():
            output = model(inputs_embeds=inputs)

        # Should produce valid output (may be float16 or float32 depending on autocast behavior)
        assert torch.isfinite(output).all()

        # Also test explicit half precision
        model_half = model.half()
        inputs_half = inputs.half()
        output_half = model_half(inputs_embeds=inputs_half)

        assert output_half.dtype == torch.float16
        assert torch.isfinite(output_half).all()


class TestModelRobustness:
    """Test model robustness to various inputs."""

    def test_variable_sequence_lengths(self):
        """Test models with different sequence lengths."""
        model = FNet(
            hidden_dim=128,
            num_layers=2,
            max_sequence_length=512,  # Max length
        )

        # Test with various lengths
        for seq_length in [64, 128, 256, 512]:
            inputs = torch.randn(2, seq_length, 128)
            output = model(inputs_embeds=inputs)
            assert output.shape == (2, seq_length, 128)

    def test_single_sample_batch(self):
        """Test models with batch size of 1."""
        model = AFNOModel(
            hidden_dim=128,
            num_layers=2,
            max_sequence_length=256,
            modes_seq=32,
            modes_hidden=32,
        )

        # Single sample
        inputs = torch.randn(1, 256, 128)
        output = model(inputs_embeds=inputs)
        assert output.shape == (1, 256, 128)

    def test_empty_gradient_checkpointing(self):
        """Test that gradient checkpointing doesn't break in eval mode."""
        model = FNet(
            hidden_dim=128,
            num_layers=2,
            max_sequence_length=256,
            gradient_checkpointing=True,
        )

        # Eval mode should work even with gradient checkpointing enabled
        model.eval()
        inputs = torch.randn(2, 256, 128)
        with torch.no_grad():
            output = model(inputs_embeds=inputs)
        assert output.shape == (2, 256, 128)
