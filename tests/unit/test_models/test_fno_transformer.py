"""Comprehensive tests for FNO transformer models.

This module tests the Fourier Neural Operator transformer implementations,
including the complete transformer, encoder, and decoder variants.
"""

import pytest
import torch

from spectrans.models.fno_transformer import FNODecoder, FNOEncoder, FNOTransformer


class TestFNOTransformer:
    """Test suite for FNOTransformer model."""

    def test_fno_initialization(self):
        """Test FNO transformer initialization."""
        model = FNOTransformer(
            hidden_dim=256,
            num_layers=4,
            modes=16,
            max_sequence_length=512,
        )

        assert model.hidden_dim == 256
        assert model.num_layers == 4
        assert model.modes == 16
        assert model.max_sequence_length == 512
        assert len(model.blocks) == 4

    def test_fno_forward_pass(self):
        """Test forward pass through FNO transformer."""
        batch_size = 8
        seq_len = 128
        hidden_dim = 256

        model = FNOTransformer(
            hidden_dim=hidden_dim,
            num_layers=4,
            modes=16,
            max_sequence_length=512,
        )

        # Test with pre-embedded inputs
        inputs = torch.randn(batch_size, seq_len, hidden_dim)
        output = model(inputs_embeds=inputs)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_fno_with_vocab(self):
        """Test FNO transformer with vocabulary and token inputs."""
        batch_size = 8
        seq_len = 128
        vocab_size = 1000
        hidden_dim = 256

        model = FNOTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=4,
            modes=16,
            max_sequence_length=512,
        )

        # Test with token inputs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = model(input_ids)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert model.embedding is not None
        assert model.embedding.num_embeddings == vocab_size

    def test_fno_with_classification(self):
        """Test FNO transformer with classification head."""
        batch_size = 8
        seq_len = 128
        hidden_dim = 256
        num_classes = 10

        model = FNOTransformer(
            hidden_dim=hidden_dim,
            num_layers=4,
            modes=16,
            num_classes=num_classes,
            max_sequence_length=512,
        )

        inputs = torch.randn(batch_size, seq_len, hidden_dim)
        output = model(inputs_embeds=inputs)

        assert output.shape == (batch_size, num_classes)
        assert model.output_head is not None

    def test_fno_encoder(self):
        """Test FNO encoder model."""
        batch_size = 8
        seq_len = 128
        hidden_dim = 256

        encoder = FNOEncoder(
            hidden_dim=hidden_dim,
            num_layers=4,
            modes=16,
            max_sequence_length=512,
        )

        inputs = torch.randn(batch_size, seq_len, hidden_dim)
        output = encoder(inputs_embeds=inputs)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert encoder.output_type == "none"
        assert encoder.embedding is None

    def test_fno_decoder(self):
        """Test FNO decoder model."""
        batch_size = 8
        seq_len = 128
        vocab_size = 1000
        hidden_dim = 256

        decoder = FNODecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=4,
            modes=16,
            max_sequence_length=512,
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = decoder(input_ids)

        assert output.shape == (batch_size, seq_len, vocab_size)
        assert decoder.output_type == "lm"
        assert hasattr(decoder, "lm_head")

    def test_fno_gradient_flow(self):
        """Test gradient flow through FNO transformer."""
        model = FNOTransformer(
            hidden_dim=128,
            num_layers=2,
            modes=8,
            num_classes=10,
            max_sequence_length=256,
        )

        inputs = torch.randn(4, 64, 128, requires_grad=True)
        output = model(inputs_embeds=inputs)
        loss = output.mean()
        loss.backward()

        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()
        assert not torch.isinf(inputs.grad).any()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_fno_different_modes(self):
        """Test FNO with different numbers of Fourier modes."""
        for modes in [8, 16, 32, 64]:
            model = FNOTransformer(
                hidden_dim=256,
                num_layers=2,
                modes=modes,
                max_sequence_length=512,
            )

            assert model.modes == modes
            inputs = torch.randn(4, 128, 256)
            output = model(inputs_embeds=inputs)
            assert output.shape == (4, 128, 256)

    def test_fno_mlp_ratio(self):
        """Test FNO with different MLP ratios."""
        for mlp_ratio in [1.0, 2.0, 4.0]:
            model = FNOTransformer(
                hidden_dim=256,
                num_layers=2,
                modes=16,
                mlp_ratio=mlp_ratio,
                max_sequence_length=512,
            )

            assert model.mlp_ratio == mlp_ratio
            inputs = torch.randn(4, 128, 256)
            output = model(inputs_embeds=inputs)
            assert output.shape == (4, 128, 256)

    def test_fno_2d_configuration(self):
        """Test FNO with 2D spatial configuration."""
        spatial_dim = 64
        model = FNOTransformer(
            hidden_dim=256,
            num_layers=2,
            modes=16,
            use_2d=True,
            spatial_dim=spatial_dim,
            max_sequence_length=spatial_dim * spatial_dim,
        )

        assert model.use_2d is True
        assert model.spatial_dim == spatial_dim
        # Sequence length viewed as spatial grid
        inputs = torch.randn(4, spatial_dim * spatial_dim, 256)
        output = model(inputs_embeds=inputs)
        assert output.shape == (4, spatial_dim * spatial_dim, 256)

    def test_fno_2d_validation(self):
        """Test validation of 2D configuration."""
        # Should raise error if spatial_dim not provided
        with pytest.raises(ValueError, match="spatial_dim must be specified"):
            FNOTransformer(
                hidden_dim=256,
                num_layers=2,
                modes=16,
                use_2d=True,
                max_sequence_length=4096,
            )

        # Should raise error if sequence length doesn't match spatial_dim²
        with pytest.raises(ValueError, match="must equal spatial_dim²"):
            FNOTransformer(
                hidden_dim=256,
                num_layers=2,
                modes=16,
                use_2d=True,
                spatial_dim=64,
                max_sequence_length=1000,  # Not 64²
            )

    def test_fno_with_positional_encoding(self):
        """Test FNO with different positional encoding settings."""
        # With sinusoidal encoding
        model1 = FNOTransformer(
            hidden_dim=256,
            num_layers=2,
            modes=16,
            use_positional_encoding=True,
            positional_encoding_type="sinusoidal",
            max_sequence_length=512,
        )

        # With learned encoding
        model2 = FNOTransformer(
            hidden_dim=256,
            num_layers=2,
            modes=16,
            use_positional_encoding=True,
            positional_encoding_type="learned",
            max_sequence_length=512,
        )

        # Without encoding
        model3 = FNOTransformer(
            hidden_dim=256,
            num_layers=2,
            modes=16,
            use_positional_encoding=False,
            max_sequence_length=512,
        )

        inputs = torch.randn(4, 128, 256)
        for model in [model1, model2, model3]:
            output = model(inputs_embeds=inputs)
            assert output.shape == (4, 128, 256)


class TestFNOEncoder:
    """Test suite for FNOEncoder model."""

    def test_encoder_gradient_flow(self):
        """Test gradient flow through FNO encoder."""
        encoder = FNOEncoder(
            hidden_dim=128,
            num_layers=2,
            modes=8,
            max_sequence_length=256,
        )

        inputs = torch.randn(4, 64, 128, requires_grad=True)
        output = encoder(inputs_embeds=inputs)
        loss = output.mean()
        loss.backward()

        assert inputs.grad is not None
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestFNODecoder:
    """Test suite for FNODecoder model."""

    def test_decoder_gradient_flow(self):
        """Test gradient flow through FNO decoder."""
        vocab_size = 1000
        decoder = FNODecoder(
            vocab_size=vocab_size,
            hidden_dim=128,
            num_layers=2,
            modes=8,
            max_sequence_length=256,
        )

        input_ids = torch.randint(0, vocab_size, (4, 64))
        output = decoder(input_ids)
        loss = output.mean()
        loss.backward()

        for param in decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_decoder_generation_shape(self):
        """Test decoder output shape for generation."""
        vocab_size = 1000
        batch_size = 4
        seq_len = 64

        decoder = FNODecoder(
            vocab_size=vocab_size,
            hidden_dim=256,
            num_layers=4,
            modes=16,
            causal=True,
            max_sequence_length=512,
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = decoder(input_ids)

        assert output.shape == (batch_size, seq_len, vocab_size)
        # Check output is valid logits
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_decoder_with_gradient_checkpointing(self):
        """Test FNO decoder with gradient checkpointing."""
        decoder = FNODecoder(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=4,
            modes=16,
            gradient_checkpointing=True,
            max_sequence_length=512,
        )

        assert decoder.gradient_checkpointing is True
        input_ids = torch.randint(0, 1000, (4, 64))
        output = decoder(input_ids)
        assert output.shape == (4, 64, 1000)
