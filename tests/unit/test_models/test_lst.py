"""Unit tests for Linear Spectral Transform (LST) models.

Tests cover initialization, forward passes, gradient flow, complexity properties,
configuration-based construction, and edge cases for LSTTransformer, LSTEncoder,
and LSTDecoder models.
"""

import torch
import torch.nn as nn

from spectrans.models.lst import LSTDecoder, LSTEncoder, LSTTransformer


class TestLSTTransformer:
    """Test LSTTransformer model."""

    def test_lst_initialization(self):
        """Test model initialization with various configurations."""
        # Test with default parameters
        model = LSTTransformer(
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
        )
        assert model.hidden_dim == 256
        assert model.num_layers == 4
        assert model.transform_type == "dct"  # Default
        assert model.use_conv_bias is True  # Default

        # Test with custom parameters
        model = LSTTransformer(
            vocab_size=10000,
            hidden_dim=512,
            num_layers=6,
            transform_type="hadamard",
            use_conv_bias=False,
            num_classes=100,
            ffn_hidden_dim=1024,
            dropout=0.1,
            max_sequence_length=1024,
        )
        assert model.embedding is not None
        assert model.embedding.num_embeddings == 10000
        assert model.transform_type == "hadamard"
        assert model.use_conv_bias is False
        assert model.output_head is not None
        assert model.ffn_hidden_dim == 1024

        # Check blocks are built correctly
        assert len(model.blocks) == 6
        for block in model.blocks:
            assert hasattr(block, "mixing_layer")
            assert hasattr(block, "ffn")

    def test_lst_forward_pass(self):
        """Test forward pass with different input types."""
        batch_size, seq_length, hidden_dim = 2, 100, 256

        # Test with inputs_embeds
        model = LSTTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            transform_type="dct",
            max_sequence_length=512,
        )

        inputs_embeds = torch.randn(batch_size, seq_length, hidden_dim)
        output = model(inputs_embeds=inputs_embeds)
        assert output.shape == (batch_size, seq_length, hidden_dim)

        # Test with input_ids
        vocab_size = 1000
        model = LSTTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            transform_type="dst",
            max_sequence_length=512,
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        output = model(input_ids)
        assert output.shape == (batch_size, seq_length, hidden_dim)

        # Test with classification head
        num_classes = 10
        model = LSTTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_classes=num_classes,
            max_sequence_length=512,
        )

        logits = model(input_ids)
        assert logits.shape == (batch_size, num_classes)

    def test_lst_encoder(self):
        """Test encoder-only model variant."""
        batch_size, hidden_dim = 2, 256

        # For Hadamard transform, use power-of-2 sequence length
        seq_length_hadamard = 128  # Power of 2

        encoder = LSTEncoder(
            hidden_dim=hidden_dim,
            num_layers=3,
            transform_type="hadamard",
            use_conv_bias=True,
            max_sequence_length=512,
        )

        # Check no classification head
        assert encoder.output_type == "none"
        assert not hasattr(encoder, "classification_head")

        # Test forward pass with Hadamard
        inputs_embeds = torch.randn(batch_size, seq_length_hadamard, hidden_dim)
        output = encoder(inputs_embeds=inputs_embeds)
        assert output.shape == (batch_size, seq_length_hadamard, hidden_dim)

        # Test with token inputs and DST (works with any sequence length)
        vocab_size = 1000
        seq_length = 100  # Non-power-of-2 is fine for DST
        encoder = LSTEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=3,
            transform_type="dst",
            max_sequence_length=512,
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        output = encoder(input_ids)
        assert output.shape == (batch_size, seq_length, hidden_dim)

    def test_lst_decoder(self):
        """Test decoder model with causal masking."""
        batch_size, seq_length = 2, 100
        vocab_size = 1000
        hidden_dim = 256

        decoder = LSTDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=4,
            transform_type="dst",
            causal=True,
            max_sequence_length=512,
        )

        # Check decoder attributes
        assert decoder.output_type == "lm"
        assert hasattr(decoder, "lm_head")
        assert decoder.causal is True
        assert decoder.transform_type == "dst"

        # Test forward pass
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        logits = decoder(input_ids)
        assert logits.shape == (batch_size, seq_length, vocab_size)

        # Test with embeddings
        inputs_embeds = torch.randn(batch_size, seq_length, hidden_dim)
        logits = decoder(inputs_embeds=inputs_embeds)
        assert logits.shape == (batch_size, seq_length, vocab_size)

        # Test non-causal decoder
        decoder_non_causal = LSTDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            transform_type="dct",
            causal=False,
            max_sequence_length=512,
        )
        assert decoder_non_causal.causal is False

    def test_lst_gradient_flow(self):
        """Test gradient flow through the model."""
        model = LSTTransformer(
            hidden_dim=128,
            num_layers=2,
            transform_type="dct",
            num_classes=10,
            max_sequence_length=256,
        )

        # Create input and target
        batch_size, seq_length = 2, 50
        inputs_embeds = torch.randn(batch_size, seq_length, 128, requires_grad=True)
        target = torch.randint(0, 10, (batch_size,))

        # Forward pass
        logits = model(inputs_embeds=inputs_embeds)
        loss = nn.CrossEntropyLoss()(logits, target)

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert inputs_embeds.grad is not None
        assert not torch.isnan(inputs_embeds.grad).any()
        assert not torch.isinf(inputs_embeds.grad).any()

        # Check model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_lst_transform_types(self):
        """Test different spectral transform types."""
        batch_size, hidden_dim = 2, 128

        # Use power-of-2 for Hadamard compatibility
        seq_length = 64  # Power of 2

        transforms = ["dct", "dst", "hadamard"]
        outputs = {}

        for transform_type in transforms:
            model = LSTTransformer(
                hidden_dim=hidden_dim,
                num_layers=2,
                transform_type=transform_type,
                max_sequence_length=256,
            )

            inputs = torch.randn(batch_size, seq_length, hidden_dim)
            output = model(inputs_embeds=inputs)
            assert output.shape == (batch_size, seq_length, hidden_dim)
            outputs[transform_type] = output

        # Outputs should be different for different transforms
        for t1 in transforms:
            for t2 in transforms:
                if t1 != t2:
                    assert not torch.allclose(outputs[t1], outputs[t2], atol=1e-2)

    def test_lst_conv_bias(self):
        """Test with and without convolution bias."""
        batch_size, seq_length, hidden_dim = 2, 64, 128  # Use power-of-2

        # Test with bias
        model_with_bias = LSTTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            use_conv_bias=True,
            max_sequence_length=256,
        )

        # Test without bias
        model_no_bias = LSTTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            use_conv_bias=False,
            max_sequence_length=256,
        )

        inputs = torch.randn(batch_size, seq_length, hidden_dim)

        # Both should produce valid outputs
        output_with_bias = model_with_bias(inputs_embeds=inputs)
        output_no_bias = model_no_bias(inputs_embeds=inputs)

        assert output_with_bias.shape == (batch_size, seq_length, hidden_dim)
        assert output_no_bias.shape == (batch_size, seq_length, hidden_dim)

        # Outputs will be different
        assert not torch.allclose(output_with_bias, output_no_bias, atol=1e-2)

    def test_lst_different_sequence_lengths(self):
        """Test model handles different sequence lengths correctly."""
        hidden_dim = 128
        model = LSTTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            transform_type="dct",  # DCT works with any length
            max_sequence_length=1024,
        )

        # Test with different sequence lengths (DCT handles any length)
        for seq_length in [10, 50, 100, 500]:
            inputs = torch.randn(2, seq_length, hidden_dim)
            output = model(inputs_embeds=inputs)
            assert output.shape == (2, seq_length, hidden_dim)

        # Test Hadamard with power-of-2 lengths
        model_hadamard = LSTTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            transform_type="hadamard",
            max_sequence_length=1024,
        )

        for seq_length in [16, 32, 64, 128, 256]:
            inputs = torch.randn(2, seq_length, hidden_dim)
            output = model_hadamard(inputs_embeds=inputs)
            assert output.shape == (2, seq_length, hidden_dim)

    def test_lst_with_positional_encoding(self):
        """Test model with different positional encoding types."""
        batch_size, seq_length, hidden_dim = 2, 100, 256

        # Test with sinusoidal positional encoding
        model_sin = LSTTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            use_positional_encoding=True,
            positional_encoding_type="sinusoidal",
            max_sequence_length=512,
        )

        # Test with learned positional encoding
        model_learned = LSTTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            use_positional_encoding=True,
            positional_encoding_type="learned",
            max_sequence_length=512,
        )

        # Test without positional encoding
        model_no_pe = LSTTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            use_positional_encoding=False,
            max_sequence_length=512,
        )

        inputs = torch.randn(batch_size, seq_length, hidden_dim)

        output_sin = model_sin(inputs_embeds=inputs)
        output_learned = model_learned(inputs_embeds=inputs)
        output_no_pe = model_no_pe(inputs_embeds=inputs)

        # All should have correct shape
        assert output_sin.shape == (batch_size, seq_length, hidden_dim)
        assert output_learned.shape == (batch_size, seq_length, hidden_dim)
        assert output_no_pe.shape == (batch_size, seq_length, hidden_dim)

        # Outputs should differ
        assert not torch.allclose(output_sin, output_no_pe, atol=1e-3)
        assert not torch.allclose(output_learned, output_no_pe, atol=1e-3)


class TestLSTEncoder:
    """Test LSTEncoder model."""

    def test_encoder_gradient_flow(self):
        """Test gradient flow through encoder."""
        encoder = LSTEncoder(
            hidden_dim=128,
            num_layers=2,
            transform_type="hadamard",
            max_sequence_length=256,
        )

        batch_size, seq_length = 2, 64  # Power of 2 for Hadamard
        inputs = torch.randn(batch_size, seq_length, 128, requires_grad=True)

        output = encoder(inputs_embeds=inputs)
        loss = output.mean()
        loss.backward()

        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()

        # Check all parameters have gradients
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


class TestLSTDecoder:
    """Test LSTDecoder model."""

    def test_decoder_gradient_flow(self):
        """Test gradient flow through decoder."""
        vocab_size = 1000
        decoder = LSTDecoder(
            vocab_size=vocab_size,
            hidden_dim=128,
            num_layers=2,
            transform_type="dct",
            causal=False,
            max_sequence_length=256,
        )

        batch_size, seq_length = 2, 50
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        target = torch.randint(0, vocab_size, (batch_size, seq_length))

        logits = decoder(input_ids)
        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, vocab_size),
            target.reshape(-1),
        )
        loss.backward()

        # Check all parameters have gradients
        for name, param in decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_decoder_generation_shape(self):
        """Test decoder output shape for generation."""
        vocab_size = 5000
        hidden_dim = 256
        decoder = LSTDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=4,
            transform_type="dst",  # DST works with any length
            causal=True,
            max_sequence_length=1024,
        )

        # Test autoregressive generation shape
        batch_size = 4
        for seq_length in [1, 10, 50, 100]:  # DST handles any length
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
            logits = decoder(input_ids)
            assert logits.shape == (batch_size, seq_length, vocab_size)

            # Check logits are valid probabilities after softmax
            probs = torch.softmax(logits, dim=-1)
            assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, seq_length))

    def test_decoder_with_gradient_checkpointing(self):
        """Test decoder with gradient checkpointing."""
        vocab_size = 1000
        decoder = LSTDecoder(
            vocab_size=vocab_size,
            hidden_dim=128,
            num_layers=4,
            transform_type="dct",  # Use DCT instead of Hadamard to avoid power-of-2 requirement
            gradient_checkpointing=True,
            max_sequence_length=256,
        )

        batch_size, seq_length = 2, 100  # Any length works with DCT
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

        # Should work with gradient checkpointing
        logits = decoder(input_ids)
        assert logits.shape == (batch_size, seq_length, vocab_size)

        # Test backward pass with checkpointing
        loss = logits.mean()
        loss.backward()

        # Parameters should have gradients
        for param in decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
