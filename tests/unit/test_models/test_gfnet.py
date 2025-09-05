"""Unit tests for GFNet models."""

import torch

from spectrans.models.gfnet import GFNet, GFNetEncoder


class TestGFNet:
    """Test GFNet model."""

    def test_gfnet_initialization(self):
        """Test GFNet initialization with various configurations."""
        # Basic GFNet
        model = GFNet(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
            num_classes=10,
        )
        assert len(model.blocks) == 4
        assert model.filter_activation == "sigmoid"  # Default value

        # GFNet with tanh activation
        model_tanh = GFNet(
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
            filter_activation="tanh",
        )
        assert model_tanh.filter_activation == "tanh"

    def test_gfnet_forward_pass(self):
        """Test GFNet forward pass."""
        batch_size, seq_length = 2, 128  # Use consistent sequence length
        vocab_size, hidden_dim = 500, 128
        num_classes = 5

        model = GFNet(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,  # Match the actual sequence length
            num_classes=num_classes,
            output_type="classification",
        )

        # Test with input_ids
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        output = model(input_ids=input_ids)
        assert output.shape == (batch_size, num_classes)

        # Test with inputs_embeds
        model_no_vocab = GFNet(
            vocab_size=None,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,  # Match the actual sequence length
            num_classes=num_classes,
        )
        inputs_embeds = torch.randn(batch_size, seq_length, hidden_dim)
        output = model_no_vocab(inputs_embeds=inputs_embeds)
        assert output.shape == (batch_size, num_classes)

    def test_gfnet_encoder(self):
        """Test GFNetEncoder variant."""
        batch_size, seq_length, hidden_dim = 2, 128, 128  # Consistent sequence length

        encoder = GFNetEncoder(
            hidden_dim=hidden_dim,
            num_layers=4,
            max_sequence_length=seq_length,  # Match the actual sequence length
        )

        # Encoder should not have vocab embedding or output head
        assert encoder.embedding is None
        assert encoder.output_head is None

        # Test forward pass
        inputs = torch.randn(batch_size, seq_length, hidden_dim)
        output = encoder(inputs_embeds=inputs)
        assert output.shape == (batch_size, seq_length, hidden_dim)


    def test_gfnet_gradient_flow(self):
        """Test gradient flow through GFNet."""
        batch_size, seq_length = 2, 50
        vocab_size, hidden_dim = 100, 64
        num_classes = 5

        model = GFNet(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,  # Must match actual sequence length for filters
            num_classes=num_classes,
            filter_activation="tanh",  # Test tanh path
        )

        # Forward pass
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        target = torch.randint(0, num_classes, (batch_size,))
        output = model(input_ids=input_ids)

        # Compute loss and backward
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_gfnet_from_config(self):
        """Test creating GFNet from configuration."""
        from spectrans.config.models import GFNetModelConfig

        config = GFNetModelConfig(
            hidden_dim=256,
            num_layers=6,
            sequence_length=1024,
            dropout=0.2,
        )

        model = GFNet.from_config(config)
        assert model.hidden_dim == 256
        assert model.num_layers == 6
        assert model.max_sequence_length == 1024
        assert len(model.blocks) == 6

    def test_gfnet_filter_activations(self):
        """Test GFNet with different filter activation functions."""
        batch_size, seq_length, hidden_dim = 2, 64, 128

        # Test both sigmoid and tanh activations
        for activation in ["sigmoid", "tanh"]:
            model = GFNet(
                hidden_dim=hidden_dim,
                num_layers=2,
                max_sequence_length=seq_length,
                filter_activation=activation,
                output_type="none",
            )

            inputs = torch.randn(batch_size, seq_length, hidden_dim)
            output = model(inputs_embeds=inputs)

            assert output.shape == (batch_size, seq_length, hidden_dim)
            assert torch.isfinite(output).all()

    def test_gfnet_sequence_length_dependency(self):
        """Test that GFNet filters are tied to sequence length."""
        hidden_dim = 128

        # Create model with specific max sequence length
        model = GFNet(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=256,
            filter_activation="sigmoid",
        )

        # Test with exact sequence length
        inputs_256 = torch.randn(2, 256, hidden_dim)
        output_256 = model(inputs_embeds=inputs_256)
        assert output_256.shape == (2, 256, hidden_dim)

        # Note: GFNet requires exact sequence length matching due to fixed filter size
        # Unlike FNet which can handle variable lengths, GFNet filters are
        # initialized to a specific sequence length and must match input length

    def test_gfnet_encoder_with_different_configs(self):
        """Test GFNetEncoder with various configurations."""
        batch_size, seq_length, hidden_dim = 2, 128, 128

        # Test with different FFN dimensions
        encoder_large_ffn = GFNetEncoder(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            ffn_hidden_dim=hidden_dim * 8,  # Larger FFN
        )

        # Test with small FFN
        encoder_small_ffn = GFNetEncoder(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            ffn_hidden_dim=hidden_dim,  # Same as hidden dim
        )

        inputs = torch.randn(batch_size, seq_length, hidden_dim)

        # Both should work
        output_large = encoder_large_ffn(inputs_embeds=inputs)
        output_small = encoder_small_ffn(inputs_embeds=inputs)

        assert output_large.shape == (batch_size, seq_length, hidden_dim)
        assert output_small.shape == (batch_size, seq_length, hidden_dim)

    def test_gfnet_with_gradient_checkpointing(self):
        """Test GFNet with gradient checkpointing enabled."""
        model = GFNet(
            hidden_dim=128,
            num_layers=4,
            max_sequence_length=256,
            gradient_checkpointing=True,
        )

        # Training mode for gradient checkpointing
        model.train()
        inputs = torch.randn(2, 256, 128, requires_grad=True)
        output = model(inputs_embeds=inputs)

        # Should produce valid output
        assert output.shape == (2, 256, 128)

        # Gradient should flow
        loss = output.sum()
        loss.backward()
        assert inputs.grad is not None
