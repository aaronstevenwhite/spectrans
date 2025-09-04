"""Unit tests for FNet models."""

import torch

from spectrans.models.fnet import FNet, FNetEncoder


class TestFNet:
    """Test FNet model."""

    def test_fnet_initialization(self):
        """Test FNet initialization with various configurations."""
        # Basic FNet
        model = FNet(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
            num_classes=10,
        )
        assert len(model.blocks) == 4
        assert model.use_real_fft is True  # Default value

        # FNet without real FFT
        model_complex = FNet(
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
            use_real_fft=False,
        )
        assert model_complex.use_real_fft is False

    def test_fnet_forward_pass(self):
        """Test FNet forward pass."""
        batch_size, seq_length = 2, 100
        vocab_size, hidden_dim = 500, 128
        num_classes = 5

        model = FNet(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=512,
            num_classes=num_classes,
            output_type="classification",
        )

        # Test with input_ids
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        output = model(input_ids=input_ids)
        assert output.shape == (batch_size, num_classes)

        # Test with inputs_embeds
        model_no_vocab = FNet(
            vocab_size=None,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=512,
            num_classes=num_classes,
        )
        inputs_embeds = torch.randn(batch_size, seq_length, hidden_dim)
        output = model_no_vocab(inputs_embeds=inputs_embeds)
        assert output.shape == (batch_size, num_classes)

    def test_fnet_encoder(self):
        """Test FNetEncoder variant."""
        batch_size, seq_length, hidden_dim = 2, 100, 128

        encoder = FNetEncoder(
            hidden_dim=hidden_dim,
            num_layers=4,
            max_sequence_length=512,
        )

        # Encoder should not have vocab embedding or output head
        assert encoder.embedding is None
        assert encoder.output_head is None

        # Test forward pass
        inputs = torch.randn(batch_size, seq_length, hidden_dim)
        output = encoder(inputs_embeds=inputs)
        assert output.shape == (batch_size, seq_length, hidden_dim)

    def test_fnet_complexity(self):
        """Test FNet complexity calculation."""
        model = FNet(
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
        )

        complexity = model.complexity
        assert "time" in complexity
        assert "space" in complexity
        # FNet should have O(n log n) complexity
        assert "log" in complexity["time"]

    def test_fnet_gradient_flow(self):
        """Test gradient flow through FNet."""
        batch_size, seq_length = 2, 50
        vocab_size, hidden_dim = 100, 64
        num_classes = 5

        model = FNet(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=512,
            num_classes=num_classes,
            use_real_fft=False,  # Test complex FFT path
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

    def test_fnet_from_config(self):
        """Test creating FNet from configuration."""
        from spectrans.config.models import FNetModelConfig

        config = FNetModelConfig(
            hidden_dim=256,
            num_layers=6,
            sequence_length=1024,
            dropout=0.2,
        )

        model = FNet.from_config(config)
        assert model.hidden_dim == 256
        assert model.num_layers == 6
        assert model.max_sequence_length == 1024
        assert len(model.blocks) == 6

    def test_fnet_real_vs_complex(self):
        """Test real FFT vs complex FFT paths."""
        batch_size, seq_length, hidden_dim = 2, 64, 128

        # Create two models with same config but different FFT types
        model_real = FNet(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            use_real_fft=True,
            output_type="none",
        )

        model_complex = FNet(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            use_real_fft=False,
            output_type="none",
        )

        # Same input
        inputs = torch.randn(batch_size, seq_length, hidden_dim)

        # Both should produce valid outputs
        output_real = model_real(inputs_embeds=inputs)
        output_complex = model_complex(inputs_embeds=inputs)

        assert output_real.shape == (batch_size, seq_length, hidden_dim)
        assert output_complex.shape == (batch_size, seq_length, hidden_dim)

        # Outputs may differ slightly due to different FFT implementations
        # but both should be valid tensors
        assert torch.isfinite(output_real).all()
        assert torch.isfinite(output_complex).all()

    def test_fnet_with_different_sequence_lengths(self):
        """Test FNet with various sequence lengths."""
        model = FNet(
            hidden_dim=128,
            num_layers=2,
            max_sequence_length=512,
            use_real_fft=True,
        )

        # Test with different sequence lengths
        for seq_length in [32, 64, 128, 256, 512]:
            inputs = torch.randn(2, seq_length, 128)
            output = model(inputs_embeds=inputs)
            assert output.shape == (2, seq_length, 128)

    def test_fnet_encoder_with_positional_encoding(self):
        """Test FNetEncoder with different positional encoding types."""
        batch_size, seq_length, hidden_dim = 2, 128, 128

        # Test with sinusoidal encoding
        encoder_sin = FNetEncoder(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            positional_encoding_type="sinusoidal",
        )

        # Test with learned encoding
        encoder_learned = FNetEncoder(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            positional_encoding_type="learned",
        )

        # Test without encoding
        encoder_none = FNetEncoder(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            use_positional_encoding=False,
        )

        inputs = torch.randn(batch_size, seq_length, hidden_dim)

        # All should produce valid outputs
        output_sin = encoder_sin(inputs_embeds=inputs)
        output_learned = encoder_learned(inputs_embeds=inputs)
        output_none = encoder_none(inputs_embeds=inputs)

        assert output_sin.shape == (batch_size, seq_length, hidden_dim)
        assert output_learned.shape == (batch_size, seq_length, hidden_dim)
        assert output_none.shape == (batch_size, seq_length, hidden_dim)
