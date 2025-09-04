"""Unit tests for Wavelet Transformer models."""

import torch

from spectrans.models.wavenet_transformer import (
    WaveletDecoder,
    WaveletEncoder,
    WaveletTransformer,
)


class TestWaveletTransformer:
    """Test WaveletTransformer model."""

    def test_wavelet_transformer_initialization(self):
        """Test WaveletTransformer initialization with various configurations."""
        # Basic initialization
        model = WaveletTransformer(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
            num_classes=10,
            wavelet="db4",
            levels=3,
        )
        assert len(model.blocks) == 4
        assert model.wavelet == "db4"
        assert model.levels == 3
        assert model.mixing_mode == "pointwise"  # Default

        # Different wavelet and mixing mode
        model_sym = WaveletTransformer(
            hidden_dim=256,
            num_layers=4,
            wavelet="sym6",
            levels=2,
            mixing_mode="channel",
        )
        assert model_sym.wavelet == "sym6"
        assert model_sym.levels == 2
        assert model_sym.mixing_mode == "channel"

    def test_wavelet_transformer_forward_pass(self):
        """Test WaveletTransformer forward pass."""
        batch_size, seq_length = 2, 100
        vocab_size, hidden_dim = 500, 128
        num_classes = 5

        model = WaveletTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=512,
            num_classes=num_classes,
            wavelet="db4",
            levels=2,
            output_type="classification",
        )

        # Test with input_ids
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        output = model(input_ids=input_ids)
        assert output.shape == (batch_size, num_classes)

        # Test with inputs_embeds
        model_no_vocab = WaveletTransformer(
            vocab_size=None,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=512,
            num_classes=num_classes,
            wavelet="db4",
        )
        inputs_embeds = torch.randn(batch_size, seq_length, hidden_dim)
        output = model_no_vocab(inputs_embeds=inputs_embeds)
        assert output.shape == (batch_size, num_classes)

    def test_wavelet_encoder(self):
        """Test WaveletEncoder variant."""
        batch_size, seq_length, hidden_dim = 2, 100, 128

        encoder = WaveletEncoder(
            hidden_dim=hidden_dim,
            num_layers=4,
            max_sequence_length=512,
            wavelet="db4",
            levels=3,
        )

        # Encoder should not have vocab embedding or output head
        assert encoder.embedding is None
        assert encoder.output_head is None

        # Test forward pass
        inputs = torch.randn(batch_size, seq_length, hidden_dim)
        output = encoder(inputs_embeds=inputs)
        assert output.shape == (batch_size, seq_length, hidden_dim)

    def test_wavelet_decoder(self):
        """Test WaveletDecoder variant."""
        batch_size, seq_length = 2, 100
        vocab_size, hidden_dim = 500, 128

        decoder = WaveletDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=4,
            max_sequence_length=512,
            wavelet="db4",
            levels=2,  # Lower for causality
        )

        # Decoder should have language modeling head
        assert decoder.output_type == "lm"
        assert decoder.num_classes == vocab_size

        # Test forward pass
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        output = decoder(input_ids=input_ids)
        assert output.shape == (batch_size, seq_length, vocab_size)

    def test_wavelet_complexity(self):
        """Test WaveletTransformer complexity calculation."""
        model = WaveletTransformer(
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
            wavelet="db4",
            levels=3,
        )

        complexity = model.complexity
        assert "time" in complexity
        assert "space" in complexity
        # Wavelet should have O(n) complexity per dimension
        assert "512 * 256 * 3" in complexity["time"]  # n * d * J

    def test_different_wavelets(self):
        """Test WaveletTransformer with different wavelet families."""
        batch_size, seq_length, hidden_dim = 2, 64, 128

        wavelet_types = ["db1", "db4", "db8", "sym2", "sym6", "coif1", "coif3"]

        for wavelet in wavelet_types:
            model = WaveletTransformer(
                hidden_dim=hidden_dim,
                num_layers=2,
                max_sequence_length=512,
                wavelet=wavelet,
                levels=2,
            )

            # Test forward pass
            inputs = torch.randn(batch_size, seq_length, hidden_dim)
            output = model(inputs_embeds=inputs)
            assert output.shape == (batch_size, seq_length, hidden_dim)

    def test_mixing_modes(self):
        """Test different mixing modes in WaveletTransformer."""
        batch_size, seq_length, hidden_dim = 2, 64, 128

        mixing_modes = ["pointwise", "channel"]  # "level" requires special handling

        for mode in mixing_modes:
            model = WaveletTransformer(
                hidden_dim=hidden_dim,
                num_layers=2,
                max_sequence_length=512,
                wavelet="db4",
                levels=2,
                mixing_mode=mode,
            )

            # Test forward pass
            inputs = torch.randn(batch_size, seq_length, hidden_dim)
            output = model(inputs_embeds=inputs)
            assert output.shape == (batch_size, seq_length, hidden_dim)

    def test_gradient_flow(self):
        """Test gradient flow through WaveletTransformer."""
        batch_size, seq_length = 2, 50
        vocab_size, hidden_dim = 100, 64
        num_classes = 5

        model = WaveletTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=512,
            num_classes=num_classes,
            wavelet="db4",
            levels=2,
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
                assert not torch.isnan(param.grad).any()

    def test_from_config(self):
        """Test creating WaveletTransformer from configuration."""
        from spectrans.config.models import WaveletTransformerConfig

        config = WaveletTransformerConfig(
            hidden_dim=256,
            num_layers=6,
            sequence_length=1024,
            wavelet="sym6",
            levels=4,
            mixing_mode="channel",
            dropout=0.2,
        )

        model = WaveletTransformer.from_config(config)
        assert model.hidden_dim == 256
        assert model.num_layers == 6
        assert model.max_sequence_length == 1024
        assert model.wavelet == "sym6"
        assert model.levels == 4
        assert model.mixing_mode == "channel"
        assert len(model.blocks) == 6

    def test_positional_encoding_options(self):
        """Test WaveletTransformer with different positional encoding options."""
        batch_size, seq_length = 2, 64
        vocab_size, hidden_dim = 100, 128

        # With sinusoidal positional encoding (default)
        model_sin = WaveletTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            use_positional_encoding=True,
            positional_encoding_type="sinusoidal",
        )

        # With learned positional encoding
        model_learned = WaveletTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            use_positional_encoding=True,
            positional_encoding_type="learned",
        )

        # Without positional encoding
        model_no_pe = WaveletTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            use_positional_encoding=False,
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

        # All should produce valid outputs
        for model in [model_sin, model_learned, model_no_pe]:
            output = model(input_ids=input_ids)
            assert output.shape == (batch_size, seq_length, hidden_dim)

    def test_output_types(self):
        """Test WaveletTransformer with different output types."""
        batch_size, seq_length = 2, 64
        vocab_size, hidden_dim = 100, 128
        num_classes = 10

        # Classification head
        model_cls = WaveletTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_classes=num_classes,
            output_type="classification",
        )

        # Regression head
        model_reg = WaveletTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_classes=1,
            output_type="regression",
        )

        # Sequence head
        model_seq = WaveletTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_classes=num_classes,
            output_type="sequence",
        )

        # No head
        model_none = WaveletTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            output_type="none",
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

        # Check output shapes
        assert model_cls(input_ids=input_ids).shape == (batch_size, num_classes)
        assert model_reg(input_ids=input_ids).shape == (batch_size, 1)
        assert model_seq(input_ids=input_ids).shape == (batch_size, seq_length, num_classes)
        assert model_none(input_ids=input_ids).shape == (batch_size, seq_length, hidden_dim)

