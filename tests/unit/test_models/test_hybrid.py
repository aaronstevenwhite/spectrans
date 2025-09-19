"""Unit tests for Hybrid Transformer models."""

import torch

from spectrans.models.hybrid import AlternatingTransformer, HybridEncoder, HybridTransformer


class TestHybridTransformer:
    """Test HybridTransformer model."""

    def test_hybrid_transformer_initialization(self):
        """Test HybridTransformer initialization with various configurations."""
        # Basic initialization with Fourier-Attention
        model = HybridTransformer(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
            num_classes=10,
            spectral_type="fourier",
            spatial_type="attention",
        )
        assert len(model.blocks) == 4
        assert model.spectral_type == "fourier"
        assert model.spatial_type == "attention"
        assert model.alternation_pattern == "even_spectral"

        # Wavelet-SpectralAttention hybrid
        model_wavelet = HybridTransformer(
            hidden_dim=256,
            num_layers=4,
            spectral_type="wavelet",
            spatial_type="spectral_attention",
            spectral_config={"wavelet": "db4", "levels": 3},
            spatial_config={"num_features": 256},
        )
        assert model_wavelet.spectral_type == "wavelet"
        assert model_wavelet.spatial_type == "spectral_attention"

    def test_hybrid_forward_pass(self):
        """Test HybridTransformer forward pass."""
        batch_size, seq_length = 2, 100
        vocab_size, hidden_dim = 500, 128
        num_classes = 5

        model = HybridTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=4,
            max_sequence_length=512,
            num_classes=num_classes,
            spectral_type="fourier",
            spatial_type="attention",
            num_heads=4,
            output_type="classification",
        )

        # Test with input_ids
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        output = model(input_ids=input_ids)
        assert output.shape == (batch_size, num_classes)

        # Test with inputs_embeds
        model_no_vocab = HybridTransformer(
            vocab_size=None,
            hidden_dim=hidden_dim,
            num_layers=4,
            max_sequence_length=512,
            num_classes=num_classes,
        )
        inputs_embeds = torch.randn(batch_size, seq_length, hidden_dim)
        output = model_no_vocab(inputs_embeds=inputs_embeds)
        assert output.shape == (batch_size, num_classes)

    def test_different_spectral_types(self):
        """Test HybridTransformer with different spectral mixing types."""
        batch_size, seq_length, hidden_dim = 2, 64, 128

        spectral_types = ["fourier", "wavelet", "afno", "gfnet"]

        for spectral_type in spectral_types:
            # Configure based on type
            spectral_config = {}
            if spectral_type == "wavelet":
                spectral_config = {"wavelet": "db4", "levels": 2}
            elif spectral_type == "afno":
                spectral_config = {"n_modes": 32}

            # GFNet requires fixed sequence length
            test_seq_length = 128 if spectral_type == "gfnet" else seq_length

            model = HybridTransformer(
                hidden_dim=hidden_dim,
                num_layers=4,
                max_sequence_length=128 if spectral_type == "gfnet" else 512,
                spectral_type=spectral_type,
                spatial_type="attention",
                spectral_config=spectral_config,
            )

            # Test forward pass with appropriate sequence length
            inputs = torch.randn(
                batch_size, test_seq_length if spectral_type == "gfnet" else seq_length, hidden_dim
            )
            output = model(inputs_embeds=inputs)
            assert output.shape == (
                batch_size,
                test_seq_length if spectral_type == "gfnet" else seq_length,
                hidden_dim,
            )

    def test_different_spatial_types(self):
        """Test HybridTransformer with different spatial mixing types."""
        batch_size, seq_length, hidden_dim = 2, 64, 128

        spatial_types = ["attention", "spectral_attention", "lst"]

        for spatial_type in spatial_types:
            # Configure based on type
            spatial_config = {}
            if spatial_type == "spectral_attention":
                spatial_config = {"num_features": 128}
            elif spatial_type == "lst":
                spatial_config = {"transform_type": "dct"}

            model = HybridTransformer(
                hidden_dim=hidden_dim,
                num_layers=4,
                max_sequence_length=512,
                spectral_type="fourier",
                spatial_type=spatial_type,
                spatial_config=spatial_config,
                num_heads=4,
            )

            # Test forward pass
            inputs = torch.randn(batch_size, seq_length, hidden_dim)
            output = model(inputs_embeds=inputs)
            assert output.shape == (batch_size, seq_length, hidden_dim)

    def test_alternation_patterns(self):
        """Test different alternation patterns."""
        batch_size, seq_length, hidden_dim = 2, 64, 128

        patterns = ["even_spectral", "alternate"]

        for pattern in patterns:
            model = HybridTransformer(
                hidden_dim=hidden_dim,
                num_layers=6,
                max_sequence_length=512,
                alternation_pattern=pattern,
            )

            # Test forward pass
            inputs = torch.randn(batch_size, seq_length, hidden_dim)
            output = model(inputs_embeds=inputs)
            assert output.shape == (batch_size, seq_length, hidden_dim)

    def test_hybrid_encoder(self):
        """Test HybridEncoder variant."""
        batch_size, seq_length, hidden_dim = 2, 100, 128

        encoder = HybridEncoder(
            hidden_dim=hidden_dim,
            num_layers=4,
            max_sequence_length=512,
            spectral_type="afno",
            spatial_type="lst",
            spectral_config={"n_modes": 32},
            spatial_config={"transform_type": "dct"},
        )

        # Encoder should not have vocab embedding or output head
        assert encoder.embedding is None
        assert encoder.output_head is None

        # Test forward pass
        inputs = torch.randn(batch_size, seq_length, hidden_dim)
        output = encoder(inputs_embeds=inputs)
        assert output.shape == (batch_size, seq_length, hidden_dim)

    def test_alternating_transformer(self):
        """Test AlternatingTransformer variant."""
        batch_size, seq_length, hidden_dim = 2, 64, 128

        model = AlternatingTransformer(
            hidden_dim=hidden_dim,
            num_layers=6,
            max_sequence_length=512,
            layer1_type="fourier",
            layer2_type="attention",
            layer1_config={"use_real_fft": True},
            layer2_config={},
        )

        # Test forward pass
        inputs = torch.randn(batch_size, seq_length, hidden_dim)
        output = model(inputs_embeds=inputs)
        assert output.shape == (batch_size, seq_length, hidden_dim)

    def test_gradient_flow(self):
        """Test gradient flow through HybridTransformer."""
        batch_size, seq_length = 2, 50
        vocab_size, hidden_dim = 100, 64
        num_classes = 5

        model = HybridTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=4,
            max_sequence_length=512,
            num_classes=num_classes,
            spectral_type="wavelet",
            spatial_type="spectral_attention",
            spectral_config={"wavelet": "db4", "levels": 2},
            spatial_config={"num_features": 64},
            num_heads=4,
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
        """Test creating HybridTransformer from configuration."""
        from spectrans.config.models import HybridModelConfig

        config = HybridModelConfig(
            hidden_dim=256,
            num_layers=8,
            sequence_length=1024,
            spectral_type="afno",
            spatial_type="lst",
            alternation_pattern="alternate",
            num_heads=8,
            spectral_config={"n_modes": 64},
            spatial_config={"transform_type": "dst"},
            dropout=0.2,
        )

        model = HybridTransformer.from_config(config)
        assert model.hidden_dim == 256
        assert model.num_layers == 8
        assert model.max_sequence_length == 1024
        assert model.spectral_type == "afno"
        assert model.spatial_type == "lst"
        assert model.alternation_pattern == "alternate"
        assert model.num_heads == 8
        assert len(model.blocks) == 8

    def test_mixed_configurations(self):
        """Test various mixed configurations."""
        batch_size, seq_length, hidden_dim = 2, 64, 128

        configs = [
            # AFNO + Standard Attention
            {
                "spectral_type": "afno",
                "spatial_type": "attention",
                "spectral_config": {"n_modes": 32, "compression_ratio": 0.5},
                "spatial_config": {},
            },
            # GFNet + LST
            {
                "spectral_type": "gfnet",
                "spatial_type": "lst",
                "spectral_config": {"activation": "sigmoid"},
                "spatial_config": {"transform_type": "dct"},
            },
            # Wavelet + Spectral Attention
            {
                "spectral_type": "wavelet",
                "spatial_type": "spectral_attention",
                "spectral_config": {"wavelet": "sym6", "levels": 2, "mixing_mode": "channel"},
                "spatial_config": {"num_features": 128, "kernel_type": "gaussian"},
            },
        ]

        for config_dict in configs:
            # GFNet requires matching sequence length
            is_gfnet = config_dict["spectral_type"] == "gfnet"
            test_seq_length = 128 if is_gfnet else seq_length

            model = HybridTransformer(
                hidden_dim=hidden_dim,
                num_layers=4,
                max_sequence_length=128 if is_gfnet else 512,
                num_heads=4,
                **config_dict,
            )

            # Test forward pass with appropriate sequence length
            inputs = torch.randn(batch_size, test_seq_length, hidden_dim)
            output = model(inputs_embeds=inputs)
            assert output.shape == (batch_size, test_seq_length, hidden_dim)

    def test_output_types(self):
        """Test HybridTransformer with different output types."""
        batch_size, seq_length = 2, 64
        vocab_size, hidden_dim = 100, 128
        num_classes = 10

        output_configs = [
            ("classification", (batch_size, num_classes)),
            ("regression", (batch_size, 1)),
            ("sequence", (batch_size, seq_length, num_classes)),
            ("none", (batch_size, seq_length, hidden_dim)),
        ]

        for output_type, expected_shape in output_configs:
            model = HybridTransformer(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_layers=2,
                num_classes=num_classes if output_type != "regression" else 1,
                output_type=output_type,
            )

            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
            output = model(input_ids=input_ids)
            assert output.shape == expected_shape
