"""Unit tests for AFNO models."""

import torch

from spectrans.models.afno import AFNOEncoder, AFNOModel


class TestAFNO:
    """Test AFNO model."""

    def test_afno_initialization(self):
        """Test AFNO initialization with various configurations."""
        # Basic AFNO
        model = AFNOModel(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
            num_classes=10,
        )
        assert len(model.blocks) == 4
        assert model.modes_seq == 256  # Default: max_seq_length // 2
        assert model.modes_hidden == 128  # Default: hidden_dim // 2
        assert model.mlp_ratio == 2.0  # Default value

        # AFNO with custom modes
        model_custom = AFNOModel(
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=1024,
            modes_seq=128,
            modes_hidden=64,
            mlp_ratio=4.0,
        )
        assert model_custom.modes_seq == 128
        assert model_custom.modes_hidden == 64
        assert model_custom.mlp_ratio == 4.0

    def test_afno_forward_pass(self):
        """Test AFNO forward pass."""
        batch_size, seq_length = 2, 128  # Smaller for testing
        vocab_size, hidden_dim = 500, 128
        num_classes = 5

        model = AFNOModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            num_classes=num_classes,
            output_type="classification",
            modes_seq=32,  # Small for testing
            modes_hidden=32,
        )

        # Test with input_ids
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        output = model(input_ids=input_ids)
        assert output.shape == (batch_size, num_classes)

        # Test with inputs_embeds
        model_no_vocab = AFNOModel(
            vocab_size=None,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            num_classes=num_classes,
            modes_seq=32,
            modes_hidden=32,
        )
        inputs_embeds = torch.randn(batch_size, seq_length, hidden_dim)
        output = model_no_vocab(inputs_embeds=inputs_embeds)
        assert output.shape == (batch_size, num_classes)

    def test_afno_encoder(self):
        """Test AFNOEncoder variant."""
        batch_size, seq_length, hidden_dim = 2, 128, 128

        encoder = AFNOEncoder(
            hidden_dim=hidden_dim,
            num_layers=4,
            max_sequence_length=seq_length,
            modes_seq=32,
            modes_hidden=32,
        )

        # Encoder should not have vocab embedding or output head
        assert encoder.embedding is None
        assert encoder.output_head is None

        # Test forward pass
        inputs = torch.randn(batch_size, seq_length, hidden_dim)
        output = encoder(inputs_embeds=inputs)
        assert output.shape == (batch_size, seq_length, hidden_dim)

    def test_afno_complexity(self):
        """Test AFNO complexity calculation."""
        model = AFNOModel(
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=1024,
            modes_seq=128,
            modes_hidden=64,
        )

        complexity = model.complexity
        assert "time" in complexity
        assert "space" in complexity
        # AFNO should have mode-dependent complexity
        assert "128" in complexity["time"]  # modes_seq
        assert "64" in complexity["space"]  # modes_hidden

    def test_afno_gradient_flow(self):
        """Test gradient flow through AFNO."""
        batch_size, seq_length = 2, 64  # Small for testing
        vocab_size, hidden_dim = 100, 64
        num_classes = 5

        model = AFNOModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            num_classes=num_classes,
            modes_seq=16,  # Small for testing
            modes_hidden=16,
            mlp_ratio=1.0,  # Smaller ratio for testing
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

    def test_afno_long_sequence(self):
        """Test AFNO with long sequences (efficiency test)."""
        batch_size = 1  # Small batch for memory
        seq_length = 1024  # Long sequence
        hidden_dim = 128

        # AFNO should handle long sequences efficiently with mode truncation
        model = AFNOModel(
            vocab_size=None,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            modes_seq=64,  # Aggressive truncation
            modes_hidden=32,  # Aggressive truncation
            output_type="none",
        )

        inputs = torch.randn(batch_size, seq_length, hidden_dim)
        output = model(inputs_embeds=inputs)
        assert output.shape == (batch_size, seq_length, hidden_dim)

    def test_afno_mode_truncation_effect(self):
        """Test the effect of different mode truncation levels."""
        batch_size, seq_length, hidden_dim = 2, 128, 128

        # Model with minimal truncation
        model_full = AFNOModel(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            modes_seq=seq_length // 2,  # Keep half the modes
            modes_hidden=hidden_dim // 2,
            output_type="none",
        )

        # Model with aggressive truncation
        model_truncated = AFNOModel(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            modes_seq=16,  # Keep only 16 modes
            modes_hidden=16,
            output_type="none",
        )

        inputs = torch.randn(batch_size, seq_length, hidden_dim)

        # Both should produce valid outputs
        output_full = model_full(inputs_embeds=inputs)
        output_truncated = model_truncated(inputs_embeds=inputs)

        assert output_full.shape == (batch_size, seq_length, hidden_dim)
        assert output_truncated.shape == (batch_size, seq_length, hidden_dim)

        # Both should be valid tensors
        assert torch.isfinite(output_full).all()
        assert torch.isfinite(output_truncated).all()

    def test_afno_mlp_ratio(self):
        """Test AFNO with different MLP expansion ratios."""
        batch_size, seq_length, hidden_dim = 2, 64, 128

        # Test different MLP ratios
        for mlp_ratio in [1.0, 2.0, 4.0]:
            model = AFNOModel(
                hidden_dim=hidden_dim,
                num_layers=2,
                max_sequence_length=seq_length,
                modes_seq=32,
                modes_hidden=32,
                mlp_ratio=mlp_ratio,
                output_type="none",
            )

            inputs = torch.randn(batch_size, seq_length, hidden_dim)
            output = model(inputs_embeds=inputs)

            assert output.shape == (batch_size, seq_length, hidden_dim)
            assert torch.isfinite(output).all()

    def test_afno_encoder_with_different_modes(self):
        """Test AFNOEncoder with various mode configurations."""
        batch_size, seq_length, hidden_dim = 2, 256, 128

        # Test with symmetric modes
        encoder_symmetric = AFNOEncoder(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            modes_seq=64,
            modes_hidden=64,
        )

        # Test with asymmetric modes
        encoder_asymmetric = AFNOEncoder(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=seq_length,
            modes_seq=128,
            modes_hidden=32,
        )

        inputs = torch.randn(batch_size, seq_length, hidden_dim)

        # Both should work
        output_symmetric = encoder_symmetric(inputs_embeds=inputs)
        output_asymmetric = encoder_asymmetric(inputs_embeds=inputs)

        assert output_symmetric.shape == (batch_size, seq_length, hidden_dim)
        assert output_asymmetric.shape == (batch_size, seq_length, hidden_dim)

    def test_afno_with_gradient_checkpointing(self):
        """Test AFNO with gradient checkpointing for memory efficiency."""
        model = AFNOModel(
            hidden_dim=128,
            num_layers=4,
            max_sequence_length=256,
            modes_seq=32,
            modes_hidden=32,
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
