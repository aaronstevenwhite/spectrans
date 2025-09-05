"""Unit tests for base model components."""

import pytest
import torch
import torch.nn as nn

from spectrans.blocks.base import PreNormBlock
from spectrans.layers.mixing.fourier import FourierMixing
from spectrans.models.base import (
    BaseModel,
    ClassificationHead,
    LearnedPositionalEncoding,
    PositionalEncoding,
    RegressionHead,
    SequenceHead,
)


class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""

    def build_blocks(self) -> nn.ModuleList:
        """Build transformer blocks for testing."""
        return nn.ModuleList([
            PreNormBlock(
                mixing_layer=FourierMixing(hidden_dim=self.hidden_dim),
                hidden_dim=self.hidden_dim,
                ffn_hidden_dim=self.ffn_hidden_dim,
                dropout=0.1,
            )
            for _ in range(self.num_layers)
        ])


class TestPositionalEncoding:
    """Test positional encoding modules."""

    def test_sinusoidal_positional_encoding(self):
        """Test sinusoidal positional encoding."""
        batch_size, seq_length, hidden_dim = 2, 100, 256
        encoder = PositionalEncoding(hidden_dim, max_sequence_length=1000)

        # Create input tensor
        x = torch.randn(batch_size, seq_length, hidden_dim)

        # Apply encoding
        output = encoder(x)

        # Check shape preservation
        assert output.shape == x.shape

        # Check that encoding is added (output != x due to positional encoding)
        assert not torch.allclose(output, x)

        # Test with different sequence lengths
        x_short = torch.randn(batch_size, 50, hidden_dim)
        output_short = encoder(x_short)
        assert output_short.shape == x_short.shape

    def test_learned_positional_encoding(self):
        """Test learned positional embeddings."""
        batch_size, seq_length, hidden_dim = 2, 100, 256
        encoder = LearnedPositionalEncoding(hidden_dim, max_sequence_length=1000)

        # Create input tensor
        x = torch.randn(batch_size, seq_length, hidden_dim)

        # Apply encoding
        output = encoder(x)

        # Check shape preservation
        assert output.shape == x.shape

        # Check that encoding is added
        assert not torch.allclose(output, x)

        # Check that embeddings are learnable (have gradients)
        loss = output.sum()
        loss.backward()
        assert encoder.position_embeddings.weight.grad is not None


class TestOutputHeads:
    """Test output head modules."""

    def test_classification_head(self):
        """Test classification output head."""
        batch_size, seq_length, hidden_dim = 4, 50, 128
        num_classes = 10

        # Test different pooling strategies
        for pooling in ["cls", "mean", "max"]:
            head = ClassificationHead(hidden_dim, num_classes, pooling=pooling)
            x = torch.randn(batch_size, seq_length, hidden_dim)

            # Test without mask
            output = head(x)
            assert output.shape == (batch_size, num_classes)

            # Test with mask
            mask = torch.ones(batch_size, seq_length)
            mask[:, seq_length//2:] = 0  # Mask second half
            output_masked = head(x, mask)
            assert output_masked.shape == (batch_size, num_classes)

            # For mean pooling, masked output should differ
            if pooling == "mean":
                assert not torch.allclose(output, output_masked)

    def test_regression_head(self):
        """Test regression output head."""
        batch_size, seq_length, hidden_dim = 4, 50, 128

        for pooling in ["cls", "mean", "max"]:
            head = RegressionHead(hidden_dim, pooling=pooling)
            x = torch.randn(batch_size, seq_length, hidden_dim)

            # Test output shape
            output = head(x)
            assert output.shape == (batch_size, 1)

            # Test with mask
            mask = torch.ones(batch_size, seq_length)
            mask[:, seq_length//2:] = 0
            output_masked = head(x, mask)
            assert output_masked.shape == (batch_size, 1)

    def test_sequence_head(self):
        """Test sequence-to-sequence output head."""
        batch_size, seq_length, hidden_dim = 4, 50, 128
        vocab_size = 1000

        head = SequenceHead(hidden_dim, vocab_size)
        x = torch.randn(batch_size, seq_length, hidden_dim)

        # Test output shape
        output = head(x)
        assert output.shape == (batch_size, seq_length, vocab_size)


class TestBaseModel:
    """Test BaseModel functionality."""

    def test_model_initialization(self):
        """Test model initialization with various configurations."""
        # Test with token embeddings
        model = ConcreteModel(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
            num_classes=10,
        )
        assert model.embedding is not None
        assert model.positional_encoding is not None
        assert model.output_head is not None
        assert len(model.blocks) == 4

        # Test without token embeddings
        model_no_embed = ConcreteModel(
            vocab_size=None,
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
            num_classes=10,
        )
        assert model_no_embed.embedding is None

        # Test without positional encoding
        model_no_pos = ConcreteModel(
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
            use_positional_encoding=False,
        )
        assert model_no_pos.positional_encoding is None

        # Test with learned positional encoding
        model_learned_pos = ConcreteModel(
            hidden_dim=256,
            num_layers=4,
            max_sequence_length=512,
            positional_encoding_type="learned",
        )
        assert isinstance(model_learned_pos.positional_encoding, LearnedPositionalEncoding)

    def test_forward_with_input_ids(self):
        """Test forward pass with input token IDs."""
        batch_size, seq_length = 2, 100
        vocab_size, hidden_dim = 1000, 128
        num_classes = 10

        model = ConcreteModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=512,
            num_classes=num_classes,
            output_type="classification",
        )

        # Create input IDs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

        # Forward pass
        output = model(input_ids=input_ids)

        # Check output shape for classification
        assert output.shape == (batch_size, num_classes)

    def test_forward_with_inputs_embeds(self):
        """Test forward pass with pre-embedded inputs."""
        batch_size, seq_length, hidden_dim = 2, 100, 128

        model = ConcreteModel(
            vocab_size=None,  # No embedding layer
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=512,
            output_type="none",  # Return hidden states
        )

        # Create embedded inputs
        inputs_embeds = torch.randn(batch_size, seq_length, hidden_dim)

        # Forward pass
        output = model(inputs_embeds=inputs_embeds)

        # Check output shape
        assert output.shape == (batch_size, seq_length, hidden_dim)

    def test_different_output_types(self):
        """Test model with different output head types."""
        batch_size, seq_length, hidden_dim = 2, 50, 128

        # Test regression output
        model_reg = ConcreteModel(
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=512,
            output_type="regression",
        )
        x = torch.randn(batch_size, seq_length, hidden_dim)
        output = model_reg(inputs_embeds=x)
        assert output.shape == (batch_size, 1)

        # Test sequence output
        vocab_size = 500
        model_seq = ConcreteModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=512,
            output_type="sequence",
        )
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        output = model_seq(input_ids=input_ids)
        assert output.shape == (batch_size, seq_length, vocab_size)

    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        batch_size, seq_length = 2, 50
        vocab_size, hidden_dim = 100, 64
        num_classes = 5

        model = ConcreteModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            max_sequence_length=512,
            num_classes=num_classes,
            output_type="classification",
        )

        # Create input and target
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        target = torch.randint(0, num_classes, (batch_size,))

        # Forward pass
        output = model(input_ids=input_ids)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(output, target)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        model = ConcreteModel(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=2,
            max_sequence_length=512,
        )

        # Test with no inputs
        with pytest.raises(ValueError, match="Either input_ids or inputs_embeds"):
            model()

        # Test with input_ids but no embedding layer
        model_no_embed = ConcreteModel(
            vocab_size=None,
            hidden_dim=128,
            num_layers=2,
            max_sequence_length=512,
        )
        input_ids = torch.randint(0, 100, (2, 50))
        with pytest.raises(ValueError, match="no embedding layer"):
            model_no_embed(input_ids=input_ids)
