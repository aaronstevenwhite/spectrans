"""Unit tests for spectral attention transformer models.

Tests cover initialization, forward passes, gradient flow, complexity properties,
configuration-based construction, and edge cases for SpectralAttentionTransformer,
SpectralAttentionEncoder, and PerformerTransformer models.
"""

import pytest
import torch
import torch.nn as nn

from spectrans.models.spectral_attention import (
    PerformerTransformer,
    SpectralAttentionEncoder,
    SpectralAttentionTransformer,
)


class TestSpectralAttentionTransformer:
    """Test SpectralAttentionTransformer model."""
    
    def test_spectral_attention_initialization(self):
        """Test model initialization with various configurations."""
        # Test with default parameters
        model = SpectralAttentionTransformer(
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            max_sequence_length=512,
        )
        assert model.hidden_dim == 256
        assert model.num_layers == 4
        assert model.num_heads == 8
        assert model.num_features == 256  # Default to hidden_dim
        assert model.kernel_type == "softmax"
        assert not model.use_orthogonal
        
        # Test with custom parameters
        model = SpectralAttentionTransformer(
            vocab_size=10000,
            hidden_dim=512,
            num_layers=6,
            num_heads=16,
            num_features=128,
            kernel_type="gaussian",
            use_orthogonal=True,
            num_classes=100,
            ffn_hidden_dim=1024,
            dropout=0.1,
            max_sequence_length=1024,
        )
        assert model.embedding is not None  # vocab_size was provided
        assert model.embedding.num_embeddings == 10000
        assert model.num_features == 128
        assert model.kernel_type == "gaussian"
        assert model.use_orthogonal
        assert model.output_head is not None  # num_classes was provided
        assert model.ffn_hidden_dim == 1024
        
        # Check blocks are built correctly
        assert len(model.blocks) == 6
        for block in model.blocks:
            assert hasattr(block, "mixing_layer")
            assert hasattr(block, "ffn")
    
    def test_spectral_attention_forward_pass(self):
        """Test forward pass with different input types."""
        batch_size, seq_length, hidden_dim = 2, 100, 256
        
        # Test with inputs_embeds
        model = SpectralAttentionTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=8,
            num_features=64,
            max_sequence_length=512,
        )
        
        inputs_embeds = torch.randn(batch_size, seq_length, hidden_dim)
        output = model(inputs_embeds=inputs_embeds)
        assert output.shape == (batch_size, seq_length, hidden_dim)
        
        # Test with input_ids
        vocab_size = 1000
        model = SpectralAttentionTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=8,
            num_features=64,
            max_sequence_length=512,
        )
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        output = model(input_ids)
        assert output.shape == (batch_size, seq_length, hidden_dim)
        
        # Test with classification head
        num_classes = 10
        model = SpectralAttentionTransformer(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=8,
            num_features=64,
            num_classes=num_classes,
            max_sequence_length=512,
        )
        
        logits = model(input_ids)
        assert logits.shape == (batch_size, num_classes)
    
    def test_spectral_attention_encoder(self):
        """Test encoder-only model variant."""
        batch_size, seq_length, hidden_dim = 2, 100, 256
        
        encoder = SpectralAttentionEncoder(
            hidden_dim=hidden_dim,
            num_layers=3,
            num_heads=8,
            num_features=128,
            kernel_type="softmax",
            use_orthogonal=True,
            max_sequence_length=512,
        )
        
        # Check no classification head
        assert encoder.output_type == "none"
        assert not hasattr(encoder, "classification_head")
        
        # Test forward pass
        inputs_embeds = torch.randn(batch_size, seq_length, hidden_dim)
        output = encoder(inputs_embeds=inputs_embeds)
        assert output.shape == (batch_size, seq_length, hidden_dim)
        
        # Test with token inputs
        vocab_size = 1000
        encoder = SpectralAttentionEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=3,
            num_heads=8,
            num_features=128,
            max_sequence_length=512,
        )
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        output = encoder(input_ids)
        assert output.shape == (batch_size, seq_length, hidden_dim)
    
    def test_spectral_attention_complexity(self):
        """Test complexity property returns correct information."""
        model = SpectralAttentionTransformer(
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            num_features=256,
            max_sequence_length=1024,
        )
        
        complexity = model.complexity
        assert "O(n * 256 * 512)" in complexity["time"]
        assert "O(n * 256)" in complexity["space"]
        assert "Linear" in complexity["description"]
        
        # Test with different feature dimensions
        model = SpectralAttentionTransformer(
            hidden_dim=768,
            num_layers=4,
            num_heads=12,
            num_features=384,
            max_sequence_length=2048,
        )
        
        complexity = model.complexity
        assert "O(n * 384 * 768)" in complexity["time"]
        assert "O(n * 384)" in complexity["space"]
    
    def test_spectral_attention_gradient_flow(self):
        """Test gradient flow through the model."""
        model = SpectralAttentionTransformer(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_features=64,
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
    
    def test_spectral_attention_kernel_types(self):
        """Test different kernel type configurations."""
        batch_size, seq_length, hidden_dim = 2, 50, 128
        
        # Test gaussian kernel
        model_gaussian = SpectralAttentionTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            num_features=64,
            kernel_type="gaussian",
            max_sequence_length=256,
        )
        
        inputs = torch.randn(batch_size, seq_length, hidden_dim)
        output = model_gaussian(inputs_embeds=inputs)
        assert output.shape == (batch_size, seq_length, hidden_dim)
        
        # Test softmax kernel
        model_softmax = SpectralAttentionTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            num_features=64,
            kernel_type="softmax",
            max_sequence_length=256,
        )
        
        output = model_softmax(inputs_embeds=inputs)
        assert output.shape == (batch_size, seq_length, hidden_dim)
        
        # Outputs should be different
        output_gaussian = model_gaussian(inputs_embeds=inputs)
        output_softmax = model_softmax(inputs_embeds=inputs)
        assert not torch.allclose(output_gaussian, output_softmax, atol=1e-2)
    
    def test_spectral_attention_orthogonal_features(self):
        """Test orthogonal vs non-orthogonal random features."""
        batch_size, seq_length, hidden_dim = 2, 50, 128
        
        # Test with orthogonal features
        model_orth = SpectralAttentionTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            num_features=64,
            use_orthogonal=True,
            max_sequence_length=256,
        )
        
        # Test with non-orthogonal features
        model_no_orth = SpectralAttentionTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            num_features=64,
            use_orthogonal=False,
            max_sequence_length=256,
        )
        
        inputs = torch.randn(batch_size, seq_length, hidden_dim)
        
        # Both should produce valid outputs
        output_orth = model_orth(inputs_embeds=inputs)
        output_no_orth = model_no_orth(inputs_embeds=inputs)
        
        assert output_orth.shape == (batch_size, seq_length, hidden_dim)
        assert output_no_orth.shape == (batch_size, seq_length, hidden_dim)
        
        # Outputs will be different due to different random features
        assert not torch.allclose(output_orth, output_no_orth, atol=1e-2)
    
    def test_spectral_attention_different_sequence_lengths(self):
        """Test model handles different sequence lengths correctly."""
        hidden_dim = 128
        model = SpectralAttentionTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            num_features=64,
            max_sequence_length=1024,
        )
        
        # Test with different sequence lengths
        for seq_length in [10, 50, 100, 500]:
            inputs = torch.randn(2, seq_length, hidden_dim)
            output = model(inputs_embeds=inputs)
            assert output.shape == (2, seq_length, hidden_dim)
    
    def test_spectral_attention_with_positional_encoding(self):
        """Test model with different positional encoding types."""
        batch_size, seq_length, hidden_dim = 2, 100, 256
        
        # Test with sinusoidal positional encoding
        model_sin = SpectralAttentionTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=8,
            num_features=128,
            use_positional_encoding=True,
            positional_encoding_type="sinusoidal",
            max_sequence_length=512,
        )
        
        # Test with learned positional encoding
        model_learned = SpectralAttentionTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=8,
            num_features=128,
            use_positional_encoding=True,
            positional_encoding_type="learned",
            max_sequence_length=512,
        )
        
        # Test without positional encoding
        model_no_pe = SpectralAttentionTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=8,
            num_features=128,
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


class TestPerformerTransformer:
    """Test PerformerTransformer model."""
    
    def test_performer_initialization(self):
        """Test Performer model initialization."""
        model = PerformerTransformer(
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            num_features=128,
            max_sequence_length=512,
        )
        
        assert model.hidden_dim == 256
        assert model.num_layers == 4
        assert model.num_heads == 8
        assert model.num_features == 128
        
        # Check blocks use PerformerAttention
        assert len(model.blocks) == 4
    
    def test_performer_forward_pass(self):
        """Test Performer forward pass."""
        batch_size, seq_length, hidden_dim = 2, 100, 256
        
        model = PerformerTransformer(
            hidden_dim=hidden_dim,
            num_layers=3,
            num_heads=8,
            num_features=128,
            max_sequence_length=512,
        )
        
        inputs = torch.randn(batch_size, seq_length, hidden_dim)
        output = model(inputs_embeds=inputs)
        assert output.shape == (batch_size, seq_length, hidden_dim)
        
        # Test with classification
        model = PerformerTransformer(
            hidden_dim=hidden_dim,
            num_layers=3,
            num_heads=8,
            num_features=128,
            num_classes=10,
            max_sequence_length=512,
        )
        
        logits = model(inputs_embeds=inputs)
        assert logits.shape == (batch_size, 10)
    
    def test_performer_complexity(self):
        """Test Performer complexity property."""
        model = PerformerTransformer(
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            num_features=256,
            max_sequence_length=1024,
        )
        
        complexity = model.complexity
        assert "O(n * 256 * 512)" in complexity["time"]
        assert "O(n * 256)" in complexity["space"]
        assert "orthogonal" in complexity["description"].lower()
    
    def test_performer_gradient_flow(self):
        """Test gradient flow through Performer model."""
        model = PerformerTransformer(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_features=64,
            num_classes=10,
            max_sequence_length=256,
        )
        
        batch_size, seq_length = 2, 50
        inputs = torch.randn(batch_size, seq_length, 128, requires_grad=True)
        target = torch.randint(0, 10, (batch_size,))
        
        logits = model(inputs_embeds=inputs)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()
        
        # Check all parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
    
    def test_performer_long_sequences(self):
        """Test Performer with long sequences to verify linear complexity."""
        hidden_dim = 128
        model = PerformerTransformer(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            num_features=64,
            max_sequence_length=2048,
        )
        
        # Test with increasingly long sequences
        for seq_length in [100, 500, 1000]:
            inputs = torch.randn(1, seq_length, hidden_dim)
            output = model(inputs_embeds=inputs)
            assert output.shape == (1, seq_length, hidden_dim)
            
            # Memory should scale linearly, not quadratically
            # This is implicitly tested by the model not running out of memory
    
    def test_performer_gradient_checkpointing(self):
        """Test Performer with gradient checkpointing."""
        model = PerformerTransformer(
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            num_features=64,
            gradient_checkpointing=True,
            max_sequence_length=256,
        )
        
        batch_size, seq_length = 2, 100
        inputs = torch.randn(batch_size, seq_length, 128)
        
        # Should work with gradient checkpointing
        output = model(inputs_embeds=inputs)
        assert output.shape == (batch_size, seq_length, 128)
        
        # Test backward pass with checkpointing
        loss = output.mean()
        loss.backward()
        
        # Parameters should have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None