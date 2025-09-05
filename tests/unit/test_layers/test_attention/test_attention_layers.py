"""Unit tests for attention layers (combined for brevity)."""

import torch

from spectrans.layers.attention import (
    DCTAttention,
    HadamardAttention,
    KernelAttention,
    LSTAttention,
    MixedSpectralAttention,
    PerformerAttention,
    SpectralAttention,
)


class TestSpectralAttention:
    """Test spectral attention mechanisms."""

    def test_spectral_attention_forward_shape(self, random_tensor):
        """Test SpectralAttention output shape."""
        batch_size, seq_len, hidden_dim = random_tensor.shape

        attn = SpectralAttention(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_features=64,
        )

        output = attn(random_tensor)
        assert output.shape == random_tensor.shape

    def test_spectral_attention_with_mask(self, random_tensor, attention_mask):
        """Test SpectralAttention with attention mask."""
        batch_size, seq_len, hidden_dim = random_tensor.shape

        attn = SpectralAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            kernel_type="softmax",
        )

        output = attn(random_tensor, mask=attention_mask)
        assert output.shape == random_tensor.shape

        # Check that masked positions don't affect output significantly
        output_no_mask = attn(random_tensor)
        assert not torch.allclose(output, output_no_mask)

    def test_performer_attention_shapes(self, random_tensor):
        """Test PerformerAttention output shape."""
        batch_size, seq_len, hidden_dim = random_tensor.shape

        performer = PerformerAttention(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_features=32,
        )

        output = performer(random_tensor)
        assert output.shape == random_tensor.shape

    def test_performer_generalized_attention(self, random_tensor):
        """Test generalized Performer attention."""
        batch_size, seq_len, hidden_dim = random_tensor.shape

        performer = PerformerAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            generalized=True,
        )

        output = performer(random_tensor)
        assert output.shape == random_tensor.shape

    def test_kernel_attention_types(self, random_tensor):
        """Test different kernel types in KernelAttention."""
        batch_size, seq_len, hidden_dim = random_tensor.shape

        # Test Gaussian kernel
        attn_gaussian = KernelAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            kernel_type="gaussian",
            num_features=32,
        )
        output_gaussian = attn_gaussian(random_tensor)
        assert output_gaussian.shape == random_tensor.shape

        # Test polynomial kernel
        attn_poly = KernelAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            kernel_type="polynomial",
            rank=16,
        )
        output_poly = attn_poly(random_tensor)
        assert output_poly.shape == random_tensor.shape

        # Test spectral kernel
        attn_spectral = KernelAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            kernel_type="spectral",
            rank=16,
        )
        output_spectral = attn_spectral(random_tensor)
        assert output_spectral.shape == random_tensor.shape

    def test_attention_gradient_flow(self, random_tensor):
        """Test gradient flow through spectral attention."""
        random_tensor.requires_grad_(True)

        attn = SpectralAttention(
            hidden_dim=random_tensor.shape[-1],
            num_heads=4,
        )

        output = attn(random_tensor)
        loss = output.mean()
        loss.backward()

        assert random_tensor.grad is not None
        assert not torch.all(random_tensor.grad == 0)


class TestLSTAttention:
    """Test LST attention mechanisms."""

    def test_lst_attention_shapes(self, random_tensor):
        """Test LSTAttention output shape."""
        batch_size, seq_len, hidden_dim = random_tensor.shape

        attn = LSTAttention(
            hidden_dim=hidden_dim,
            num_heads=8,
            transform_type="dct",
        )

        output = attn(random_tensor)
        assert output.shape == random_tensor.shape

    def test_lst_transform_types(self, random_tensor):
        """Test different transform types in LST."""
        batch_size, seq_len, hidden_dim = random_tensor.shape

        # Test DCT
        attn_dct = LSTAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            transform_type="dct",
        )
        output_dct = attn_dct(random_tensor)
        assert output_dct.shape == random_tensor.shape

        # Test DST
        attn_dst = LSTAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            transform_type="dst",
        )
        output_dst = attn_dst(random_tensor)
        assert output_dst.shape == random_tensor.shape

        # Test Hadamard
        attn_hadamard = LSTAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            transform_type="hadamard",
        )
        output_hadamard = attn_hadamard(random_tensor)
        assert output_hadamard.shape == random_tensor.shape

        # Test mixed transforms
        attn_mixed = LSTAttention(
            hidden_dim=hidden_dim,
            num_heads=8,  # Divisible by hidden_dim
            transform_type="mixed",
        )
        output_mixed = attn_mixed(random_tensor)
        assert output_mixed.shape == random_tensor.shape

    def test_dct_attention(self, random_tensor):
        """Test specialized DCT attention."""
        batch_size, seq_len, hidden_dim = random_tensor.shape

        attn = DCTAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            dct_type=2,
        )

        output = attn(random_tensor)
        assert output.shape == random_tensor.shape

    def test_hadamard_attention(self, random_tensor):
        """Test specialized Hadamard attention."""
        batch_size, seq_len, hidden_dim = random_tensor.shape

        attn = HadamardAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            scale_by_sqrt=True,
        )

        output = attn(random_tensor)
        assert output.shape == random_tensor.shape

    def test_mixed_spectral_attention(self, random_tensor):
        """Test mixed spectral attention."""
        batch_size, seq_len, hidden_dim = random_tensor.shape

        attn = MixedSpectralAttention(
            hidden_dim=hidden_dim,
            num_heads=8,  # Divisible by hidden_dim
            use_fft=True,
            use_dct=True,
            use_hadamard=True,
        )

        output = attn(random_tensor)
        assert output.shape == random_tensor.shape

    def test_lst_with_mask(self, random_tensor, attention_mask):
        """Test LST attention with mask."""
        batch_size, seq_len, hidden_dim = random_tensor.shape

        attn = LSTAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            transform_type="dct",
        )

        output = attn(random_tensor, mask=attention_mask)
        assert output.shape == random_tensor.shape

    def test_lst_gradient_flow(self, random_tensor):
        """Test gradient flow through LST attention."""
        random_tensor.requires_grad_(True)

        attn = LSTAttention(
            hidden_dim=random_tensor.shape[-1],
            num_heads=4,
            transform_type="dct",
        )

        output = attn(random_tensor)
        loss = output.mean()
        loss.backward()

        assert random_tensor.grad is not None
        assert not torch.all(random_tensor.grad == 0)




class TestAttentionTraining:
    """Test attention layers in training scenarios."""

    def test_attention_dropout(self, random_tensor):
        """Test dropout in attention layers."""
        batch_size, seq_len, hidden_dim = random_tensor.shape

        attn = SpectralAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=0.5,
        )

        # Training mode
        attn.train()
        output1 = attn(random_tensor)
        output2 = attn(random_tensor)

        # Outputs should differ due to dropout
        assert not torch.equal(output1, output2)

        # Eval mode
        attn.eval()
        output3 = attn(random_tensor)
        output4 = attn(random_tensor)

        # Outputs should be the same
        assert torch.allclose(output3, output4)

    def test_learnable_parameters(self):
        """Test that attention layers have learnable parameters."""
        models = [
            SpectralAttention(hidden_dim=256, num_heads=4),
            LSTAttention(hidden_dim=256, num_heads=4, learnable_scale=True),
            KernelAttention(hidden_dim=256, num_heads=4, kernel_type="spectral"),
        ]

        for model in models:
            params = list(model.parameters())
            assert len(params) > 0

            # Check parameters require gradients
            for param in params:
                assert param.requires_grad

    def test_attention_return_weights_flag(self, random_tensor):
        """Test return_attention flag (should return None for linear attention)."""
        batch_size, seq_len, hidden_dim = random_tensor.shape

        # SpectralAttention doesn't compute explicit weights
        attn = SpectralAttention(hidden_dim=hidden_dim, num_heads=4)
        output, weights = attn(random_tensor, return_attention=True)

        assert output.shape == random_tensor.shape
        assert weights is None  # Linear attention doesn't have explicit weights
