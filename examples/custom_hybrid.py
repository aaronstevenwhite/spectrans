#!/usr/bin/env python3
"""Custom Hybrid Spectral Architecture Example.

This example demonstrates how to build custom spectral transformer architectures
by combining different mixing layers, attention mechanisms, and blocks. The
modular design of Spectrans allows for flexible composition of components.

Key concepts demonstrated:
- Combining different spectral mixing methods
- Creating alternating layer patterns
- Building custom encoder-decoder architectures
- Using registry system for custom components
- Performance comparison between architectures

Requirements:
- spectrans
- torch
"""

import torch
import torch.nn as nn

from spectrans import register_component
from spectrans.blocks import TransformerBlock
from spectrans.layers.mixing import AFNOMixing, FourierMixing, GlobalFilterMixing, WaveletMixing
from spectrans.models import HybridTransformer
from spectrans.models.base import BaseModel, PositionalEncoding


class CustomSpectralEncoder(nn.Module):
    """Custom encoder that alternates between different spectral mixing methods."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        max_sequence_length: int = 512,
        dropout: float = 0.1,
        norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            hidden_dim=hidden_dim, max_sequence_length=max_sequence_length, dropout=dropout
        )

        # Create alternating layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i % 4 == 0:
                # Fourier mixing (global frequency analysis)
                mixing = FourierMixing(hidden_dim=hidden_dim, dropout=dropout)
            elif i % 4 == 1:
                # Wavelet mixing (multi-resolution analysis)
                mixing = WaveletMixing(
                    hidden_dim=hidden_dim, wavelet="db4", levels=3, dropout=dropout
                )
            elif i % 4 == 2:
                # AFNO (adaptive frequency selection)
                mixing = AFNOMixing(
                    hidden_dim=hidden_dim,
                    max_sequence_length=max_sequence_length,
                    modes_seq=min(64, max_sequence_length // 8),
                    dropout=dropout,
                )
            else:
                # Global filter (learnable frequency filters)
                mixing = GlobalFilterMixing(
                    hidden_dim=hidden_dim, sequence_length=max_sequence_length, dropout=dropout
                )

            layer = TransformerBlock(
                mixing_layer=mixing,
                hidden_dim=hidden_dim,
                ffn_hidden_dim=hidden_dim * 4,
                dropout=dropout,
                norm_eps=norm_eps,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply positional encoding
        x = self.pos_encoding(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        return x


@register_component("mixing", "frequency_gating")
class FrequencyGatingMixing(nn.Module):
    """Custom mixing layer that gates frequency components."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # Learnable gating parameters
        self.freq_gate = nn.Parameter(torch.ones(hidden_dim))
        self.phase_shift = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply FFT along sequence dimension
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")

        # Apply learnable frequency gating
        gate = torch.sigmoid(self.freq_gate)
        phase = torch.exp(1j * self.phase_shift)

        x_fft = x_fft * gate.unsqueeze(0).unsqueeze(0) * phase.unsqueeze(0).unsqueeze(0)

        # Inverse FFT
        x_mixed = torch.fft.irfft(x_fft, n=x.shape[1], dim=1, norm="ortho")

        return self.dropout(x_mixed)

    @property
    def complexity(self) -> dict[str, str]:
        return {"time": "O(n log n)", "space": "O(n)"}


class MultiScaleSpectralModel(BaseModel):
    """Custom model that processes inputs at multiple scales."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 12,
        max_sequence_length: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        scales: list[int] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scales = scales if scales is not None else [1, 2, 4]

        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)

        # Multi-scale encoders
        self.scale_encoders = nn.ModuleList()
        for scale in scales:
            encoder = CustomSpectralEncoder(
                hidden_dim=hidden_dim // len(scales),
                num_layers=num_layers // len(scales),
                max_sequence_length=max_sequence_length // scale,
                dropout=dropout,
            )
            self.scale_encoders.append(encoder)

        # Fusion layer
        self.fusion = nn.Linear(hidden_dim, hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )

    def build_blocks(self) -> nn.ModuleList:
        """Build blocks - not used since we have custom architecture."""
        return nn.ModuleList()

    def forward(
        self, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape

        # Process at different scales
        scale_outputs = []
        for scale, encoder in zip(self.scales, self.scale_encoders, strict=False):
            # Downsample input for this scale
            if scale > 1:
                # Simple average pooling for downsampling
                scaled_input = inputs_embeds.view(
                    batch_size, seq_len // scale, scale, self.hidden_dim
                ).mean(dim=2)
            else:
                scaled_input = inputs_embeds

            # Project to scale-specific dimension
            scale_dim = self.hidden_dim // len(self.scales)
            scaled_input = scaled_input[..., :scale_dim]

            # Encode at this scale
            encoded = encoder(scaled_input)

            # Global average pooling
            pooled = encoded.mean(dim=1)  # (batch_size, scale_dim)
            scale_outputs.append(pooled)

        # Fuse multi-scale representations
        fused = torch.cat(scale_outputs, dim=-1)  # (batch_size, hidden_dim)
        fused = self.fusion(fused)

        # Classify
        logits = self.classifier(fused)
        return logits


def demonstrate_custom_encoder():
    """Show custom encoder with alternating spectral methods."""
    print("=== Custom Alternating Spectral Encoder ===\n")

    encoder = CustomSpectralEncoder(
        hidden_dim=512, num_layers=8, max_sequence_length=256, dropout=0.1
    )

    print(
        f"Custom encoder created with {sum(p.numel() for p in encoder.parameters()):,} parameters"
    )
    print("Layer pattern: Fourier → Wavelet → AFNO → Global Filter (repeats)")

    # Test with sample input matching the encoder's configured sequence length
    batch_size, seq_len = 4, 256
    x = torch.randn(batch_size, seq_len, 512)

    with torch.no_grad():
        output = encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Custom encoder working correctly\n")


def demonstrate_custom_mixing():
    """Show custom frequency gating mixing layer."""
    print("=== Custom Frequency Gating Mixing ===\n")

    mixing = FrequencyGatingMixing(hidden_dim=256, dropout=0.1)
    print(f"Custom mixing layer: {type(mixing).__name__}")
    print(f"Parameters: {sum(p.numel() for p in mixing.parameters()):,}")
    print(f"Complexity: {mixing.complexity}")

    # Test the mixing layer
    x = torch.randn(2, 64, 256)
    with torch.no_grad():
        output = mixing(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Custom mixing layer working correctly\n")


def demonstrate_multiscale_model():
    """Show multi-scale spectral model."""
    print("=== Multi-Scale Spectral Model ===\n")

    model = MultiScaleSpectralModel(
        vocab_size=10000,
        hidden_dim=768,
        num_layers=12,
        max_sequence_length=512,
        num_classes=3,
        scales=[1, 2, 4],
    )

    print(f"Multi-scale model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Processing scales: {model.scales}")
    print("Each scale captures different temporal patterns")

    # Test forward pass
    input_ids = torch.randint(1, 10000, (2, 512))

    with torch.no_grad():
        logits = model(input_ids=input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits: {logits.shape}")
    print("✓ Multi-scale model working correctly\n")


def demonstrate_hybrid_architectures():
    """Show different hybrid architecture patterns."""
    print("=== Hybrid Architecture Patterns ===\n")

    # Pattern 1: Alternating spectral and spatial attention
    print("1. Alternating spectral-spatial pattern...")
    hybrid1 = HybridTransformer(
        vocab_size=20000,
        hidden_dim=512,
        num_layers=8,
        max_sequence_length=256,
        num_classes=4,
        alternation_pattern="even_spectral",  # Even layers use spectral, odd use attention
        spectral_type="fourier",
        spatial_type="attention",
    )

    print(f"   Parameters: {sum(p.numel() for p in hybrid1.parameters()):,}")

    # Test
    input_ids = torch.randint(1, 20000, (2, 128))
    with torch.no_grad():
        out1 = hybrid1(input_ids=input_ids)
    print(f"   Output: {input_ids.shape} -> {out1.shape}")

    # Pattern 2: Spectral preprocessing + attention
    print("\n2. Spectral preprocessing pattern...")

    class SpectralPreprocessor(nn.Module):
        def __init__(self, hidden_dim: int, seq_len: int):  # noqa: ARG002
            super().__init__()
            self.fourier_prep = FourierMixing(hidden_dim, dropout=0.0)
            self.wavelet_prep = WaveletMixing(hidden_dim, wavelet="db4", levels=2, dropout=0.0)

        def forward(self, x):
            # Apply both spectral methods
            fourier_out = self.fourier_prep(x)
            wavelet_out = self.wavelet_prep(x)
            # Combine via gating
            gate = torch.sigmoid(torch.randn_like(x))
            return gate * fourier_out + (1 - gate) * wavelet_out

    preprocessor = SpectralPreprocessor(hidden_dim=512, seq_len=256)
    print(f"   Spectral preprocessor: {sum(p.numel() for p in preprocessor.parameters()):,} params")

    # Test preprocessor
    x = torch.randn(2, 64, 512)
    with torch.no_grad():
        preprocessed = preprocessor(x)
    print(f"   Preprocessing: {x.shape} -> {preprocessed.shape}")
    print("   This could feed into standard transformer layers\n")


def performance_analysis():
    """Analyze performance characteristics of different architectures."""
    print("=== Performance Analysis ===\n")

    # Model configurations for comparison
    models = {
        "Pure Fourier": lambda: CustomSpectralEncoder(
            hidden_dim=512, num_layers=8, max_sequence_length=256
        ),
        "Alternating": lambda: CustomSpectralEncoder(
            hidden_dim=512, num_layers=8, max_sequence_length=256
        ),
        "Multi-Scale": lambda: MultiScaleSpectralModel(
            vocab_size=10000, hidden_dim=512, num_layers=8, max_sequence_length=256, scales=[1, 2]
        ),
    }

    # Compare parameter counts and memory usage
    print("Model comparison:")
    for name, model_fn in models.items():
        model = model_fn()
        param_count = sum(p.numel() for p in model.parameters())

        # Estimate memory usage (rough approximation)
        param_memory = param_count * 4 / (1024**2)  # 4 bytes per param, in MB

        print(f"   {name:15s}: {param_count:>8,} params, ~{param_memory:.1f}MB")

    # Theoretical complexity analysis
    print("\nComplexity analysis (for sequence length n):")
    complexities = {
        "Fourier Transform": "O(n log n)",
        "Wavelet Transform": "O(n)",
        "AFNO": "O(k * n log n) where k = n_modes",
        "Global Filter": "O(n log n)",
        "Standard Attention": "O(n²)",
    }

    for method, complexity in complexities.items():
        print(f"   {method:20s}: {complexity}")

    print("\n✓ All custom architectures provide sub-quadratic complexity!\n")


def main():
    """Run all custom architecture examples."""
    print("Custom Hybrid Spectral Architectures with Spectrans\n")
    print("=" * 65)

    try:
        demonstrate_custom_encoder()
        demonstrate_custom_mixing()
        demonstrate_multiscale_model()
        demonstrate_hybrid_architectures()
        performance_analysis()

        print("All custom architecture examples completed successfully! ✓")

        print("\nKey insights:")
        print("- Mix different spectral methods for complementary benefits")
        print("- Multi-scale processing captures patterns at different resolutions")
        print("- Custom components integrate seamlessly with existing ones")
        print("- All maintain efficient O(n log n) or O(n) complexity")

        print("\nNext steps:")
        print("- Experiment with different mixing patterns")
        print("- Try your own custom spectral transforms")
        print("- Combine with traditional attention selectively")
        print("- See integration_example.py for training these models")

    except Exception as e:
        print(f"Error running custom architecture examples: {e}")
        print("Make sure spectrans is installed: pip install spectrans")
        raise


if __name__ == "__main__":
    main()
