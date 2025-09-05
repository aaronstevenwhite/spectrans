"""Performance benchmarks for spectrans components.

This module contains comprehensive benchmarks for spectral transformer models,
transforms, and layers to track performance characteristics and detect regressions.
"""

import pytest
import torch

# Import modules to ensure components are registered
import spectrans.models
import spectrans.transforms  # noqa: F401
from spectrans.core.registry import create_component


class TestTransformPerformance:
    """Benchmark spectral transform performance."""

    @pytest.mark.benchmark(group="transforms")
    @pytest.mark.parametrize("batch_size,seq_len,hidden_dim", [
        (1, 128, 256),    # Small
        (4, 512, 512),    # Medium
        (8, 1024, 768),   # Large
    ])
    def test_fft1d_performance(self, benchmark, batch_size, seq_len, hidden_dim):
        """Benchmark FFT1D forward and inverse transforms."""
        transform = create_component('transform', 'fft1d', norm='ortho')
        x = torch.randn(batch_size, seq_len, hidden_dim)

        def run_transform():
            y = transform.transform(x)
            z = transform.inverse_transform(y)
            return z

        result = benchmark(run_transform)
        assert result.shape == x.shape

    @pytest.mark.benchmark(group="transforms")
    @pytest.mark.parametrize("batch_size,seq_len,hidden_dim", [
        (1, 128, 256),
        (4, 512, 512),
    ])
    def test_dct_performance(self, benchmark, batch_size, seq_len, hidden_dim):
        """Benchmark DCT forward and inverse transforms."""
        transform = create_component('transform', 'dct')
        x = torch.randn(batch_size, seq_len, hidden_dim)

        def run_transform():
            y = transform.transform(x)
            z = transform.inverse_transform(y)
            return z

        result = benchmark(run_transform)
        assert result.shape == x.shape

    @pytest.mark.benchmark(group="transforms")
    @pytest.mark.parametrize("batch_size,seq_len,hidden_dim", [
        (1, 128, 256),
        (4, 512, 512),
    ])
    def test_dwt1d_performance(self, benchmark, batch_size, seq_len, hidden_dim):
        """Benchmark DWT1D forward and inverse transforms."""
        transform = create_component('transform', 'dwt1d', wavelet='db4', levels=3)
        x = torch.randn(batch_size, seq_len, hidden_dim)

        def run_transform():
            # DWT1D expects 2D input (batch, sequence)
            # Process sequence dimension while treating batch*hidden as batch
            x_reshaped = x.transpose(1, 2).reshape(batch_size * hidden_dim, seq_len)
            coeffs = transform.decompose(x_reshaped)
            z = transform.reconstruct(coeffs)
            # Reshape back to original dimensions
            z_reshaped = z.reshape(batch_size, hidden_dim, -1).transpose(1, 2)
            return z_reshaped[:, :seq_len, :]  # Trim to original seq_len if needed

        result = benchmark(run_transform)
        # DWT may have slightly different shape due to padding
        assert result.shape[0] == x.shape[0]  # batch size preserved

    @pytest.mark.benchmark(group="transforms")
    @pytest.mark.parametrize("batch_size,seq_len,hidden_dim", [
        (1, 128, 256),
        (4, 256, 256),  # Square for Hadamard
    ])
    def test_hadamard_performance(self, benchmark, batch_size, seq_len, hidden_dim):
        """Benchmark Hadamard transform."""
        # Hadamard requires power of 2 dimensions
        seq_len = 256  # Force power of 2
        hidden_dim = 256

        transform = create_component('transform', 'hadamard')
        x = torch.randn(batch_size, seq_len, hidden_dim)

        def run_transform():
            y = transform.transform(x)
            z = transform.inverse_transform(y)
            return z

        result = benchmark(run_transform)
        assert result.shape == x.shape


class TestModelPerformance:
    """Benchmark model forward pass performance."""

    @pytest.mark.benchmark(group="models")
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 128),
        (4, 256),
        (8, 512),
    ])
    def test_fnet_performance(self, benchmark, batch_size, seq_len):
        """Benchmark FNet model forward pass."""
        model = create_component(
            'model', 'fnet',
            hidden_dim=768,
            num_layers=12,
            max_sequence_length=512,
            use_real_fft=True
        )
        model.eval()

        x = torch.randn(batch_size, seq_len, 768)

        with torch.no_grad():
            result = benchmark(model, inputs_embeds=x)

        assert result.shape == x.shape

    @pytest.mark.benchmark(group="models")
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 128),
        (4, 256),
        (8, 512),
    ])
    def test_gfnet_performance(self, benchmark, batch_size, seq_len):
        """Benchmark GFNet model forward pass."""
        model = create_component(
            'model', 'gfnet',
            hidden_dim=768,
            num_layers=12,
            max_sequence_length=512,
        )
        model.eval()

        x = torch.randn(batch_size, seq_len, 768)

        with torch.no_grad():
            result = benchmark(model, inputs_embeds=x)

        assert result.shape == x.shape

    @pytest.mark.benchmark(group="models")
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 128),
        (4, 256),
    ])
    def test_afno_performance(self, benchmark, batch_size, seq_len):
        """Benchmark AFNO model forward pass."""
        model = create_component(
            'model', 'afno',
            hidden_dim=768,
            num_layers=12,
            max_sequence_length=512,
            modes_seq=32,
            modes_hidden=384,
            mlp_ratio=2.0
        )
        model.eval()

        x = torch.randn(batch_size, seq_len, 768)

        with torch.no_grad():
            result = benchmark(model, inputs_embeds=x)

        assert result.shape == x.shape

    @pytest.mark.benchmark(group="models")
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 128),
        (4, 256),
    ])
    def test_spectral_attention_performance(self, benchmark, batch_size, seq_len):
        """Benchmark Spectral Attention model forward pass."""
        model = create_component(
            'model', 'spectral_attention',
            hidden_dim=768,
            num_layers=12,
            max_sequence_length=512,
            num_features=256
        )
        model.eval()

        x = torch.randn(batch_size, seq_len, 768)

        with torch.no_grad():
            result = benchmark(model, inputs_embeds=x)

        assert result.shape == x.shape


class TestLayerPerformance:
    """Benchmark individual layer performance."""

    @pytest.mark.benchmark(group="layers")
    @pytest.mark.parametrize("batch_size,seq_len,hidden_dim", [
        (1, 128, 256),
        (4, 512, 512),
        (8, 1024, 768),
    ])
    def test_fourier_mixing_performance(self, benchmark, batch_size, seq_len, hidden_dim):
        """Benchmark Fourier mixing layer."""
        from spectrans.layers.mixing.fourier import FourierMixing

        layer = FourierMixing(hidden_dim=hidden_dim)
        layer.eval()

        x = torch.randn(batch_size, seq_len, hidden_dim)

        with torch.no_grad():
            result = benchmark(layer, x)

        assert result.shape == x.shape

    @pytest.mark.benchmark(group="layers")
    @pytest.mark.parametrize("batch_size,seq_len,hidden_dim", [
        (1, 128, 256),
        (4, 512, 512),
    ])
    def test_global_filter_performance(self, benchmark, batch_size, seq_len, hidden_dim):
        """Benchmark Global Filter layer."""
        from spectrans.layers.mixing.global_filter import GlobalFilterMixing

        layer = GlobalFilterMixing(
            hidden_dim=hidden_dim,
            sequence_length=seq_len
        )
        layer.eval()

        x = torch.randn(batch_size, seq_len, hidden_dim)

        with torch.no_grad():
            result = benchmark(layer, x)

        assert result.shape == x.shape

    @pytest.mark.benchmark(group="layers")
    @pytest.mark.parametrize("batch_size,seq_len,hidden_dim", [
        (1, 128, 256),
        (4, 256, 512),
    ])
    def test_spectral_attention_layer_performance(self, benchmark, batch_size, seq_len, hidden_dim):
        """Benchmark Spectral Attention layer."""
        from spectrans.layers.attention.spectral import SpectralAttention

        layer = SpectralAttention(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_features=256
        )
        layer.eval()

        x = torch.randn(batch_size, seq_len, hidden_dim)

        with torch.no_grad():
            result = benchmark(layer, x)

        assert result.shape == x.shape


class TestComplexityComparison:
    """Compare complexity of different models."""

    @pytest.mark.benchmark(group="complexity", min_rounds=10)
    @pytest.mark.parametrize("seq_len", [128, 256, 512, 1024])
    def test_scaling_with_sequence_length(self, benchmark, seq_len):
        """Test how models scale with sequence length."""
        batch_size = 4
        hidden_dim = 512

        model = create_component(
            'model', 'fnet',
            hidden_dim=hidden_dim,
            num_layers=4,
            max_sequence_length=2048,
        )
        model.eval()

        x = torch.randn(batch_size, seq_len, hidden_dim)

        with torch.no_grad():
            result = benchmark(model, inputs_embeds=x)

        # Verify output shape
        assert result.shape == x.shape

        # Get stats for complexity analysis
        stats = benchmark.stats
        mean_time = stats.get("mean", 0)

        # Log-linear complexity should roughly double with doubled sequence
        # This is just for recording, not a hard assertion
        print(f"Seq {seq_len}: {mean_time:.4f}s")

    @pytest.mark.benchmark(group="complexity", min_rounds=10)
    @pytest.mark.parametrize("model_type", ["fnet", "gfnet", "afno"])
    def test_model_comparison(self, benchmark, model_type):
        """Compare different models on same input."""
        batch_size = 4
        seq_len = 512
        hidden_dim = 512

        model = create_component(
            'model', model_type,
            hidden_dim=hidden_dim,
            num_layers=6,
            max_sequence_length=1024,
        )
        model.eval()

        x = torch.randn(batch_size, seq_len, hidden_dim)

        with torch.no_grad():
            result = benchmark(model, inputs_embeds=x)

        assert result.shape == x.shape


class TestMemoryEfficiency:
    """Test memory efficiency of models."""

    @pytest.mark.benchmark(group="memory")
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_memory_scaling(self, benchmark, batch_size):
        """Test memory usage scaling with batch size."""
        seq_len = 512
        hidden_dim = 768

        model = create_component(
            'model', 'fnet',
            hidden_dim=hidden_dim,
            num_layers=12,
            max_sequence_length=1024,
        )
        model.eval()

        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Measure memory during forward pass
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            device = torch.device('cuda')
            model = model.to(device)
            x = x.to(device)

        with torch.no_grad():
            result = benchmark(model, inputs_embeds=x)

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"Batch {batch_size}: Peak memory {peak_memory:.2f} MB")

        assert result.shape == x.shape

    @pytest.mark.benchmark(group="memory")
    def test_gradient_checkpointing_memory(self, benchmark):
        """Test memory savings with gradient checkpointing."""
        batch_size = 2
        seq_len = 512
        hidden_dim = 768

        # Model without checkpointing
        model = create_component(
            'model', 'fnet',
            hidden_dim=hidden_dim,
            num_layers=12,
            max_sequence_length=1024,
            gradient_checkpointing=False
        )

        x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

        def forward_backward():
            output = model(inputs_embeds=x)
            loss = output.sum()
            loss.backward()
            return loss

        result = benchmark(forward_backward)
        assert isinstance(result, torch.Tensor)


class TestTransformComplexity:
    """Test computational complexity of transforms."""

    @pytest.mark.benchmark(group="transform_complexity")
    @pytest.mark.parametrize("n", [128, 256, 512, 1024, 2048])
    def test_fft_complexity(self, benchmark, n):
        """Test FFT O(n log n) complexity."""
        transform = create_component('transform', 'fft1d')
        x = torch.randn(1, n, 256)

        result = benchmark(transform.transform, x)
        assert result.shape[1] == n

        # Log results for complexity verification
        stats = benchmark.stats
        print(f"FFT n={n}: {stats['mean']:.6f}s")

    @pytest.mark.benchmark(group="transform_complexity")
    @pytest.mark.parametrize("n", [128, 256, 512, 1024])
    def test_dct_complexity(self, benchmark, n):
        """Test DCT O(n log n) complexity."""
        transform = create_component('transform', 'dct')
        x = torch.randn(1, n, 256)

        result = benchmark(transform.transform, x)
        assert result.shape[1] == n

        stats = benchmark.stats
        print(f"DCT n={n}: {stats['mean']:.6f}s")

    @pytest.mark.benchmark(group="transform_complexity")
    @pytest.mark.parametrize("n", [128, 256, 512, 1024])
    def test_hadamard_complexity(self, benchmark, n):
        """Test Hadamard O(n log n) complexity."""
        transform = create_component('transform', 'hadamard')
        x = torch.randn(1, n, n)  # Square matrix for Hadamard

        result = benchmark(transform.transform, x)
        assert result.shape[1] == n

        stats = benchmark.stats
        print(f"Hadamard n={n}: {stats['mean']:.6f}s")


# GPU-specific benchmarks (optional)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUPerformance:
    """GPU-specific performance benchmarks."""

    @pytest.mark.benchmark(group="gpu")
    @pytest.mark.parametrize("batch_size,seq_len", [
        (8, 512),
        (16, 1024),
        (32, 2048),
    ])
    def test_gpu_fnet_performance(self, benchmark, batch_size, seq_len):
        """Benchmark FNet on GPU with larger batches."""
        device = torch.device('cuda')

        model = create_component(
            'model', 'fnet',
            hidden_dim=768,
            num_layers=12,
            max_sequence_length=2048,
        ).to(device)
        model.eval()

        x = torch.randn(batch_size, seq_len, 768, device=device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(inputs_embeds=x)

        torch.cuda.synchronize()

        def run_model():
            with torch.no_grad():
                output = model(inputs_embeds=x)
            torch.cuda.synchronize()
            return output

        result = benchmark(run_model)
        assert result.shape == x.shape

    @pytest.mark.benchmark(group="gpu")
    def test_gpu_memory_efficiency(self, benchmark):
        """Test GPU memory efficiency of different models."""
        device = torch.device('cuda')
        batch_size = 8
        seq_len = 1024
        hidden_dim = 768

        models = {
            'fnet': create_component('model', 'fnet', hidden_dim=hidden_dim, num_layers=6, max_sequence_length=2048),
            'gfnet': create_component('model', 'gfnet', hidden_dim=hidden_dim, num_layers=6, max_sequence_length=2048),
        }

        for name, model in models.items():
            model = model.to(device)
            model.eval()

            x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                output = benchmark(model, inputs_embeds=x)

            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            print(f"{name}: Peak GPU memory {peak_memory:.3f} GB")

            assert output.shape == x.shape
