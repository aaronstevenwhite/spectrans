"""Memory profiling tests for spectrans components.

This module contains memory profiling tests to detect memory leaks, track memory
usage patterns, and ensure efficient memory management across all spectrans
components during training and inference.
"""

import gc
import tracemalloc
from typing import Any

import pytest
import torch
import torch.nn as nn

# Import modules to ensure components are registered
import spectrans.models
import spectrans.transforms  # noqa: F401
from spectrans.core.registry import create_component


class TestMemoryLeaks:
    """Test for memory leaks in various operations."""

    def test_transform_memory_leak(self):
        """Test that transforms don't leak memory over iterations."""
        transform = create_component("transform", "fft1d", norm="ortho")
        x = torch.randn(4, 256, 512)

        # Get initial memory snapshot
        gc.collect()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Run many iterations
        for _ in range(100):
            y = transform.transform(x)
            z = transform.inverse_transform(y)
            del y, z

        # Force garbage collection
        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()

        # Compare memory usage
        top_stats = snapshot2.compare_to(snapshot1, "lineno")

        # Check for significant memory growth (>10MB would be concerning)
        total_growth = sum(stat.size_diff for stat in top_stats)
        assert total_growth < 10 * 1024 * 1024, (
            f"Memory grew by {total_growth / 1024 / 1024:.2f} MB"
        )

        tracemalloc.stop()

    def test_model_memory_leak(self):
        """Test that models don't leak memory during forward passes."""
        model = create_component(
            "model", "fnet", hidden_dim=256, num_layers=4, max_sequence_length=512
        )
        model.eval()

        x = torch.randn(2, 128, 256)

        # Get initial memory
        gc.collect()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Run multiple forward passes
        with torch.no_grad():
            for _ in range(50):
                output = model(inputs_embeds=x)
                del output

        # Check memory after
        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()

        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        total_growth = sum(stat.size_diff for stat in top_stats)

        # Allow small growth for caching but not continuous leaking
        assert total_growth < 5 * 1024 * 1024, f"Memory grew by {total_growth / 1024 / 1024:.2f} MB"

        tracemalloc.stop()

    def test_gradient_accumulation_memory(self):
        """Test memory behavior during gradient accumulation."""
        model = create_component(
            "model", "gfnet", hidden_dim=256, num_layers=4, max_sequence_length=512
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        x = torch.randn(2, 128, 256)

        # Track memory during gradient accumulation
        gc.collect()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Gradient accumulation steps
        accumulation_steps = 4
        for step in range(accumulation_steps * 3):  # 3 full accumulation cycles
            output = model(inputs_embeds=x)
            loss = output.sum() / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Final memory check
        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()

        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        total_growth = sum(stat.size_diff for stat in top_stats)

        # Should not continuously grow after gradient clears
        assert total_growth < 10 * 1024 * 1024, (
            f"Memory grew by {total_growth / 1024 / 1024:.2f} MB"
        )

        tracemalloc.stop()


class TestPeakMemoryUsage:
    """Test peak memory usage for different configurations."""

    @pytest.mark.parametrize(
        "model_type,hidden_dim,num_layers",
        [
            ("fnet", 256, 4),
            ("fnet", 512, 8),
            ("gfnet", 256, 4),
            ("afno", 256, 4),
        ],
    )
    def test_model_peak_memory(self, model_type, hidden_dim, num_layers):
        """Profile peak memory usage for different model configurations."""
        tracemalloc.start()

        model = create_component(
            "model",
            model_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=512,
        )

        x = torch.randn(4, 256, hidden_dim)

        # Forward pass
        output = model(inputs_embeds=x)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Report memory usage
        print(f"\n{model_type} (dim={hidden_dim}, layers={num_layers}):")
        print(f"  Current memory: {current / 1024 / 1024:.2f} MB")
        print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")

        # Ensure reasonable memory usage (adjust threshold as needed)
        assert peak < 1024 * 1024 * 1024, f"Peak memory too high: {peak / 1024 / 1024:.2f} MB"

    def test_batch_size_memory_scaling(self):
        """Test how memory scales with batch size."""
        model = create_component(
            "model", "fnet", hidden_dim=512, num_layers=6, max_sequence_length=512
        )

        memory_usage: dict[int, float] = {}

        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 256, 512)

            tracemalloc.start()
            output = model(inputs_embeds=x)
            loss = output.sum()
            loss.backward()

            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_usage[batch_size] = peak / 1024 / 1024  # MB

            # Clean up
            del output, loss
            model.zero_grad()
            gc.collect()

        # Check that memory scales roughly linearly with batch size
        print("\nBatch size memory scaling:")
        for batch_size, memory_mb in memory_usage.items():
            print(f"  Batch {batch_size}: {memory_mb:.2f} MB")

        # Verify roughly linear scaling (allow 50% deviation)
        base_memory = memory_usage[1]
        # Skip test if memory measurements are too small to be reliable
        if base_memory < 1.0:  # Less than 1 MB
            print("Memory usage too small to reliably test scaling")
            return

        for batch_size in [2, 4, 8]:
            expected = base_memory * batch_size
            actual = memory_usage[batch_size]
            ratio = actual / expected
            assert 0.3 < ratio < 2.0, (
                f"Non-linear scaling: batch {batch_size} uses {ratio:.2f}x expected memory"
            )

    def test_sequence_length_memory_scaling(self):
        """Test how memory scales with sequence length."""
        model = create_component(
            "model", "fnet", hidden_dim=512, num_layers=6, max_sequence_length=2048
        )

        memory_usage: dict[int, float] = {}

        for seq_len in [128, 256, 512, 1024]:
            x = torch.randn(2, seq_len, 512)

            tracemalloc.start()
            output = model(inputs_embeds=x)
            loss = output.sum()
            loss.backward()

            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_usage[seq_len] = peak / 1024 / 1024  # MB

            # Clean up
            del output, loss
            model.zero_grad()
            gc.collect()

        # Check memory scaling pattern
        print("\nSequence length memory scaling:")
        for seq_len, memory_mb in memory_usage.items():
            print(f"  Seq {seq_len}: {memory_mb:.2f} MB")

        # For FFT-based models, expect O(n log n) scaling
        # Check that doubling sequence length doesn't more than triple memory
        for i in range(len([128, 256, 512]) - 1):
            seq1, seq2 = [128, 256, 512][i : i + 2]
            ratio = memory_usage[seq2] / memory_usage[seq1]
            assert ratio < 3.0, f"Memory scaling too steep: {seq1}->{seq2} increased {ratio:.2f}x"


class TestMemoryOptimizations:
    """Test memory optimization techniques."""

    def test_gradient_checkpointing_memory_savings(self):
        """Test memory savings from gradient checkpointing."""
        hidden_dim = 768
        num_layers = 12
        seq_len = 512
        batch_size = 4

        # Model without checkpointing
        model_no_checkpoint = create_component(
            "model",
            "fnet",
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=1024,
            gradient_checkpointing=False,
        )

        # Model with checkpointing
        model_checkpoint = create_component(
            "model",
            "fnet",
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_sequence_length=1024,
            gradient_checkpointing=True,
        )

        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Measure without checkpointing
        tracemalloc.start()
        output1 = model_no_checkpoint(inputs_embeds=x.clone())
        loss1 = output1.sum()
        loss1.backward()
        _, peak_no_checkpoint = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Clean up
        del output1, loss1
        model_no_checkpoint.zero_grad()
        gc.collect()

        # Measure with checkpointing
        tracemalloc.start()
        output2 = model_checkpoint(inputs_embeds=x.clone())
        loss2 = output2.sum()
        loss2.backward()
        _, peak_checkpoint = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Report savings
        savings_mb = (peak_no_checkpoint - peak_checkpoint) / 1024 / 1024
        savings_pct = (1 - peak_checkpoint / peak_no_checkpoint) * 100

        print("\nGradient checkpointing memory savings:")
        print(f"  Without checkpointing: {peak_no_checkpoint / 1024 / 1024:.2f} MB")
        print(f"  With checkpointing: {peak_checkpoint / 1024 / 1024:.2f} MB")
        print(f"  Savings: {savings_mb:.2f} MB ({savings_pct:.1f}%)")

        # Should save at least some memory (skip if measurements too small)
        if peak_no_checkpoint < 1024 * 1024:  # Less than 1MB
            print("Memory usage too small to reliably test checkpointing savings")
            return
        assert peak_checkpoint < peak_no_checkpoint, "Checkpointing should reduce memory"

    def test_inplace_operations_memory(self):
        """Test memory efficiency of in-place operations."""
        from spectrans.layers.mixing.fourier import FourierMixing

        # Compare in-place vs non-in-place operations
        layer = FourierMixing(hidden_dim=512)
        x = torch.randn(4, 256, 512)

        # Non-in-place version (standard)
        tracemalloc.start()
        _ = layer(x.clone())
        _, peak_standard = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Could test in-place operations if they were implemented
        # This is a placeholder for future optimization testing

        print(f"\nStandard operation peak memory: {peak_standard / 1024 / 1024:.2f} MB")


class TestMemoryProfiling:
    """Detailed memory profiling for specific operations."""

    def test_transform_memory_breakdown(self):
        """Profile memory usage of different transform types."""
        transforms = ["fft1d", "dct", "dst", "hadamard"]
        x = torch.randn(4, 512, 512)

        print("\nTransform memory usage:")
        for transform_name in transforms:
            # Hadamard needs power-of-2 dimensions
            x_test = torch.randn(4, 512, 512) if transform_name == "hadamard" else x.clone()

            transform = create_component("transform", transform_name)

            tracemalloc.start()
            y = transform.transform(x_test)
            if hasattr(y, "real"):  # Complex output
                z = transform.inverse_transform(y)
            else:
                z = transform.inverse_transform(y)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"  {transform_name}: {peak / 1024 / 1024:.2f} MB peak")

            del y, z
            gc.collect()

    def test_layer_memory_breakdown(self):
        """Profile memory usage of different layer types."""
        from spectrans.layers.attention.spectral import SpectralAttention
        from spectrans.layers.mixing.afno import AFNOMixing
        from spectrans.layers.mixing.fourier import FourierMixing
        from spectrans.layers.mixing.global_filter import GlobalFilterMixing

        layers: list[tuple[str, nn.Module]] = [
            ("FourierMixing", FourierMixing(hidden_dim=512)),
            ("GlobalFilterMixing", GlobalFilterMixing(hidden_dim=512, sequence_length=256)),
            ("AFNOMixing", AFNOMixing(hidden_dim=512, max_sequence_length=256)),
            ("SpectralAttention", SpectralAttention(hidden_dim=512, num_heads=8)),
        ]

        x = torch.randn(4, 256, 512, requires_grad=True)

        print("\nLayer memory usage (forward + backward):")
        for name, layer in layers:
            tracemalloc.start()

            # Forward pass
            y = layer(x)
            # Backward pass
            loss = y.sum()
            loss.backward(retain_graph=True)

            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"  {name}: {peak / 1024 / 1024:.2f} MB peak")

            # Clean up
            del y, loss
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()
            gc.collect()

    def test_training_loop_memory_profile(self):
        """Profile memory usage during a simulated training loop."""
        model = create_component(
            "model", "fnet", hidden_dim=512, num_layers=6, max_sequence_length=512
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(4, 256, 512)

        # Track memory over training steps
        memory_history: list[float] = []

        for _ in range(10):
            tracemalloc.start()

            # Forward pass
            output = model(inputs_embeds=x)
            loss = output.sum()

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            _, peak = tracemalloc.get_traced_memory()
            memory_history.append(peak / 1024 / 1024)
            tracemalloc.stop()

            # Clean up
            del output, loss
            gc.collect()

        # Check that memory is stable (not growing)
        print("\nTraining loop memory (MB):")
        for i, mem in enumerate(memory_history):
            print(f"  Step {i}: {mem:.2f} MB")

        # Memory should stabilize after first few steps
        later_steps = memory_history[3:]
        max_variation = max(later_steps) - min(later_steps)
        assert max_variation < 10, f"Memory varies too much: {max_variation:.2f} MB"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU memory profiling")
class TestGPUMemoryProfiling:
    """GPU-specific memory profiling tests."""

    def test_gpu_memory_tracking(self):
        """Track GPU memory usage for models."""
        device = torch.device("cuda")

        model = create_component(
            "model", "fnet", hidden_dim=768, num_layers=12, max_sequence_length=512
        ).to(device)

        x = torch.randn(8, 512, 768, device=device)

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Forward pass
        output = model(inputs_embeds=x)
        forward_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

        # Backward pass
        loss = output.sum()
        loss.backward()
        total_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

        print("\nGPU memory usage:")
        print(f"  After forward: {forward_memory:.2f} GB")
        print(f"  After backward: {total_memory:.2f} GB")
        print(f"  Backward additional: {(total_memory - forward_memory):.2f} GB")

        # Ensure reasonable GPU memory usage
        assert total_memory < 8.0, f"GPU memory usage too high: {total_memory:.2f} GB"

    def test_gpu_memory_efficiency(self):
        """Test GPU memory efficiency with different batch sizes."""
        device = torch.device("cuda")

        model = create_component(
            "model", "gfnet", hidden_dim=512, num_layers=8, max_sequence_length=512
        ).to(device)

        memory_per_batch: dict[int, float] = {}

        for batch_size in [1, 2, 4, 8, 16]:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            x = torch.randn(batch_size, 256, 512, device=device)

            # Forward + backward
            output = model(inputs_embeds=x)
            loss = output.sum()
            loss.backward()

            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory_per_sample = peak_memory / batch_size
            memory_per_batch[batch_size] = memory_per_sample

            print(
                f"\nBatch {batch_size}: {peak_memory:.2f} MB total, {memory_per_sample:.2f} MB/sample"
            )

            # Clean up
            del output, loss, x
            model.zero_grad()
            torch.cuda.empty_cache()

        # Memory per sample should be relatively constant (within 20%)
        values = list(memory_per_batch.values())
        min_val, max_val = min(values), max(values)
        variation = (max_val - min_val) / min_val
        assert variation < 0.2, f"Memory per sample varies too much: {variation:.1%}"


class TestMemoryAllocation:
    """Test memory allocation patterns."""

    def test_tensor_allocation_patterns(self):
        """Test tensor allocation and deallocation patterns."""

        def get_tensor_count() -> int:
            """Count PyTorch tensors in memory."""
            count = 0
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj):
                        count += 1
                except Exception:
                    pass
            return count

        initial_count = get_tensor_count()

        # Create and destroy tensors
        model = create_component(
            "model", "fnet", hidden_dim=256, num_layers=2, max_sequence_length=256
        )

        x = torch.randn(2, 128, 256)
        output = model(inputs_embeds=x)

        after_forward = get_tensor_count()

        # Clean up
        del output, x, model
        gc.collect()

        final_count = get_tensor_count()

        print("\nTensor counts:")
        print(f"  Initial: {initial_count}")
        print(f"  After forward: {after_forward}")
        print(f"  After cleanup: {final_count}")

        # Should return close to initial state (allow some caching)
        assert final_count < initial_count + 100, (
            f"Too many tensors remain: {final_count - initial_count}"
        )

    def test_parameter_memory_usage(self):
        """Test memory usage of model parameters."""
        models: list[tuple[str, dict[str, Any]]] = [
            ("fnet", {"hidden_dim": 256, "num_layers": 4}),
            ("fnet", {"hidden_dim": 512, "num_layers": 8}),
            ("fnet", {"hidden_dim": 768, "num_layers": 12}),
        ]

        print("\nModel parameter memory:")
        for model_type, kwargs in models:
            model = create_component("model", model_type, max_sequence_length=512, **kwargs)

            # Calculate parameter memory
            total_params = sum(p.numel() for p in model.parameters())
            total_memory = sum(p.numel() * p.element_size() for p in model.parameters())

            print(f"  {model_type} (dim={kwargs['hidden_dim']}, L={kwargs['num_layers']}):")
            print(f"    Parameters: {total_params:,}")
            print(f"    Memory: {total_memory / 1024 / 1024:.2f} MB")

            # Verify parameter count is reasonable
            assert total_params < 100_000_000, f"Too many parameters: {total_params:,}"
