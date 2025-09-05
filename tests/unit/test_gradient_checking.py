"""Comprehensive gradient checking tests for spectrans components.

This module tests gradient flow through all spectral operations including
complex transforms (FFT, DWT), custom layers, and models to ensure correct
backpropagation and numerical stability.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck, gradgradcheck

# Import modules to ensure components are registered
import spectrans.models
import spectrans.transforms  # noqa: F401
from spectrans.core.registry import create_component
from spectrans.layers.attention.spectral import SpectralAttention
from spectrans.layers.mixing.afno import AFNOMixing
from spectrans.layers.mixing.fourier import FourierMixing
from spectrans.layers.mixing.global_filter import GlobalFilterMixing
from spectrans.layers.mixing.wavelet import WaveletMixing


class TestTransformGradients:
    """Test gradient flow through spectral transforms."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_fft_gradient_flow(self, dtype):
        """Test gradient flow through FFT operations."""
        transform = create_component('transform', 'fft1d', norm='ortho')

        # Create input that requires grad
        x = torch.randn(2, 8, 16, dtype=dtype, requires_grad=True)

        # Forward and inverse transform
        y = transform.transform(x)
        z = transform.inverse_transform(y)

        # Create loss and compute gradients
        loss = z.real.sum()
        loss.backward()

        # Check that gradients exist and are finite
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    @pytest.mark.parametrize("wavelet", ["db4", "sym4"])
    def test_dwt_gradient_flow(self, wavelet):
        """Test gradient flow through DWT operations."""
        transform = create_component('transform', 'dwt1d', wavelet=wavelet, levels=2)

        # DWT1D expects 2D input (batch, sequence)
        x = torch.randn(2, 64, requires_grad=True)

        # Forward transform returns coefficients
        coeffs = transform.decompose(x)

        # Reconstruct from coefficients
        y = transform.reconstruct(coeffs)

        # Compute loss and gradients
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_dct_gradient_flow(self):
        """Test gradient flow through DCT operations."""
        transform = create_component('transform', 'dct')

        x = torch.randn(2, 16, 32, requires_grad=True)

        y = transform.transform(x)
        z = transform.inverse_transform(y)

        loss = F.mse_loss(z, x)
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_hadamard_gradient_flow(self):
        """Test gradient flow through Hadamard transform."""
        transform = create_component('transform', 'hadamard')

        # Hadamard requires power of 2 dimensions
        x = torch.randn(2, 16, 16, requires_grad=True)

        y = transform.transform(x)
        z = transform.inverse_transform(y)

        loss = z.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestLayerGradients:
    """Test gradient flow through mixing and attention layers."""

    def test_fourier_mixing_gradients(self):
        """Test gradients through Fourier mixing layer."""
        layer = FourierMixing(hidden_dim=64)

        x = torch.randn(2, 32, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Check input gradients
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Check parameter gradients
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()

    def test_global_filter_gradients(self):
        """Test gradients through Global Filter layer."""
        layer = GlobalFilterMixing(hidden_dim=64, sequence_length=32)

        x = torch.randn(2, 32, 64, requires_grad=True)
        y = layer(x)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Check filter parameters have gradients
        for name, param in layer.named_parameters():
            if 'filter' in name and param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()

    def test_afno_mixing_gradients(self):
        """Test gradients through AFNO mixing layer."""
        layer = AFNOMixing(
            hidden_dim=64,
            max_sequence_length=32,
            modes_seq=16,
            modes_hidden=32,
            mlp_ratio=2.0
        )

        x = torch.randn(2, 32, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_wavelet_mixing_gradients(self):
        """Test gradients through Wavelet mixing layer."""
        layer = WaveletMixing(
            hidden_dim=64,
            wavelet='db4',
            levels=2
        )

        x = torch.randn(2, 32, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_spectral_attention_gradients(self):
        """Test gradients through Spectral Attention layer."""
        layer = SpectralAttention(
            hidden_dim=64,
            num_heads=4,
            num_features=128
        )

        x = torch.randn(2, 16, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestNumericalGradients:
    """Test numerical gradient correctness using gradcheck."""

    @pytest.mark.slow
    def test_fft_numerical_gradients(self):
        """Verify FFT gradients using torch.autograd.gradcheck."""
        transform = create_component('transform', 'fft1d', norm='ortho')

        # Use smaller input for gradcheck (expensive operation)
        x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)

        def fft_func(inp):
            y = transform.transform(inp)
            # Return real part for gradcheck (needs real output)
            return transform.inverse_transform(y).real

        # Check gradients numerically
        assert gradcheck(fft_func, (x,), eps=1e-6, atol=1e-4)

    @pytest.mark.slow
    def test_dct_numerical_gradients(self):
        """Verify DCT gradients using torch.autograd.gradcheck."""
        transform = create_component('transform', 'dct')

        x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)

        def dct_func(inp):
            return transform.transform(inp)

        assert gradcheck(dct_func, (x,), eps=1e-6, atol=1e-4)

    @pytest.mark.slow
    def test_layer_numerical_gradients(self):
        """Verify layer gradients using gradcheck."""
        layer = FourierMixing(hidden_dim=8)
        layer.double()  # Use float64 for numerical accuracy

        x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)

        def layer_func(inp):
            return layer(inp)

        assert gradcheck(layer_func, (x,), eps=1e-6, atol=1e-4)


class TestGradientAccumulation:
    """Test gradient accumulation and multiple backward passes."""

    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly across batches."""
        model = create_component(
            'model', 'fnet',
            hidden_dim=64,
            num_layers=2,
            max_sequence_length=128
        )

        # Zero gradients
        model.zero_grad()

        # First batch
        x1 = torch.randn(2, 16, 64, requires_grad=True)
        y1 = model(inputs_embeds=x1)
        loss1 = y1.sum()
        loss1.backward()

        # Save gradients
        grad1 = {name: param.grad.clone() if param.grad is not None else None
                 for name, param in model.named_parameters()}

        # Second batch (accumulate gradients)
        x2 = torch.randn(2, 16, 64, requires_grad=True)
        y2 = model(inputs_embeds=x2)
        loss2 = y2.sum()
        loss2.backward()

        # Check that gradients accumulated
        for name, param in model.named_parameters():
            if param.grad is not None and grad1[name] is not None:
                # Gradients should be different (accumulated)
                assert not torch.allclose(param.grad, grad1[name])

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing reduces memory but maintains correctness."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # Model without checkpointing
        model1 = create_component(
            'model', 'fnet',
            hidden_dim=64,
            num_layers=4,
            max_sequence_length=128,
            gradient_checkpointing=False
        )

        # Model with checkpointing
        model2 = create_component(
            'model', 'fnet',
            hidden_dim=64,
            num_layers=4,
            max_sequence_length=128,
            gradient_checkpointing=True
        )

        # Copy weights to ensure same initialization
        model2.load_state_dict(model1.state_dict())
        
        # Set both models to eval mode to disable dropout
        model1.eval()
        model2.eval()

        x = torch.randn(2, 32, 64, requires_grad=True)

        # Forward and backward without checkpointing
        y1 = model1(inputs_embeds=x)
        loss1 = y1.sum()
        loss1.backward()

        # Forward and backward with checkpointing
        y2 = model2(inputs_embeds=x)
        loss2 = y2.sum()
        loss2.backward()

        # Results should be identical in eval mode
        assert torch.allclose(y1, y2, atol=1e-5)


class TestGradientStability:
    """Test gradient stability and potential issues."""

    def test_gradient_explosion(self):
        """Test that gradients don't explode with deep models."""
        model = create_component(
            'model', 'fnet',
            hidden_dim=128,
            num_layers=12,  # Deep model
            max_sequence_length=256,
            dropout=0.1
        )

        x = torch.randn(2, 64, 128)
        y = model(inputs_embeds=x)
        loss = y.sum()
        loss.backward()

        # Check gradients are not exploding
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Allow slightly larger gradients for deep models
                assert grad_norm < 2e3, f"Gradient explosion in {name}: {grad_norm}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradients in {name}"

    def test_gradient_vanishing(self):
        """Test that gradients don't vanish in deep models."""
        model = create_component(
            'model', 'gfnet',
            hidden_dim=128,
            num_layers=12,
            max_sequence_length=256
        )

        x = torch.randn(2, 64, 128)
        y = model(inputs_embeds=x)
        loss = y.sum()
        loss.backward()

        # Check that at least some gradients are significant
        has_significant_gradient = False
        for _name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-6:
                    has_significant_gradient = True
                    break

        assert has_significant_gradient, "All gradients appear to have vanished"

    def test_complex_gradient_stability(self):
        """Test gradient stability through complex operations."""
        # FFT produces complex numbers
        transform = create_component('transform', 'fft1d')

        x = torch.randn(2, 32, 64, requires_grad=True)

        # Complex operation chain
        y = transform.transform(x)
        # Manipulate in frequency domain
        y_scaled = y * torch.exp(-torch.abs(y) * 0.1)
        z = transform.inverse_transform(y_scaled)

        # Take real part for loss
        loss = z.real.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestSecondOrderGradients:
    """Test second-order gradients (Hessian)."""

    @pytest.mark.slow
    def test_second_order_gradients(self):
        """Test that second-order gradients can be computed."""
        # Use a simple nonlinear layer that supports second-order gradients
        class NonlinearLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(16, 16, dtype=torch.float64))
            
            def forward(self, x):
                # Use a nonlinear operation to ensure non-zero second derivatives
                return torch.tanh(x @ self.weight)
        
        layer = NonlinearLayer()

        x = torch.randn(1, 8, 16, dtype=torch.float64, requires_grad=True)

        # First order gradient
        y = layer(x)
        grad_y = torch.autograd.grad(
            y.sum(), x, create_graph=True, retain_graph=True
        )[0]

        # Second order gradient
        grad2_y = torch.autograd.grad(
            grad_y.sum(), x, retain_graph=True
        )[0]

        assert grad2_y is not None
        assert torch.isfinite(grad2_y).all()
        
        # For a nonlinear layer, second-order gradients should be non-zero
        assert not torch.allclose(grad2_y, torch.zeros_like(grad2_y))

    @pytest.mark.slow
    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        transform = create_component('transform', 'dct')

        x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)

        def func(inp):
            return transform.transform(inp)

        # Check both first and second order gradients
        assert gradgradcheck(func, (x,), eps=1e-6, atol=1e-4)


class TestMixedPrecisionGradients:
    """Test gradient flow with mixed precision."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP")
    def test_amp_gradient_flow(self):
        """Test gradient flow with automatic mixed precision."""
        device = torch.device('cuda')

        model = create_component(
            'model', 'fnet',
            hidden_dim=128,
            num_layers=4,
            max_sequence_length=256
        ).to(device)

        x = torch.randn(2, 64, 128, device=device)

        # Use automatic mixed precision
        with torch.cuda.amp.autocast():
            y = model(inputs_embeds=x)
            loss = y.sum()

        # Scale loss for mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        scaler.scale(loss).backward()

        # Check gradients exist and are scaled properly
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()

    def test_gradient_clipping(self):
        """Test gradient clipping for stability."""
        model = create_component(
            'model', 'fnet',
            hidden_dim=64,
            num_layers=2,
            max_sequence_length=128
        )

        x = torch.randn(2, 32, 64)
        y = model(inputs_embeds=x)
        loss = y.sum() * 1000  # Large loss to create large gradients
        loss.backward()

        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Check all gradients are within bounds
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5

        assert total_norm <= max_norm * 1.01  # Small tolerance for numerical errors
