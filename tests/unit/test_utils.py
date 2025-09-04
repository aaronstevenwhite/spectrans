"""Unit tests for utility functions."""

import math

import pytest
import torch
import torch.nn as nn

from spectrans.utils import (
    circular_pad,
    complex_conjugate,
    complex_divide,
    complex_dropout,
    complex_exp,
    complex_kaiming_init,
    complex_log,
    complex_modulus,
    complex_multiply,
    complex_normal_init,
    complex_phase,
    complex_polar,
    complex_relu,
    complex_xavier_init,
    dct_init,
    frequency_init,
    hadamard_init,
    init_conv_spectral,
    init_linear_spectral,
    kaiming_spectral_init,
    make_complex,
    orthogonal_spectral_init,
    pad_for_convolution,
    pad_for_fft,
    pad_sequence,
    pad_to_length,
    pad_to_power_of_2,
    reflect_pad,
    spectral_init,
    split_complex,
    symmetric_pad,
    unpad_sequence,
    unpad_to_length,
    wavelet_init,
    xavier_spectral_init,
    zero_pad,
)


class TestComplexOperations:
    """Test complex tensor operations."""

    def test_complex_multiply_basic(self, device):
        """Test basic complex multiplication."""
        a = torch.complex(torch.tensor([1.0, 2.0], device=device),
                         torch.tensor([1.0, 0.0], device=device))
        b = torch.complex(torch.tensor([2.0, 1.0], device=device),
                         torch.tensor([1.0, -1.0], device=device))

        result = complex_multiply(a, b)

        # (1+i)*(2+i) = 2 + i + 2i - 1 = 1 + 3i
        # (2+0i)*(1-i) = 2 - 2i
        expected = torch.complex(torch.tensor([1.0, 2.0], device=device),
                                torch.tensor([3.0, -2.0], device=device))

        torch.testing.assert_close(result, expected)

    def test_complex_multiply_broadcasting(self, device):
        """Test complex multiplication with broadcasting."""
        a = torch.complex(torch.tensor([[1.0]], device=device),
                         torch.tensor([[1.0]], device=device))
        b = torch.complex(torch.tensor([1.0, 2.0], device=device),
                         torch.tensor([0.0, 1.0], device=device))

        result = complex_multiply(a, b)

        # (1+i) * [1, 2+i] = [1+i, 1+3i]
        expected = torch.complex(torch.tensor([[1.0, 1.0]], device=device),
                                torch.tensor([[1.0, 3.0]], device=device))

        torch.testing.assert_close(result, expected)

    def test_complex_multiply_type_error(self, device):
        """Test complex multiplication type checking."""
        real_tensor = torch.tensor([1.0, 2.0], device=device)
        complex_tensor = torch.complex(real_tensor, real_tensor)

        with pytest.raises(TypeError, match="First argument must be complex"):
            complex_multiply(real_tensor, complex_tensor)

        with pytest.raises(TypeError, match="Second argument must be complex"):
            complex_multiply(complex_tensor, real_tensor)

    def test_complex_conjugate(self, device):
        """Test complex conjugate."""
        x = torch.complex(torch.tensor([1.0, 2.0], device=device),
                         torch.tensor([3.0, -1.0], device=device))

        result = complex_conjugate(x)
        expected = torch.complex(torch.tensor([1.0, 2.0], device=device),
                                torch.tensor([-3.0, 1.0], device=device))

        torch.testing.assert_close(result, expected)

    def test_complex_conjugate_type_error(self, device):
        """Test complex conjugate type checking."""
        real_tensor = torch.tensor([1.0, 2.0], device=device)

        with pytest.raises(TypeError, match="Input must be complex"):
            complex_conjugate(real_tensor)

    def test_complex_modulus(self, device):
        """Test complex modulus/magnitude."""
        x = torch.complex(torch.tensor([3.0, 4.0], device=device),
                         torch.tensor([4.0, 3.0], device=device))

        result = complex_modulus(x)
        expected = torch.tensor([5.0, 5.0], device=device)  # sqrt(3²+4²) = 5

        torch.testing.assert_close(result, expected)

    def test_complex_phase(self, device):
        """Test complex phase angle."""
        x = torch.complex(torch.tensor([1.0, 0.0, -1.0], device=device),
                         torch.tensor([0.0, 1.0, 0.0], device=device))

        result = complex_phase(x)
        expected = torch.tensor([0.0, math.pi/2, math.pi], device=device)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_complex_polar(self, device):
        """Test complex construction from polar coordinates."""
        magnitude = torch.tensor([1.0, 2.0], device=device)
        phase = torch.tensor([0.0, math.pi/2], device=device)

        result = complex_polar(magnitude, phase)

        # r=1, θ=0 → 1+0i
        # r=2, θ=π/2 → 0+2i
        expected = torch.complex(torch.tensor([1.0, 0.0], device=device),
                                torch.tensor([0.0, 2.0], device=device))

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_complex_polar_negative_magnitude_error(self, device):
        """Test complex polar with negative magnitude."""
        magnitude = torch.tensor([-1.0], device=device)
        phase = torch.tensor([0.0], device=device)

        with pytest.raises(ValueError, match="Magnitude must be non-negative"):
            complex_polar(magnitude, phase)

    def test_complex_polar_type_errors(self, device):
        """Test complex polar type checking."""
        real_tensor = torch.tensor([1.0], device=device)
        complex_tensor = torch.complex(real_tensor, real_tensor)

        with pytest.raises(TypeError, match="Magnitude must be real"):
            complex_polar(complex_tensor, real_tensor)

        with pytest.raises(TypeError, match="Phase must be real"):
            complex_polar(real_tensor, complex_tensor)

    def test_complex_exp(self, device):
        """Test complex exponential."""
        x = torch.complex(torch.tensor([0.0, 1.0], device=device),
                         torch.tensor([math.pi, 0.0], device=device))

        result = complex_exp(x)

        # e^(iπ) = -1, e^1 ≈ 2.718
        expected_real = torch.tensor([-1.0, math.e], device=device)
        expected_imag = torch.tensor([0.0, 0.0], device=device)
        expected = torch.complex(expected_real, expected_imag)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_complex_log(self, device):
        """Test complex logarithm."""
        x = torch.complex(torch.tensor([1.0, math.e], device=device),
                         torch.tensor([0.0, 0.0], device=device))

        result = complex_log(x)
        expected = torch.complex(torch.tensor([0.0, 1.0], device=device),
                                torch.tensor([0.0, 0.0], device=device))

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_complex_log_zero_error(self, device):
        """Test complex log with zero input."""
        x = torch.complex(torch.tensor([0.0], device=device),
                         torch.tensor([0.0], device=device))

        with pytest.raises(ValueError, match="Logarithm undefined for zero"):
            complex_log(x)

    def test_complex_divide(self, device):
        """Test complex division."""
        a = torch.complex(torch.tensor([4.0, 6.0], device=device),
                         torch.tensor([2.0, 0.0], device=device))
        b = torch.complex(torch.tensor([2.0, 3.0], device=device),
                         torch.tensor([1.0, 0.0], device=device))

        result = complex_divide(a, b)

        # (4+2i)/(2+i) = (4+2i)(2-i)/(5) = (10+0i)/5 = 2
        # (6+0i)/(3+0i) = 2
        expected = torch.complex(torch.tensor([2.0, 2.0], device=device),
                                torch.tensor([0.0, 0.0], device=device))

        torch.testing.assert_close(result, expected)

    def test_complex_divide_zero_error(self, device):
        """Test complex division by zero."""
        a = torch.complex(torch.tensor([1.0], device=device),
                         torch.tensor([1.0], device=device))
        b = torch.complex(torch.tensor([0.0], device=device),
                         torch.tensor([0.0], device=device))

        with pytest.raises(ValueError, match="Division by zero"):
            complex_divide(a, b)

    def test_make_complex(self, device):
        """Test complex construction from parts."""
        real = torch.tensor([1.0, 2.0], device=device)
        imag = torch.tensor([3.0, 4.0], device=device)

        result = make_complex(real, imag)
        expected = torch.complex(real, imag)

        torch.testing.assert_close(result, expected)

    def test_split_complex(self, device):
        """Test splitting complex into parts."""
        x = torch.complex(torch.tensor([1.0, 2.0], device=device),
                         torch.tensor([3.0, 4.0], device=device))

        real_part, imag_part = split_complex(x)

        torch.testing.assert_close(real_part, torch.tensor([1.0, 2.0], device=device))
        torch.testing.assert_close(imag_part, torch.tensor([3.0, 4.0], device=device))

    def test_complex_relu(self, device):
        """Test complex ReLU activation."""
        x = torch.complex(torch.tensor([1.0, -1.0], device=device),
                         torch.tensor([-1.0, 2.0], device=device))

        result = complex_relu(x)
        expected = torch.complex(torch.tensor([1.0, 0.0], device=device),
                                torch.tensor([0.0, 2.0], device=device))

        torch.testing.assert_close(result, expected)

    def test_complex_dropout_training(self, device):
        """Test complex dropout in training mode."""
        torch.manual_seed(42)
        x = torch.complex(torch.ones(100, device=device),
                         torch.ones(100, device=device))

        result = complex_dropout(x, p=0.5, training=True)

        # Should have some zeros and some scaled values
        assert torch.sum(torch.abs(result) == 0) > 0  # Some dropped
        assert torch.sum(torch.abs(result) > 1) > 0   # Some scaled up

    def test_complex_dropout_inference(self, device):
        """Test complex dropout in inference mode."""
        x = torch.complex(torch.ones(10, device=device),
                         torch.ones(10, device=device))

        result = complex_dropout(x, p=0.5, training=False)

        # Should be unchanged in inference mode
        torch.testing.assert_close(result, x)

    def test_complex_dropout_invalid_p(self, device):
        """Test complex dropout with invalid probability."""
        x = torch.complex(torch.ones(10, device=device),
                         torch.ones(10, device=device))

        with pytest.raises(ValueError, match="Dropout probability must be in"):
            complex_dropout(x, p=-0.1)

        with pytest.raises(ValueError, match="Dropout probability must be in"):
            complex_dropout(x, p=1.1)


class TestPaddingOperations:
    """Test padding operations."""

    def test_pad_to_length_zero(self, device):
        """Test zero padding to specific length."""
        x = torch.tensor([[1, 2, 3]], device=device)
        result = pad_to_length(x, 5, dim=-1, mode="zero")
        expected = torch.tensor([[1, 2, 3, 0, 0]], device=device)

        torch.testing.assert_close(result, expected)

    def test_pad_to_length_no_change(self, device):
        """Test padding when no change needed."""
        x = torch.tensor([1, 2, 3], device=device)
        result = pad_to_length(x, 3, dim=-1, mode="zero")

        torch.testing.assert_close(result, x)

    def test_pad_to_length_invalid_target(self, device):
        """Test padding with invalid target length."""
        x = torch.tensor([1, 2, 3], device=device)

        with pytest.raises(ValueError, match="Target length .* must be >= current length"):
            pad_to_length(x, 2, dim=-1)

    def test_pad_to_length_invalid_mode(self, device):
        """Test padding with invalid mode."""
        x = torch.tensor([1, 2, 3], device=device)

        with pytest.raises(ValueError, match="Invalid padding mode"):
            pad_to_length(x, 5, dim=-1, mode="invalid")

    def test_unpad_to_length(self, device):
        """Test unpadding to original length."""
        x = torch.tensor([[1, 2, 3, 0, 0]], device=device)
        result = unpad_to_length(x, 3, dim=-1)
        expected = torch.tensor([[1, 2, 3]], device=device)

        torch.testing.assert_close(result, expected)

    def test_unpad_to_length_invalid_target(self, device):
        """Test unpadding with invalid target."""
        x = torch.tensor([1, 2, 3], device=device)

        with pytest.raises(ValueError, match="Target length .* must be <= current length"):
            unpad_to_length(x, 5)

    def test_pad_sequence(self, device):
        """Test padding sequence list."""
        seq1 = torch.tensor([[1, 2]], device=device)
        seq2 = torch.tensor([[3, 4, 5]], device=device)
        seq3 = torch.tensor([[6]], device=device)

        result = pad_sequence([seq1, seq2, seq3], padding_value=-1)
        expected = torch.tensor([
            [[1, 2, -1]],
            [[3, 4, 5]],
            [[6, -1, -1]]
        ], device=device)

        torch.testing.assert_close(result, expected)

    def test_pad_sequence_empty_list(self):
        """Test padding empty sequence list."""
        with pytest.raises(ValueError, match="Cannot pad empty list"):
            pad_sequence([])

    def test_pad_sequence_incompatible_shapes(self, device):
        """Test padding with incompatible shapes."""
        seq1 = torch.tensor([[1, 2]], device=device)  # Shape: (1, 2)
        seq2 = torch.tensor([[3, 4], [5, 6]], device=device)  # Shape: (2, 2)

        with pytest.raises(ValueError, match="same shape except in padding dimension"):
            pad_sequence([seq1, seq2])

    def test_unpad_sequence(self, device):
        """Test unpadding sequence."""
        padded = torch.tensor([
            [[1, 2, 0]],
            [[3, 4, 5]],
            [[6, 0, 0]]
        ], device=device)

        result = unpad_sequence(padded, [2, 3, 1])

        expected = [
            torch.tensor([[1, 2]], device=device),
            torch.tensor([[3, 4, 5]], device=device),
            torch.tensor([[6]], device=device)
        ]

        for r, e in zip(result, expected, strict=False):
            torch.testing.assert_close(r, e)

    def test_circular_pad(self, device):
        """Test circular padding."""
        x = torch.tensor([1, 2, 3, 4], device=device)
        result = circular_pad(x, 2)
        expected = torch.tensor([1, 2, 3, 4, 3, 4], device=device)

        torch.testing.assert_close(result, expected)

    def test_circular_pad_exceeds_size(self, device):
        """Test circular padding exceeds tensor size."""
        x = torch.tensor([1, 2], device=device)

        with pytest.raises(ValueError, match="exceeds tensor size"):
            circular_pad(x, 3)

    def test_reflect_pad(self, device):
        """Test reflection padding."""
        x = torch.tensor([1, 2, 3, 4], device=device)
        result = reflect_pad(x, 2)
        # Reflect padding: mirrors without repeating the edge
        # For [1, 2, 3, 4] with pad=2, we take elements [2,3] (excluding edge) and flip to get [3, 2]
        expected = torch.tensor([1, 2, 3, 4, 3, 2], device=device)

        torch.testing.assert_close(result, expected)

    def test_reflect_pad_too_large(self, device):
        """Test reflection padding too large."""
        x = torch.tensor([1, 2], device=device)

        with pytest.raises(ValueError, match="must be < tensor size"):
            reflect_pad(x, 2)

    def test_symmetric_pad(self, device):
        """Test symmetric padding."""
        x = torch.tensor([1, 2, 3, 4], device=device)
        result = symmetric_pad(x, 2)
        # Symmetric padding: mirrors including the edge
        # For [1, 2, 3, 4] with pad=2, we take last 2 elements [3, 4] and flip to get [4, 3]
        expected = torch.tensor([1, 2, 3, 4, 4, 3], device=device)

        torch.testing.assert_close(result, expected)

    def test_zero_pad_custom_value(self, device):
        """Test zero padding with custom value."""
        x = torch.tensor([1.0, 2.0], device=device)
        result = zero_pad(x, 2, value=5.0)
        expected = torch.tensor([1.0, 2.0, 5.0, 5.0], device=device)

        torch.testing.assert_close(result, expected)

    def test_pad_for_fft(self, device):
        """Test padding for FFT optimization."""
        x = torch.randn(10, device=device)
        padded, original_length = pad_for_fft(x)

        assert original_length == 10
        assert padded.shape[-1] == 16  # Next power of 2
        torch.testing.assert_close(padded[:10], x)

    def test_pad_to_power_of_2(self):
        """Test finding next power of 2."""
        assert pad_to_power_of_2(1) == 1
        assert pad_to_power_of_2(2) == 2
        assert pad_to_power_of_2(3) == 4
        assert pad_to_power_of_2(8) == 8
        assert pad_to_power_of_2(9) == 16
        assert pad_to_power_of_2(100) == 128

    def test_pad_to_power_of_2_invalid(self):
        """Test power of 2 with invalid input."""
        with pytest.raises(ValueError, match="Length must be positive"):
            pad_to_power_of_2(0)

        with pytest.raises(ValueError, match="Length must be positive"):
            pad_to_power_of_2(-1)

    def test_pad_for_convolution(self, device):
        """Test padding for convolution."""
        x = torch.randn(10, device=device)
        result = pad_for_convolution(x, kernel_size=5)

        # Should pad by 2 on each side for kernel size 5
        assert result.shape[-1] == 14
        # Check that original data is preserved (2 zeros on left, original data, 2 zeros on right)
        torch.testing.assert_close(result[2:12], x)

    def test_pad_for_convolution_invalid_kernel(self, device):
        """Test convolution padding with invalid kernel."""
        x = torch.randn(10, device=device)

        with pytest.raises(ValueError, match="Kernel size must be positive odd"):
            pad_for_convolution(x, kernel_size=4)  # Even

        with pytest.raises(ValueError, match="Kernel size must be positive odd"):
            pad_for_convolution(x, kernel_size=0)  # Zero


class TestInitialization:
    """Test initialization functions."""

    def test_spectral_init_normal(self, device):
        """Test normal spectral initialization."""
        tensor = torch.empty(10, 20, device=device)
        result = spectral_init(tensor, mode="normal", gain=2.0)

        # Should be in-place modification
        assert result is tensor

        # Check statistics
        assert abs(tensor.mean().item()) < 0.5  # Approximately zero mean
        assert abs(tensor.std().item() - 2.0) < 0.5  # Approximately gain std

    def test_spectral_init_uniform(self, device):
        """Test uniform spectral initialization."""
        tensor = torch.empty(10, 20, device=device)
        spectral_init(tensor, mode="uniform", gain=1.5)

        # Check bounds
        assert tensor.min() >= -1.5
        assert tensor.max() <= 1.5

    def test_spectral_init_invalid_mode(self, device):
        """Test spectral init with invalid mode."""
        tensor = torch.empty(10, 20, device=device)

        with pytest.raises(ValueError, match="Unsupported initialization mode"):
            spectral_init(tensor, mode="invalid")

    def test_spectral_init_invalid_gain(self, device):
        """Test spectral init with invalid gain."""
        tensor = torch.empty(10, 20, device=device)

        with pytest.raises(ValueError, match="Gain must be positive"):
            spectral_init(tensor, gain=0.0)

    def test_xavier_spectral_init_normal(self, device):
        """Test Xavier spectral initialization with normal distribution."""
        tensor = torch.empty(50, 30, device=device)
        xavier_spectral_init(tensor, distribution="normal")

        # Xavier scaling: sqrt(2/(fan_in + fan_out)) = sqrt(2/80) ≈ 0.158
        expected_std = math.sqrt(2.0 / 80)
        assert abs(tensor.std().item() - expected_std) < 0.05

    def test_xavier_spectral_init_uniform(self, device):
        """Test Xavier spectral initialization with uniform distribution."""
        tensor = torch.empty(40, 60, device=device)
        xavier_spectral_init(tensor, distribution="uniform")

        # Xavier bound: sqrt(6/(fan_in + fan_out)) = sqrt(6/100) = sqrt(0.06) ≈ 0.245
        expected_bound = math.sqrt(6.0 / 100)
        assert tensor.min() >= -expected_bound - 0.01
        assert tensor.max() <= expected_bound + 0.01

    def test_xavier_spectral_init_1d_error(self, device):
        """Test Xavier init error with 1D tensor."""
        tensor = torch.empty(10, device=device)

        with pytest.raises(ValueError, match="at least 2D tensor"):
            xavier_spectral_init(tensor)

    def test_kaiming_spectral_init_fan_in(self, device):
        """Test Kaiming spectral initialization with fan_in."""
        tensor = torch.empty(50, 30, device=device)
        kaiming_spectral_init(tensor, mode="fan_in", nonlinearity="relu")

        # Kaiming scaling for ReLU: sqrt(2)/sqrt(fan_in) = sqrt(2/50) ≈ 0.2
        expected_std = math.sqrt(2.0 / 50)
        assert abs(tensor.std().item() - expected_std) < 0.05

    def test_kaiming_spectral_init_fan_out(self, device):
        """Test Kaiming spectral initialization with fan_out."""
        tensor = torch.empty(50, 30, device=device)
        kaiming_spectral_init(tensor, mode="fan_out", nonlinearity="linear")

        # Linear gain is 1.0, so std = 1/sqrt(30)
        expected_std = 1.0 / math.sqrt(30)
        assert abs(tensor.std().item() - expected_std) < 0.05

    def test_kaiming_spectral_init_invalid_nonlinearity(self, device):
        """Test Kaiming init with invalid nonlinearity."""
        tensor = torch.empty(10, 20, device=device)

        with pytest.raises(ValueError, match="Unsupported nonlinearity"):
            kaiming_spectral_init(tensor, nonlinearity="invalid")

    def test_orthogonal_spectral_init(self, device):
        """Test orthogonal spectral initialization."""
        tensor = torch.empty(20, 20, device=device)
        orthogonal_spectral_init(tensor)

        # Check orthogonality: A @ A.T = I
        product = torch.matmul(tensor, tensor.T)
        identity = torch.eye(20, device=device)

        torch.testing.assert_close(product, identity, rtol=1e-3, atol=1e-4)

    def test_orthogonal_spectral_init_non_square(self, device):
        """Test orthogonal init with non-square matrix."""
        tensor = torch.empty(20, 15, device=device)
        orthogonal_spectral_init(tensor)

        # For non-square, check that columns are orthogonal
        if tensor.shape[0] >= tensor.shape[1]:
            # More rows than columns
            product = torch.matmul(tensor.T, tensor)
            identity = torch.eye(15, device=device)
            torch.testing.assert_close(product, identity, rtol=1e-3, atol=1e-4)

    def test_orthogonal_spectral_init_1d_error(self, device):
        """Test orthogonal init error with 1D tensor."""
        tensor = torch.empty(10, device=device)

        with pytest.raises(ValueError, match="2D tensor"):
            orthogonal_spectral_init(tensor)

    def test_complex_normal_init(self, device):
        """Test complex normal initialization."""
        tensor = torch.empty(10, 20, dtype=torch.complex64, device=device)
        complex_normal_init(tensor, std=2.0)

        # Each component should have std ≈ 2/sqrt(2) ≈ 1.414
        real_part = tensor.real
        imag_part = tensor.imag

        component_std = 2.0 / math.sqrt(2)
        assert abs(real_part.std().item() - component_std) < 0.2
        assert abs(imag_part.std().item() - component_std) < 0.2

    def test_complex_normal_init_type_error(self, device):
        """Test complex normal init type error."""
        tensor = torch.empty(10, 20, device=device)  # Real tensor

        with pytest.raises(TypeError, match="Tensor must be complex"):
            complex_normal_init(tensor)

    def test_complex_xavier_init(self, device):
        """Test complex Xavier initialization."""
        tensor = torch.empty(40, 60, dtype=torch.complex64, device=device)
        complex_xavier_init(tensor)

        # Xavier std for complex: sqrt(1/(fan_in + fan_out)) = sqrt(1/100) = 0.1
        expected_std = math.sqrt(1.0 / 100)

        # Total variance should be close to expected
        total_var = tensor.real.var() + tensor.imag.var()
        expected_var = expected_std ** 2

        assert abs(total_var.item() - expected_var) < 0.02

    def test_complex_kaiming_init(self, device):
        """Test complex Kaiming initialization."""
        tensor = torch.empty(50, 30, dtype=torch.complex64, device=device)
        complex_kaiming_init(tensor, mode="fan_in")

        # Kaiming std for complex: 1/sqrt(fan_in) = 1/sqrt(50) ≈ 0.141
        expected_std = 1.0 / math.sqrt(50)

        # Each component should have approximately expected_std/sqrt(2)
        component_std = expected_std / math.sqrt(2)

        assert abs(tensor.real.std().item() - component_std) < 0.05
        assert abs(tensor.imag.std().item() - component_std) < 0.05

    def test_frequency_init(self, device):
        """Test frequency-domain initialization."""
        tensor = torch.empty(10, 32, device=device)
        frequency_init(tensor, max_freq=2.0)

        # Low frequencies should have larger values than high frequencies
        low_freq_var = tensor[..., :8].var()  # First quarter
        high_freq_var = tensor[..., -8:].var()  # Last quarter

        assert low_freq_var > high_freq_var

    def test_frequency_init_invalid_freq(self, device):
        """Test frequency init with invalid frequency."""
        tensor = torch.empty(10, 32, device=device)

        with pytest.raises(ValueError, match="Max frequency must be positive"):
            frequency_init(tensor, max_freq=0.0)

    def test_wavelet_init_haar(self, device):
        """Test wavelet initialization with Haar."""
        tensor = torch.empty(10, 8, device=device)
        wavelet_init(tensor, wavelet_type="db1")

        # Check that alternating sign pattern was applied
        # The sign flipping should be applied - check that odd/even indices differ
        # After sign flipping, odd and even indices should have different characteristics
        assert not torch.allclose(tensor[..., 0::2], tensor[..., 1::2])

    def test_wavelet_init_invalid_type(self, device):
        """Test wavelet init with invalid type."""
        tensor = torch.empty(10, 8, device=device)

        with pytest.raises(ValueError, match="Wavelet type must be one of"):
            wavelet_init(tensor, wavelet_type="invalid")

    def test_hadamard_init(self, device):
        """Test Hadamard matrix initialization."""
        tensor = torch.empty(8, 8, device=device)
        hadamard_init(tensor)

        # Check orthogonality
        product = torch.matmul(tensor, tensor.T)
        identity = torch.eye(8, device=device)

        torch.testing.assert_close(product, identity, rtol=1e-5, atol=1e-6)

    def test_hadamard_init_non_square(self, device):
        """Test Hadamard init with non-square matrix."""
        tensor = torch.empty(8, 4, device=device)

        with pytest.raises(ValueError, match="square tensor"):
            hadamard_init(tensor)

    def test_hadamard_init_non_power_of_2(self, device):
        """Test Hadamard init with non-power-of-2 size."""
        tensor = torch.empty(6, 6, device=device)

        with pytest.raises(ValueError, match="power-of-2 size"):
            hadamard_init(tensor)

    def test_dct_init(self, device):
        """Test DCT matrix initialization."""
        tensor = torch.empty(8, 8, device=device)
        dct_init(tensor)

        # Check orthogonality
        product = torch.matmul(tensor, tensor.T)
        identity = torch.eye(8, device=device)

        torch.testing.assert_close(product, identity, rtol=1e-5, atol=1e-6)

    def test_dct_init_1d_error(self, device):
        """Test DCT init error with 1D tensor."""
        tensor = torch.empty(10, device=device)

        with pytest.raises(ValueError, match="2D tensor"):
            dct_init(tensor)

    def test_init_linear_spectral_xavier(self, device):
        """Test linear layer spectral initialization with Xavier."""
        linear = nn.Linear(20, 10).to(device)
        init_linear_spectral(linear, method="xavier")

        # Check Xavier scaling
        expected_std = math.sqrt(2.0 / 30)  # fan_in + fan_out = 30
        assert abs(linear.weight.std().item() - expected_std) < 0.05

        # Bias should be zero
        torch.testing.assert_close(linear.bias, torch.zeros_like(linear.bias))

    def test_init_linear_spectral_orthogonal(self, device):
        """Test linear layer spectral initialization with orthogonal."""
        linear = nn.Linear(15, 15).to(device)  # Square for orthogonal
        init_linear_spectral(linear, method="orthogonal")

        # Check orthogonality
        weight = linear.weight
        product = torch.matmul(weight, weight.T)
        identity = torch.eye(15, device=device)

        torch.testing.assert_close(product, identity, rtol=1e-3, atol=1e-4)

    def test_init_conv_spectral(self, device):
        """Test convolution layer spectral initialization."""
        conv = nn.Conv1d(16, 32, kernel_size=3).to(device)
        init_conv_spectral(conv, method="kaiming")

        # Check that bias is zero
        if conv.bias is not None:
            torch.testing.assert_close(conv.bias, torch.zeros_like(conv.bias))

        # Weight should be properly initialized (hard to test exact values)
        assert not torch.allclose(conv.weight, torch.zeros_like(conv.weight))

    def test_init_linear_spectral_invalid_method(self, device):
        """Test linear init with invalid method."""
        linear = nn.Linear(10, 5).to(device)

        with pytest.raises(ValueError, match="Unsupported method"):
            init_linear_spectral(linear, method="invalid")

    def test_init_conv_spectral_invalid_method(self, device):
        """Test conv init with invalid method."""
        conv = nn.Conv1d(8, 16, 3).to(device)

        with pytest.raises(ValueError, match="Unsupported method"):
            init_conv_spectral(conv, method="invalid")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_tensor_complex_ops(self, device):
        """Test complex operations with empty tensors."""
        empty_complex = torch.empty(0, dtype=torch.complex64, device=device)

        # Operations should handle empty tensors gracefully
        result = complex_conjugate(empty_complex)
        assert result.shape == (0,)
        assert result.dtype == torch.complex64

    def test_single_element_padding(self, device):
        """Test padding operations with single elements."""
        x = torch.tensor([5.0], device=device)
        result = zero_pad(x, 2)
        expected = torch.tensor([5.0, 0.0, 0.0], device=device)

        torch.testing.assert_close(result, expected)

    def test_large_batch_processing(self, device):
        """Test operations with large batch sizes."""
        large_batch = torch.randn(1000, 50, dtype=torch.complex64, device=device)

        # Complex operations should handle large batches
        conj = complex_conjugate(large_batch)
        assert conj.shape == large_batch.shape

        # Initialization should work on large tensors
        large_real = torch.empty(1000, 100, device=device)
        xavier_spectral_init(large_real)
        assert not torch.isnan(large_real).any()

    def test_dimension_edge_cases(self, device):
        """Test operations with edge case dimensions."""
        # Test with different dimension orders
        tensor_2d = torch.randn(5, 10, device=device)
        tensor_3d = torch.randn(2, 5, 10, device=device)
        tensor_4d = torch.randn(1, 2, 5, 10, device=device)

        for tensor in [tensor_2d, tensor_3d, tensor_4d]:
            # Padding should work on any dimension
            padded = pad_to_length(tensor, tensor.shape[-1] + 5, dim=-1)
            assert padded.shape[:-1] == tensor.shape[:-1]
            assert padded.shape[-1] == tensor.shape[-1] + 5


if __name__ == "__main__":
    pytest.main([__file__])
