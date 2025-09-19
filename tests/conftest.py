"""Pytest configuration and fixtures for spectrans tests."""

import random

import numpy as np
import pytest
import torch


# Set random seeds for reproducibility
def pytest_sessionstart(session):
    """Configure pytest session with random seeds."""
    # Set random seeds
    random.seed(42)
    # Use modern numpy random generator instead of legacy seed
    _ = np.random.default_rng(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Set torch to deterministic mode for testing
    torch.use_deterministic_algorithms(True, warn_only=True)


# Device fixtures
@pytest.fixture(scope="session")
def device():
    """Get the default device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device():
    """Get CPU device."""
    return torch.device("cpu")


# Tensor creation fixtures
@pytest.fixture
def batch_size():
    """Default batch size for testing."""
    return 4


@pytest.fixture
def sequence_length():
    """Default sequence length for testing."""
    return 128


@pytest.fixture
def hidden_dim():
    """Default hidden dimension for testing."""
    return 256


@pytest.fixture
def num_heads():
    """Default number of attention heads for testing."""
    return 8


@pytest.fixture
def random_tensor(batch_size, sequence_length, hidden_dim, device):
    """Create a random tensor with standard dimensions."""
    return torch.randn(batch_size, sequence_length, hidden_dim, device=device)


@pytest.fixture
def complex_tensor(batch_size, sequence_length, hidden_dim, device):
    """Create a random complex tensor."""
    real = torch.randn(batch_size, sequence_length, hidden_dim, device=device)
    imag = torch.randn(batch_size, sequence_length, hidden_dim, device=device)
    return torch.complex(real, imag)


@pytest.fixture
def attention_mask(batch_size, sequence_length, device):
    """Create a random attention mask."""
    # Create mask with ~10% masked positions
    mask = torch.rand(batch_size, sequence_length, device=device) > 0.1
    return mask


@pytest.fixture
def causal_mask(sequence_length, device):
    """Create a causal attention mask."""
    mask = torch.triu(
        torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool), diagonal=1
    )
    return ~mask  # Invert to get lower triangular


# Model configuration fixtures
@pytest.fixture
def simple_config():
    """Simple model configuration for testing."""
    return {
        "type": "test_model",
        "params": {
            "num_layers": 2,
            "hidden_dim": 256,
            "sequence_length": 128,
            "dropout": 0.1,
        },
    }


@pytest.fixture
def layer_config():
    """Layer configuration for testing."""
    return {
        "type": "test_layer",
        "params": {
            "hidden_dim": 256,
            "dropout": 0.1,
        },
    }


# Path fixtures
@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def config_file(test_data_dir):
    """Create a temporary config file."""
    import yaml

    config = {
        "model": {
            "type": "test_model",
            "params": {
                "num_layers": 4,
                "hidden_dim": 512,
            },
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
        },
    }

    config_path = test_data_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


# Registry fixtures
@pytest.fixture
def clean_registry():
    """Provide a clean registry for testing."""
    from spectrans.core.registry import registry

    # Store current state
    original_components = {}
    for category in registry._components:
        original_components[category] = registry._components[category].copy()

    # Clear registry
    registry.clear()

    yield registry

    # Restore original state
    for category, components in original_components.items():
        registry._components[category] = components


# Mock component fixtures
@pytest.fixture
def mock_transform_class():
    """Create a mock transform class for testing."""
    from spectrans.core.base import SpectralTransform

    class MockTransform(SpectralTransform):
        def transform(self, x, dim=-1):
            return x * 2.0

        def inverse_transform(self, x, dim=-1):
            return x / 2.0

    return MockTransform


@pytest.fixture
def mock_mixing_layer_class():
    """Create a mock mixing layer class for testing."""
    from spectrans.core.base import MixingLayer

    class MockMixingLayer(MixingLayer):
        def forward(self, x):
            return x * 1.5

    return MockMixingLayer


# Performance testing fixtures
@pytest.fixture
def benchmark_sizes():
    """Different sizes for benchmarking."""
    return [
        (2, 128, 256),  # Small
        (4, 512, 512),  # Medium
        (8, 1024, 768),  # Large
    ]


# Utility fixtures
@pytest.fixture
def assert_tensor_equal():
    """Utility function to assert tensor equality."""

    def _assert_equal(tensor1, tensor2, rtol=1e-5, atol=1e-7):
        assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

    return _assert_equal


@pytest.fixture
def assert_shape():
    """Utility function to assert tensor shape."""

    def _assert_shape(tensor, expected_shape):
        assert (
            tensor.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {tensor.shape}"

    return _assert_shape


@pytest.fixture
def assert_dtype():
    """Utility function to assert tensor dtype."""

    def _assert_dtype(tensor, expected_dtype):
        assert (
            tensor.dtype == expected_dtype
        ), f"Expected dtype {expected_dtype}, got {tensor.dtype}"

    return _assert_dtype


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "benchmark: marks performance benchmark tests")
