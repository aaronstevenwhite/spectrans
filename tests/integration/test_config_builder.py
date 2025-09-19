"""Integration tests for configuration builder and model creation."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spectrans.config import ConfigBuilder, ConfigurationError


class TestConfigBuilderIntegration:
    """Integration tests for ConfigBuilder with real YAML files."""

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary YAML configuration file."""
        content = """
model:
  model_type: "fnet"
  hidden_dim: 768
  num_layers: 12
  sequence_length: 512
  dropout: 0.1
  use_real_fft: true

layers:
  mixing:
    type: "fourier"
    hidden_dim: 768
    dropout: 0.1
    fft_norm: "ortho"

training:
  batch_size: 32
  learning_rate: 1e-4
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            temp_file = Path(f.name)

        yield temp_file
        temp_file.unlink()  # Clean up

    @pytest.fixture
    def invalid_config_file(self):
        """Create temporary invalid YAML configuration file."""
        content = """
model:
  model_type: "fnet"
  hidden_dim: -1  # Invalid negative value
  num_layers: 0   # Invalid zero value
  sequence_length: 512
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            temp_file = Path(f.name)

        yield temp_file
        temp_file.unlink()

    @pytest.fixture
    def complex_config_file(self):
        """Create temporary complex configuration file."""
        content = """
model:
  model_type: "hybrid"
  hidden_dim: 768
  num_layers: 12
  sequence_length: 1024
  dropout: 0.1
  vocab_size: 50000
  num_classes: 10
  spectral_type: "fourier"
  spatial_type: "attention"
  alternation_pattern: "even_spectral"
  num_heads: 12
  spectral_config:
    fft_norm: "ortho"
  spatial_config:
    dropout: 0.1

layers:
  spectral_mixing:
    type: "fourier"
    hidden_dim: 768
    dropout: 0.1

  spatial_attention:
    type: "attention"
    hidden_dim: 768
    num_heads: 12

training:
  batch_size: 16
  learning_rate: 5e-5
  weight_decay: 0.01
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            temp_file = Path(f.name)

        yield temp_file
        temp_file.unlink()

    def test_load_yaml_integration(self, temp_config_file):
        """Test loading real YAML configuration file."""
        builder = ConfigBuilder()
        config = builder.load_yaml(temp_config_file)

        assert "model" in config
        assert config["model"]["model_type"] == "fnet"
        assert config["model"]["hidden_dim"] == 768
        assert config["model"]["use_real_fft"] is True
        assert "layers" in config
        assert "training" in config

    def test_validate_config_integration(self, temp_config_file):
        """Test validating configuration from real file."""
        builder = ConfigBuilder()
        config = builder.load_yaml(temp_config_file)
        validated = builder.validate_config(config)

        assert validated["model"]["model_type"] == "fnet"
        assert validated["model"]["hidden_dim"] == 768
        # Validation adds default values
        assert "norm_eps" in validated["model"]

    def test_invalid_config_integration(self, invalid_config_file):
        """Test error handling with invalid configuration file."""
        builder = ConfigBuilder()
        config = builder.load_yaml(invalid_config_file)

        with pytest.raises(ConfigurationError):
            builder.validate_config(config)

    def test_complex_config_integration(self, complex_config_file):
        """Test handling complex configuration with nested sections."""
        builder = ConfigBuilder()
        config = builder.load_yaml(complex_config_file)
        validated = builder.validate_config(config)

        assert validated["model"]["model_type"] == "hybrid"
        assert validated["model"]["spectral_type"] == "fourier"
        assert validated["model"]["spatial_type"] == "attention"
        assert validated["model"]["num_heads"] == 12

    @patch("spectrans.core.registry.registry.get")
    def test_model_building_integration(self, mock_registry_get, temp_config_file):
        """Test integration with model building (mocked)."""
        # Mock a model class
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_class.from_config.return_value = mock_model_instance
        mock_registry_get.return_value = mock_model_class

        builder = ConfigBuilder()
        config = builder.load_yaml(temp_config_file)

        # This would normally build a real model
        model = builder.build_model_from_dict(config)

        # Verify the mock was called correctly
        mock_registry_get.assert_called_once_with("model", "fnet")
        mock_model_class.from_config.assert_called_once()
        assert model == mock_model_instance

    def test_end_to_end_config_workflow(self, temp_config_file):
        """Test complete workflow from file to validation."""
        builder = ConfigBuilder()

        # Load -> Validate -> Extract components
        config = builder.load_yaml(temp_config_file)
        validated = builder.validate_config(config)

        # Extract model configuration
        model_config = validated["model"]
        assert isinstance(model_config, dict)

        # Verify all required fields are present
        required_fields = ["model_type", "hidden_dim", "num_layers", "sequence_length"]
        for field in required_fields:
            assert field in model_config
            assert model_config[field] is not None

    def test_build_layer_integration(self):
        """Test building layer components from configuration."""
        builder = ConfigBuilder()

        # Test building a mixing layer
        layer_config = {"hidden_dim": 768, "dropout": 0.1, "fft_norm": "ortho"}

        # This should succeed because the fourier mixing layer is registered
        layer = builder.build_layer("fourier_mixing", layer_config)
        assert layer is not None

        # Test with invalid layer type
        with pytest.raises(ConfigurationError, match="Unknown layer type"):
            builder.build_layer("nonexistent_layer", layer_config)

    def test_builder_error_handling(self):
        """Test error handling in various builder scenarios."""
        builder = ConfigBuilder()

        # Test with missing model section
        invalid_config = {"layers": {"type": "fourier"}}
        with pytest.raises(ConfigurationError, match="must contain a 'model' section"):
            builder.build_model_from_dict(invalid_config)

        # Test with unknown layer type
        layer_config = {"hidden_dim": 768}
        with pytest.raises(ConfigurationError, match="Unknown layer type"):
            builder.build_layer("unknown_layer", layer_config)

    def test_config_builder_strict_mode(self, temp_config_file):
        """Test ConfigBuilder in strict validation mode."""
        strict_builder = ConfigBuilder(strict_validation=True)
        lenient_builder = ConfigBuilder(strict_validation=False)

        config = strict_builder.load_yaml(temp_config_file)

        # Both should validate the same way for valid configs
        strict_result = strict_builder.validate_config(config)
        lenient_result = lenient_builder.validate_config(config)

        assert strict_result["model"]["model_type"] == lenient_result["model"]["model_type"]

    def test_multiple_model_types_integration(self):
        """Test configuration validation for multiple model types."""
        builder = ConfigBuilder()

        model_configs = [
            {
                "model": {
                    "model_type": "fnet",
                    "hidden_dim": 768,
                    "num_layers": 12,
                    "sequence_length": 512,
                    "use_real_fft": True,
                }
            },
            {
                "model": {
                    "model_type": "gfnet",
                    "hidden_dim": 512,
                    "num_layers": 8,
                    "sequence_length": 224,
                    "filter_activation": "sigmoid",
                }
            },
            {
                "model": {
                    "model_type": "afno",
                    "hidden_dim": 768,
                    "num_layers": 12,
                    "sequence_length": 512,
                    "n_modes": 256,
                }
            },
        ]

        for config in model_configs:
            validated = builder.validate_config(config)
            assert "model" in validated
            assert "model_type" in validated["model"]

    def test_config_with_optional_fields(self):
        """Test configuration handling with various optional fields."""
        builder = ConfigBuilder()

        # Config with minimal required fields
        minimal_config = {
            "model": {
                "model_type": "fnet",
                "hidden_dim": 768,
                "num_layers": 12,
                "sequence_length": 512,
            }
        }

        # Config with many optional fields
        detailed_config = {
            "model": {
                "model_type": "spectral_attention",
                "hidden_dim": 768,
                "num_layers": 12,
                "sequence_length": 2048,
                "dropout": 0.1,
                "vocab_size": 50000,
                "num_classes": 1000,
                "ffn_hidden_dim": 3072,
                "use_positional_encoding": True,
                "positional_encoding_type": "alibi",
                "norm_eps": 1e-12,
                "output_type": "classification",
                "gradient_checkpointing": True,
                "num_features": 768,
                "kernel_type": "gaussian",
                "use_orthogonal": True,
                "num_heads": 8,
            }
        }

        # Both should validate successfully
        minimal_validated = builder.validate_config(minimal_config)
        detailed_validated = builder.validate_config(detailed_config)

        assert minimal_validated["model"]["model_type"] == "fnet"
        assert detailed_validated["model"]["model_type"] == "spectral_attention"
        assert detailed_validated["model"]["gradient_checkpointing"] is True
