"""Unit tests for spectrans configuration system."""

from unittest.mock import mock_open, patch

import pytest
from pydantic import ValidationError

from spectrans.config import (
    ConfigBuilder,
    ConfigurationError,
    build_model_from_config,
    load_yaml_config,
)
from spectrans.config.layers.attention import (
    LSTAttentionConfig,
    SpectralAttentionConfig,
)
from spectrans.config.layers.mixing import (
    AFNOMixingConfig,
    FourierMixingConfig,
    GlobalFilterMixingConfig,
)
from spectrans.config.models import (
    AFNOModelConfig,
    FNetModelConfig,
    GFNetModelConfig,
    HybridModelConfig,
    LSTModelConfig,
    SpectralAttentionModelConfig,
    WaveletTransformerConfig,
)


class TestBaseLayerConfigurations:
    """Test base layer configuration classes."""

    def test_fourier_mixing_config(self):
        """Test FourierMixingConfig validation."""
        config = FourierMixingConfig(hidden_dim=768)
        assert config.hidden_dim == 768
        assert config.dropout == 0.0  # default
        assert config.fft_norm == "ortho"  # default

    def test_fourier_mixing_config_invalid(self):
        """Test FourierMixingConfig with invalid parameters."""
        with pytest.raises(ValidationError):
            FourierMixingConfig(hidden_dim=0)  # Must be positive

        with pytest.raises(ValidationError):
            FourierMixingConfig(hidden_dim=768, dropout=1.5)  # Must be <= 1.0

    def test_global_filter_mixing_config(self):
        """Test GlobalFilterMixingConfig validation."""
        config = GlobalFilterMixingConfig(hidden_dim=512, sequence_length=256, activation="sigmoid")
        assert config.hidden_dim == 512
        assert config.sequence_length == 256
        assert config.activation == "sigmoid"

    def test_afno_mixing_config(self):
        """Test AFNOMixingConfig validation."""
        config = AFNOMixingConfig(
            hidden_dim=768, max_sequence_length=512, modes_seq=256, modes_hidden=384
        )
        assert config.hidden_dim == 768
        assert config.max_sequence_length == 512
        assert config.modes_seq == 256
        assert config.modes_hidden == 384
        assert config.mlp_ratio == 2.0  # default

    def test_spectral_attention_config(self):
        """Test SpectralAttentionConfig validation."""
        config = SpectralAttentionConfig(hidden_dim=768, num_heads=8, num_features=256)
        assert config.hidden_dim == 768
        assert config.num_heads == 8
        assert config.num_features == 256
        assert config.kernel_type == "softmax"  # default

    def test_lst_attention_config(self):
        """Test LSTAttentionConfig validation."""
        config = LSTAttentionConfig(hidden_dim=512, num_heads=8, transform_type="dct")
        assert config.hidden_dim == 512
        assert config.num_heads == 8
        assert config.transform_type == "dct"
        assert config.learnable_scale is True  # default


class TestModelConfigurations:
    """Test complete model configuration classes."""

    def test_fnet_model_config(self):
        """Test FNetModelConfig validation."""
        config = FNetModelConfig(
            hidden_dim=768, num_layers=12, sequence_length=512, use_real_fft=True
        )
        assert config.model_type == "fnet"
        assert config.hidden_dim == 768
        assert config.num_layers == 12
        assert config.sequence_length == 512
        assert config.use_real_fft is True

    def test_gfnet_model_config(self):
        """Test GFNetModelConfig validation."""
        config = GFNetModelConfig(
            hidden_dim=512, num_layers=8, sequence_length=224, filter_activation="sigmoid"
        )
        assert config.model_type == "gfnet"
        assert config.filter_activation == "sigmoid"

    def test_afno_model_config(self):
        """Test AFNOModelConfig validation."""
        config = AFNOModelConfig(hidden_dim=768, num_layers=12, sequence_length=512, n_modes=256)
        assert config.model_type == "afno"
        assert config.n_modes == 256
        assert config.compression_ratio == 0.5  # default

    def test_lst_model_config(self):
        """Test LSTModelConfig validation."""
        config = LSTModelConfig(
            hidden_dim=512, num_layers=6, sequence_length=1024, transform_type="dct"
        )
        assert config.model_type == "lst"
        assert config.transform_type == "dct"
        assert config.use_conv_bias is True  # default

    def test_spectral_attention_model_config(self):
        """Test SpectralAttentionModelConfig validation."""
        config = SpectralAttentionModelConfig(
            hidden_dim=768,
            num_layers=12,
            sequence_length=2048,
            num_features=768,
            kernel_type="gaussian",
        )
        assert config.model_type == "spectral_attention"
        assert config.num_features == 768
        assert config.kernel_type == "gaussian"

    def test_wavelet_transformer_config(self):
        """Test WaveletTransformerConfig validation."""
        config = WaveletTransformerConfig(
            hidden_dim=512, num_layers=8, sequence_length=512, wavelet="db4", levels=3
        )
        assert config.model_type == "wavelet_transformer"
        assert config.wavelet == "db4"
        assert config.levels == 3

    def test_hybrid_model_config(self):
        """Test HybridModelConfig validation."""
        config = HybridModelConfig(
            hidden_dim=768,
            num_layers=12,
            sequence_length=1024,
            spectral_type="fourier",
            spatial_type="attention",
        )
        assert config.model_type == "hybrid"
        assert config.spectral_type == "fourier"
        assert config.spatial_type == "attention"
        assert config.alternation_pattern == "even_spectral"  # default


class TestConfigBuilder:
    """Test ConfigBuilder functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.builder = ConfigBuilder()
        self.sample_config = {
            "model": {
                "model_type": "fnet",
                "hidden_dim": 768,
                "num_layers": 12,
                "sequence_length": 512,
            }
        }

    def test_config_builder_init(self):
        """Test ConfigBuilder initialization."""
        builder = ConfigBuilder(strict_validation=False)
        assert builder.strict_validation is False

    def test_validate_config(self):
        """Test configuration validation."""
        builder = ConfigBuilder()
        validated = builder.validate_config(self.sample_config)

        assert "model" in validated
        assert validated["model"]["model_type"] == "fnet"
        assert validated["model"]["hidden_dim"] == 768

    def test_validate_config_invalid(self):
        """Test validation with invalid configuration."""
        builder = ConfigBuilder()
        invalid_config = {
            "model": {
                "model_type": "fnet",
                "hidden_dim": -1,  # Invalid
                "num_layers": 12,
                "sequence_length": 512,
            }
        }

        with pytest.raises(ConfigurationError):
            builder.validate_config(invalid_config)

    def test_validate_config_missing_model_type(self):
        """Test validation with missing model type."""
        builder = ConfigBuilder()
        invalid_config = {"model": {"hidden_dim": 768, "num_layers": 12, "sequence_length": 512}}

        with pytest.raises(ConfigurationError, match="must specify 'model_type'"):
            builder.validate_config(invalid_config)

    def test_validate_config_unknown_model_type(self):
        """Test validation with unknown model type."""
        builder = ConfigBuilder()
        invalid_config = {
            "model": {
                "model_type": "unknown_model",
                "hidden_dim": 768,
                "num_layers": 12,
                "sequence_length": 512,
            }
        }

        with pytest.raises(ConfigurationError, match="Unknown model type"):
            builder.validate_config(invalid_config)

    @patch(
        "builtins.open",
        mock_open(
            read_data="model:\n  model_type: fnet\n  hidden_dim: 768\n  num_layers: 12\n  sequence_length: 512"
        ),
    )
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_yaml(self, mock_exists):
        """Test YAML file loading."""
        builder = ConfigBuilder()
        config = builder.load_yaml("test_config.yaml")

        assert "model" in config
        assert config["model"]["model_type"] == "fnet"

    @patch("pathlib.Path.exists", return_value=False)
    def test_load_yaml_file_not_found(self, mock_exists):
        """Test YAML loading with non-existent file."""
        builder = ConfigBuilder()

        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            builder.load_yaml("nonexistent.yaml")

    @patch("builtins.open", mock_open(read_data="invalid: yaml: content: ["))
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_yaml_invalid_yaml(self, mock_exists):
        """Test YAML loading with invalid YAML content."""
        builder = ConfigBuilder()

        with pytest.raises(ConfigurationError, match="Error parsing YAML"):
            builder.load_yaml("invalid.yaml")

    @patch("builtins.open", mock_open(read_data="not_a_dict"))
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_yaml_not_dict(self, mock_exists):
        """Test YAML loading when result is not a dictionary."""
        builder = ConfigBuilder()

        with pytest.raises(ConfigurationError, match="Configuration must be a dictionary"):
            builder.load_yaml("invalid.yaml")


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch(
        "builtins.open",
        mock_open(
            read_data="model:\n  model_type: fnet\n  hidden_dim: 768\n  num_layers: 12\n  sequence_length: 512"
        ),
    )
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_yaml_config(self, mock_exists):
        """Test load_yaml_config convenience function."""
        config = load_yaml_config("test.yaml")

        assert "model" in config
        assert config["model"]["model_type"] == "fnet"

    def test_build_model_from_config(self):
        """Test build_model_from_config convenience function."""
        from spectrans.core.registry import registry

        config = {
            "model": {
                "model_type": "fnet",
                "hidden_dim": 768,
                "num_layers": 12,
                "sequence_length": 512,
            }
        }

        # Check if fnet model is registered (may vary by test execution context)
        try:
            registry.get("model", "fnet")
            # If registered, the build should succeed
            model = build_model_from_config(config)
            assert model is not None
        except ValueError:
            # If not registered, it should fail with not registered error
            with pytest.raises(ConfigurationError, match="not registered"):
                build_model_from_config(config)

        # Test with invalid model type should always fail
        invalid_config = {
            "model": {
                "model_type": "nonexistent_model",
                "hidden_dim": 768,
                "num_layers": 12,
                "sequence_length": 512,
            }
        }
        with pytest.raises(ConfigurationError, match="Unknown model type"):
            build_model_from_config(invalid_config)


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_full_config_validation_pipeline(self):
        """Test the complete configuration validation pipeline."""
        configs_to_test = [
            {
                "model_type": "fnet",
                "config": {
                    "hidden_dim": 768,
                    "num_layers": 12,
                    "sequence_length": 512,
                    "use_real_fft": True,
                },
            },
            {
                "model_type": "gfnet",
                "config": {
                    "hidden_dim": 512,
                    "num_layers": 8,
                    "sequence_length": 224,
                    "filter_activation": "sigmoid",
                },
            },
            {
                "model_type": "afno",
                "config": {
                    "hidden_dim": 768,
                    "num_layers": 12,
                    "sequence_length": 512,
                    "n_modes": 256,
                },
            },
        ]

        builder = ConfigBuilder()

        for test_case in configs_to_test:
            model_config = {"model": {"model_type": test_case["model_type"], **test_case["config"]}}

            # Should validate without errors
            validated = builder.validate_config(model_config)
            assert validated["model"]["model_type"] == test_case["model_type"]

    def test_config_edge_cases(self):
        """Test configuration edge cases and boundary conditions."""
        builder = ConfigBuilder()

        # Test minimum valid configuration
        minimal_config = {
            "model": {"model_type": "fnet", "hidden_dim": 1, "num_layers": 1, "sequence_length": 1}
        }

        validated = builder.validate_config(minimal_config)
        assert validated["model"]["hidden_dim"] == 1

        # Test configuration with all optional parameters
        maximal_config = {
            "model": {
                "model_type": "spectral_attention",
                "hidden_dim": 2048,
                "num_layers": 24,
                "sequence_length": 4096,
                "dropout": 0.5,
                "vocab_size": 100000,
                "num_classes": 1000,
                "ffn_hidden_dim": 8192,
                "use_positional_encoding": False,
                "positional_encoding_type": "none",
                "norm_eps": 1e-8,
                "output_type": "regression",
                "gradient_checkpointing": True,
                "num_features": 2048,
                "kernel_type": "gaussian",
                "use_orthogonal": False,
                "num_heads": 16,
            }
        }

        validated = builder.validate_config(maximal_config)
        assert validated["model"]["hidden_dim"] == 2048
        assert validated["model"]["gradient_checkpointing"] is True
