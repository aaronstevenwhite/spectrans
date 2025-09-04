"""Type-safe configuration builder for spectrans components.

This module provides the ConfigBuilder class for loading YAML configurations
and constructing spectrans models and components with full type safety and
validation using Pydantic configuration models.

Classes
-------
ConfigBuilder
    Main builder class for creating models and components from configuration.

Functions
---------
load_yaml_config(path)
    Load and validate YAML configuration file.
build_model_from_config(config_dict)
    Build a complete model from configuration dictionary.
build_component_from_config(component_type, config_dict)
    Build a specific component from configuration.

Examples
--------
Building a model from YAML file:

>>> from spectrans.config.builder import ConfigBuilder
>>> builder = ConfigBuilder()
>>> model = builder.build_model("configs/fnet.yaml")
>>> print(type(model))
<class 'spectrans.models.fnet.FNet'>

Building a model from dictionary:

>>> config = {
...     "model_type": "fnet",
...     "hidden_dim": 768,
...     "num_layers": 12,
...     "sequence_length": 512
... }
>>> model = builder.build_model_from_dict(config)

Creating components directly:

>>> layer_config = {
...     "type": "fourier_mixing",
...     "hidden_dim": 768,
...     "dropout": 0.1
... }
>>> layer = builder.build_layer("fourier_mixing", layer_config)

Notes
-----
The ConfigBuilder ensures type safety by:
1. Validating all parameters using Pydantic models
2. Checking component compatibility before construction
3. Providing detailed error messages for invalid configurations
4. Supporting both YAML files and Python dictionaries

All configuration validation happens at build time, preventing runtime
errors due to invalid parameters or incompatible component combinations.

See Also
--------
spectrans.config.models : Model configuration schemas
spectrans.config.layers : Layer configuration schemas
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..core.registry import registry
from .layers import (
    AFNOMixingConfig,
    FourierMixingConfig,
    GlobalFilterMixingConfig,
    LSTAttentionConfig,
    SpectralAttentionConfig,
    WaveletMixingConfig,
)
from .models import (
    AFNOModelConfig,
    FNetModelConfig,
    FNOTransformerConfig,
    GFNetModelConfig,
    HybridModelConfig,
    LSTModelConfig,
    SpectralAttentionModelConfig,
    WaveletTransformerConfig,
)

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class ConfigBuilder:
    """Type-safe builder for spectrans models and components.

    The ConfigBuilder provides methods to load YAML configurations and
    construct models/components with full type safety and validation.

    Parameters
    ----------
    strict_validation : bool
        Whether to use strict validation mode, defaults to True.
    """

    def __init__(self, strict_validation: bool = True):
        self.strict_validation = strict_validation
        self._model_config_classes = {
            "fnet": FNetModelConfig,
            "gfnet": GFNetModelConfig,
            "afno": AFNOModelConfig,
            "lst": LSTModelConfig,
            "spectral_attention": SpectralAttentionModelConfig,
            "fno_transformer": FNOTransformerConfig,
            "wavelet_transformer": WaveletTransformerConfig,
            "hybrid": HybridModelConfig,
        }
        self._layer_config_classes = {
            # Mixing layer configurations
            "fourier_mixing": FourierMixingConfig,
            "global_filter_mixing": GlobalFilterMixingConfig,
            "wavelet_mixing": WaveletMixingConfig,
            "afno_mixing": AFNOMixingConfig,
            # Attention layer configurations
            "spectral_attention": SpectralAttentionConfig,
            "lst_attention": LSTAttentionConfig,
        }

    def load_yaml(self, config_path: str | Path) -> dict[str, Any]:
        """Load and parse YAML configuration file.

        Parameters
        ----------
        config_path : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        dict[str, Any]
            Parsed configuration dictionary.

        Raises
        ------
        ConfigurationError
            If the file cannot be loaded or parsed.
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")

            with open(config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                raise ConfigurationError(f"Configuration must be a dictionary, got {type(config)}")

            logger.info(f"Loaded configuration from {config_path}")
            return config

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML file {config_path}: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration file {config_path}: {e}") from e

    def build_model(self, config_path: str | Path) -> Any:
        """Build a model from a YAML configuration file.

        Parameters
        ----------
        config_path : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        Any
            The constructed model instance.

        Raises
        ------
        ConfigurationError
            If the model cannot be built from the configuration.
        """
        config_dict = self.load_yaml(config_path)
        return self.build_model_from_dict(config_dict)

    def build_model_from_dict(self, config_dict: dict[str, Any]) -> Any:
        """Build a model from a configuration dictionary.

        Parameters
        ----------
        config_dict : dict[str, Any]
            Configuration dictionary containing model parameters.

        Returns
        -------
        Any
            The constructed model instance.

        Raises
        ------
        ConfigurationError
            If the model cannot be built from the configuration.
        """
        try:
            # Extract model configuration
            if "model" not in config_dict:
                raise ConfigurationError("Configuration must contain a 'model' section")

            model_config_dict = config_dict["model"]
            model_type = model_config_dict.get("model_type")

            if not model_type:
                raise ConfigurationError("Model configuration must specify 'model_type'")

            # Get the appropriate configuration class
            config_class = self._model_config_classes.get(model_type)
            if not config_class:
                available_types = ", ".join(self._model_config_classes.keys())
                raise ConfigurationError(
                    f"Unknown model type '{model_type}'. Available types: {available_types}"
                )

            # Validate configuration using Pydantic
            try:
                validated_config = config_class.model_validate(model_config_dict)  # type: ignore
            except ValidationError as e:
                raise ConfigurationError(f"Invalid configuration for model type '{model_type}': {e}") from e

            # Get model class from registry
            try:
                model_class = registry.get("model", model_type)
            except ValueError as e:
                raise ConfigurationError(f"Model type '{model_type}' not registered: {e}") from e

            # Build the model using the validated configuration
            model = model_class.from_config(validated_config)  # type: ignore
            logger.info(f"Built {model_type} model with configuration: {validated_config.model_dump_json()}")

            return model

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Error building model: {e}") from e

    def build_layer(self, layer_type: str, config_dict: dict[str, Any]) -> Any:
        """Build a layer from configuration dictionary.

        Parameters
        ----------
        layer_type : str
            Type of layer to build.
        config_dict : dict[str, Any]
            Configuration dictionary containing layer parameters.

        Returns
        -------
        Any
            The constructed layer instance.

        Raises
        ------
        ConfigurationError
            If the layer cannot be built from the configuration.
        """
        try:
            # Get the appropriate configuration class
            config_class = self._layer_config_classes.get(layer_type)
            if not config_class:
                available_types = ", ".join(self._layer_config_classes.keys())
                raise ConfigurationError(
                    f"Unknown layer type '{layer_type}'. Available types: {available_types}"
                )

            # Validate configuration using Pydantic
            try:
                validated_config = config_class.model_validate(config_dict)  # type: ignore
            except ValidationError as e:
                raise ConfigurationError(f"Invalid configuration for layer type '{layer_type}': {e}") from e

            # Map layer type to registry category and name
            layer_mapping = {
                "fourier_mixing": ("mixing", "fourier"),
                "global_filter_mixing": ("mixing", "global_filter"),
                "wavelet_mixing": ("mixing", "wavelet"),
                "afno_mixing": ("mixing", "afno"),
                "spectral_attention": ("attention", "spectral"),
                "lst_attention": ("attention", "lst"),
            }

            category, registry_name = layer_mapping.get(layer_type, (None, None))
            if not category:
                raise ConfigurationError(f"Layer type '{layer_type}' not mapped to registry")

            # Get layer class from registry
            try:
                layer_class = registry.get(category, registry_name)  # type: ignore
            except ValueError as e:
                raise ConfigurationError(f"Layer type '{layer_type}' not registered: {e}") from e

            # Build the layer using the validated configuration
            if hasattr(layer_class, 'from_config'):
                layer = layer_class.from_config(validated_config)
            else:
                # Fallback to direct instantiation using config attributes
                layer = layer_class(**validated_config.model_dump())

            logger.info(f"Built {layer_type} layer with configuration: {validated_config.model_dump_json()}")
            return layer

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Error building layer '{layer_type}': {e}") from e

    def validate_config(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Validate a configuration dictionary without building components.

        Parameters
        ----------
        config_dict : dict[str, Any]
            Configuration dictionary to validate.

        Returns
        -------
        dict[str, Any]
            Validated configuration dictionary.

        Raises
        ------
        ConfigurationError
            If the configuration is invalid.
        """
        try:
            if "model" not in config_dict:
                raise ConfigurationError("Configuration must contain a 'model' section")

            model_config_dict = config_dict["model"]
            model_type = model_config_dict.get("model_type")

            if not model_type:
                raise ConfigurationError("Model configuration must specify 'model_type'")

            # Get the appropriate configuration class
            config_class = self._model_config_classes.get(model_type)
            if not config_class:
                available_types = ", ".join(self._model_config_classes.keys())
                raise ConfigurationError(
                    f"Unknown model type '{model_type}'. Available types: {available_types}"
                )

            # Validate using Pydantic
            validated_config = config_class.model_validate(model_config_dict)  # type: ignore

            # Return the original config dict with validated model section
            result = config_dict.copy()
            result["model"] = validated_config.model_dump()

            logger.info(f"Configuration validation successful for model type '{model_type}'")
            return result

        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Error validating configuration: {e}") from e


# Convenience functions
def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to the configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    builder = ConfigBuilder()
    return builder.load_yaml(config_path)


def build_model_from_config(config_dict: dict[str, Any]) -> Any:
    """Build model from configuration dictionary.

    Parameters
    ----------
    config_dict : dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    Any
        The constructed model.
    """
    builder = ConfigBuilder()
    return builder.build_model_from_dict(config_dict)


def build_component_from_config(component_type: str, config_dict: dict[str, Any]) -> Any:
    """Build component from configuration dictionary.

    Parameters
    ----------
    component_type : str
        Type of component to build.
    config_dict : dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    Any
        The constructed component.
    """
    builder = ConfigBuilder()
    return builder.build_layer(component_type, config_dict)
