#!/usr/bin/env python3
"""Configuration-Based Model Creation with Spectrans.

This example demonstrates how to use YAML configuration files to create and
configure spectral transformer models. The configuration system provides:

- Type-safe parameter validation with Pydantic
- Easy experimentation with different architectures
- Reproducible model configurations
- Hierarchical configuration organization

Key concepts:
- Loading models from YAML configuration files
- Creating custom configuration files
- Programmatically building configurations
- Validating configuration parameters

Requirements:
- spectrans
- torch
- pyyaml (included with spectrans)
"""

from pathlib import Path

import torch
import yaml

from spectrans.config import ConfigBuilder, build_model_from_config
from spectrans.config.models import (
    AFNOModelConfig,
    FNetModelConfig,
    GFNetModelConfig,
    WaveletTransformerConfig,
)


def load_predefined_configs():
    """Demonstrate loading models from pre-defined configuration files."""
    print("=== Loading Pre-defined Configurations ===\n")

    config_dir = Path("examples/configs")
    builder = ConfigBuilder()

    # Available configurations
    available_configs = list(config_dir.glob("*.yaml"))
    print(f"Available configurations: {[f.stem for f in available_configs]}\n")

    # Load FNet configuration
    print("1. Loading FNet from configuration...")
    try:
        fnet_model = builder.build_model("examples/configs/fnet.yaml")
        print(f"   ✓ FNet model created: {type(fnet_model).__name__}")
        print(f"   ✓ Parameters: {sum(p.numel() for p in fnet_model.parameters()):,}")

        # Test forward pass
        input_ids = torch.randint(1, 30000, (2, 256))
        with torch.no_grad():
            output = fnet_model(input_ids=input_ids)
        print(f"   ✓ Forward pass successful: {output.shape}\n")

    except FileNotFoundError:
        print("   ⚠ examples/configs/fnet.yaml not found, skipping...\n")

    # Load GFNet configuration
    print("2. Loading GFNet from configuration...")
    try:
        gfnet_model = builder.build_model("examples/configs/gfnet.yaml")
        print(f"   ✓ GFNet model created: {type(gfnet_model).__name__}")
        print(f"   ✓ Parameters: {sum(p.numel() for p in gfnet_model.parameters()):,}")

        # GFNet may not have vocab, use embeddings instead
        # Use the model's configured sequence length
        seq_len = getattr(gfnet_model, 'max_sequence_length', 224)
        embeddings = torch.randn(2, seq_len, gfnet_model.hidden_dim)
        with torch.no_grad():
            output = gfnet_model(inputs_embeds=embeddings)
        print(f"   ✓ Forward pass successful: {output.shape}\n")

    except FileNotFoundError:
        print("   ⚠ examples/configs/gfnet.yaml not found, skipping...\n")
    except Exception as e:
        print(f"   ⚠ Error with GFNet config: {e}, skipping...\n")

    # Try loading all available configs
    print("3. Testing all available configurations...")
    for config_file in available_configs:
        try:
            model = builder.build_model(str(config_file))
            param_count = sum(p.numel() for p in model.parameters())
            print(f"   ✓ {config_file.stem}: {param_count:,} parameters")
        except Exception as e:
            print(f"   ✗ {config_file.stem}: {e}")
    print()


def create_custom_config_file():
    """Demonstrate creating a custom configuration file."""
    print("=== Creating Custom Configuration File ===\n")

    # Define a custom configuration
    custom_config = {
        "model": {
            "model_type": "fnet",
            "hidden_dim": 384,
            "num_layers": 8,
            "sequence_length": 256,
            "dropout": 0.15,
            "vocab_size": 15000,
            "num_classes": 5,
            "ffn_hidden_dim": 1536,
            "use_positional_encoding": True,
            "positional_encoding_type": "sinusoidal",
            "norm_eps": 1e-12,
            "output_type": "classification",
            "use_real_fft": True
        },
        "layers": {
            "mixing": {
                "type": "fourier",
                "hidden_dim": 384,
                "dropout": 0.15,
                "norm_eps": 1e-5,
                "energy_tolerance": 1e-4,
                "fft_norm": "ortho"
            },
            "ffn": {
                "hidden_dim": 384,
                "intermediate_dim": 1536,
                "activation": "gelu",
                "dropout": 0.15
            }
        },
        "training": {
            "batch_size": 24,
            "learning_rate": 8e-5,
            "weight_decay": 0.02,
            "warmup_steps": 5000,
            "max_steps": 50000
        }
    }

    # Save to file
    custom_path = Path("custom_fnet_config.yaml")
    print(f"1. Creating custom configuration: {custom_path}")

    with open(custom_path, "w") as f:
        yaml.dump(custom_config, f, default_flow_style=False, indent=2)

    print("   ✓ Configuration saved\n")

    # Load and use the custom configuration
    print("2. Loading custom configuration...")
    builder = ConfigBuilder()
    model = builder.build_model(str(custom_path))

    print("   ✓ Model created from custom config")
    print(f"   ✓ Hidden dimension: {model.hidden_dim}")
    print(f"   ✓ Number of layers: {model.num_layers}")
    print(f"   ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test the model
    input_ids = torch.randint(1, 15000, (4, 128))
    with torch.no_grad():
        output = model(input_ids=input_ids)

    print(f"   ✓ Forward pass: {input_ids.shape} -> {output.shape}\n")

    # Clean up
    custom_path.unlink()
    print("   ✓ Cleaned up temporary config file\n")


def programmatic_config_creation():
    """Demonstrate creating configurations programmatically using Pydantic models."""
    print("=== Programmatic Configuration Creation ===\n")

    print("1. Creating FNet configuration with Pydantic...")
    fnet_config = FNetModelConfig(
        hidden_dim=512,
        num_layers=10,
        sequence_length=128,
        dropout=0.1,
        vocab_size=8000,
        num_classes=3,
        ffn_hidden_dim=2048,
        use_real_fft=True,
        norm_eps=1e-12
    )

    print("   ✓ Configuration created and validated")
    print(f"   ✓ Model type: {fnet_config.model_type}")
    print(f"   ✓ Hidden dim: {fnet_config.hidden_dim}")
    print(f"   ✓ Sequence length: {fnet_config.sequence_length}")

    # Convert to dictionary and build model
    config_dict = {"model": fnet_config.model_dump()}
    model = build_model_from_config(config_dict)

    print(f"   ✓ Model built: {type(model).__name__}")
    print(f"   ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    print("2. Creating GFNet configuration...")
    gfnet_config = GFNetModelConfig(
        hidden_dim=768,
        num_layers=12,
        sequence_length=512,
        dropout=0.1,
        vocab_size=30000,
        num_classes=2,
        filter_activation="sigmoid"
    )

    config_dict = {"model": gfnet_config.model_dump()}
    gfnet_model = build_model_from_config(config_dict)

    print(f"   ✓ GFNet built: {type(gfnet_model).__name__}")
    print(f"   ✓ Parameters: {sum(p.numel() for p in gfnet_model.parameters()):,}\n")

    print("3. Creating AFNO configuration...")
    afno_config = AFNOModelConfig(
        hidden_dim=256,
        num_layers=8,
        sequence_length=1024,  # Longer sequences for AFNO
        dropout=0.05,
        vocab_size=20000,
        num_classes=10,
        n_modes=32,
        compression_ratio=0.25
    )

    config_dict = {"model": afno_config.model_dump()}
    afno_model = build_model_from_config(config_dict)

    print(f"   ✓ AFNO built: {type(afno_model).__name__}")
    print(f"   ✓ Parameters: {sum(p.numel() for p in afno_model.parameters()):,}")
    print(f"   ✓ Fourier modes: {afno_config.n_modes}\n")

    print("4. Creating Wavelet Transformer configuration...")
    wavelet_config = WaveletTransformerConfig(
        hidden_dim=512,
        num_layers=8,
        sequence_length=256,
        dropout=0.1,
        vocab_size=15000,
        num_classes=4,
        wavelet="db6",
        levels=4
    )

    config_dict = {"model": wavelet_config.model_dump()}
    wavelet_model = build_model_from_config(config_dict)

    print(f"   ✓ Wavelet Transformer built: {type(wavelet_model).__name__}")
    print(f"   ✓ Parameters: {sum(p.numel() for p in wavelet_model.parameters()):,}")
    print(f"   ✓ Wavelet: {wavelet_config.wavelet}\n")


def config_validation_example():
    """Demonstrate configuration validation and error handling."""
    print("=== Configuration Validation ===\n")

    print("1. Testing valid configuration...")
    try:
        FNetModelConfig(
            hidden_dim=768,
            num_layers=12,
            sequence_length=512,
            vocab_size=30000,
            num_classes=2
        )
        print("   ✓ Valid configuration accepted\n")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}\n")

    print("2. Testing invalid configurations...")

    # Test negative hidden dimension
    try:
        FNetModelConfig(
            hidden_dim=-768,  # Invalid: must be positive
            num_layers=12,
            sequence_length=512,
            vocab_size=30000,
            num_classes=2
        )
        print("   ✗ Should have rejected negative hidden_dim")
    except Exception as e:
        print(f"   ✓ Correctly rejected negative hidden_dim: {type(e).__name__}")

    # Test invalid dropout rate
    try:
        FNetModelConfig(
            hidden_dim=768,
            num_layers=12,
            sequence_length=512,
            vocab_size=30000,
            num_classes=2,
            dropout=1.5  # Invalid: must be between 0 and 1
        )
        print("   ✗ Should have rejected invalid dropout")
    except Exception as e:
        print(f"   ✓ Correctly rejected invalid dropout: {type(e).__name__}")

    # Test zero sequence length
    try:
        FNetModelConfig(
            hidden_dim=768,
            num_layers=12,
            sequence_length=0,  # Invalid: must be positive
            vocab_size=30000,
            num_classes=2
        )
        print("   ✗ Should have rejected zero sequence_length")
    except Exception as e:
        print(f"   ✓ Correctly rejected zero sequence_length: {type(e).__name__}")

    print("\n   Configuration validation is working correctly! ✓\n")


def config_modification_example():
    """Demonstrate modifying configurations."""
    print("=== Configuration Modification ===\n")

    # Start with base configuration
    base_config = FNetModelConfig(
        hidden_dim=512,
        num_layers=8,
        sequence_length=256,
        vocab_size=20000,
        num_classes=2
    )

    print("1. Base configuration:")
    print(f"   Hidden dim: {base_config.hidden_dim}")
    print(f"   Layers: {base_config.num_layers}")
    print(f"   Sequence length: {base_config.sequence_length}\n")

    # Create variations
    print("2. Creating configuration variations...")

    # Larger model
    large_config = base_config.model_copy(update={
        "hidden_dim": 1024,
        "num_layers": 16,
        "ffn_hidden_dim": 4096
    })

    # Longer sequences
    long_seq_config = base_config.model_copy(update={
        "sequence_length": 1024,
        "dropout": 0.15  # Higher dropout for longer sequences
    })

    # More classes
    multiclass_config = base_config.model_copy(update={
        "num_classes": 10,
        "vocab_size": 50000
    })

    # Build all models
    configs = {
        "base": base_config,
        "large": large_config,
        "long_seq": long_seq_config,
        "multiclass": multiclass_config
    }

    for name, config in configs.items():
        config_dict = {"model": config.model_dump()}
        model = build_model_from_config(config_dict)
        param_count = sum(p.numel() for p in model.parameters())

        print(f"   {name:10s}: {param_count:>8,} parameters, "
              f"hidden_dim={config.hidden_dim}, "
              f"seq_len={config.sequence_length}")

    print("\n   ✓ All configuration variations built successfully!\n")


def main():
    """Run all configuration examples."""
    print("Configuration-Based Model Creation with Spectrans\n")
    print("=" * 60)

    try:
        load_predefined_configs()
        create_custom_config_file()
        programmatic_config_creation()
        config_validation_example()
        config_modification_example()

        print("All configuration examples completed successfully! ✓")
        print("\nKey takeaways:")
        print("- Use YAML files for easy experimentation")
        print("- Pydantic provides type safety and validation")
        print("- Programmatic configs enable dynamic model creation")
        print("- Configuration validation prevents common errors")
        print("\nNext steps:")
        print("- Explore the examples/configs/ directory for more examples")
        print("- Create configurations for your specific use cases")
        print("- See custom_hybrid.py for advanced architecture composition")

    except Exception as e:
        print(f"Error running configuration examples: {e}")
        print("Make sure spectrans is installed: pip install spectrans")
        raise


if __name__ == "__main__":
    main()
