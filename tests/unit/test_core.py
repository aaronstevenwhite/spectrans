"""Unit tests for spectrans core module.

This module tests the core functionality including base classes, registry system,
and type definitions that form the foundation of the spectrans library.
"""

from typing import Any

import pytest
import torch
import torch.nn as nn

# Import modules to ensure components are registered
import spectrans.models
import spectrans.transforms  # noqa: F401
from spectrans.core.base import (
    AttentionLayer,
    BaseModel,
    SpectralComponent,
    TransformerBlock,
)
from spectrans.core.registry import (
    ComponentRegistry,
    create_component,
    get_component,
    list_components,
    register_component,
    registry,
)
from spectrans.core.types import (
    ComplexTensor,
    Tensor,
)
from spectrans.layers.mixing.base import MixingLayer


class TestSpectralComponent:
    """Test the SpectralComponent abstract base class."""

    def test_abstract_base_class(self):
        """Test that SpectralComponent cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SpectralComponent()

    def test_concrete_implementation(self):
        """Test concrete implementation of SpectralComponent."""

        class ConcreteComponent(SpectralComponent):
            def __init__(self, hidden_dim: int = 256):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.layer = nn.Linear(hidden_dim, hidden_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.layer(x)

        # Should instantiate successfully
        component = ConcreteComponent(hidden_dim=128)
        assert isinstance(component, SpectralComponent)
        assert component.hidden_dim == 128

        # Test forward pass
        x = torch.randn(2, 10, 128)
        output = component(x)
        assert output.shape == x.shape

    def test_missing_abstract_methods(self):
        """Test that missing abstract methods raise TypeError."""

        class IncompleteComponent(SpectralComponent):
            # Missing forward method
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteComponent()


class TestMixingLayer:
    """Test the MixingLayer base class."""

    def test_mixing_layer_inheritance(self):
        """Test that MixingLayer inherits from SpectralComponent."""
        assert issubclass(MixingLayer, SpectralComponent)

    def test_concrete_mixing_layer(self):
        """Test concrete implementation of MixingLayer."""

        class ConcreteMixing(MixingLayer):
            def __init__(self, hidden_dim: int = 256):
                super().__init__(hidden_dim=hidden_dim)
                self.transform = nn.Linear(hidden_dim, hidden_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.transform(x)


            def get_spectral_properties(self) -> dict[str, Any]:
                return {'transform_type': 'linear', 'preserves_energy': False}

        layer = ConcreteMixing(hidden_dim=256)
        assert isinstance(layer, MixingLayer)
        assert isinstance(layer, SpectralComponent)

        # Test forward pass
        x = torch.randn(4, 32, 256)
        output = layer(x)
        assert output.shape == x.shape


class TestAttentionLayer:
    """Test the AttentionLayer base class."""

    def test_attention_layer_inheritance(self):
        """Test that AttentionLayer inherits from SpectralComponent."""
        assert issubclass(AttentionLayer, SpectralComponent)

    def test_concrete_attention_layer(self):
        """Test concrete implementation of AttentionLayer."""

        class ConcreteAttention(AttentionLayer):
            def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
                super().__init__(hidden_dim=hidden_dim, num_heads=num_heads)
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Transpose for MultiheadAttention (seq_len, batch, hidden_dim)
                x_t = x.transpose(0, 1)
                attn_output, _ = self.attention(x_t, x_t, x_t)
                return attn_output.transpose(0, 1)


        layer = ConcreteAttention(hidden_dim=256, num_heads=8)
        assert isinstance(layer, AttentionLayer)
        assert isinstance(layer, SpectralComponent)

        # Test forward pass
        x = torch.randn(2, 16, 256)
        output = layer(x)
        assert output.shape == x.shape


class TestTransformerBlock:
    """Test the TransformerBlock base class."""

    def test_transformer_block_inheritance(self):
        """Test that TransformerBlock inherits from SpectralComponent."""
        assert issubclass(TransformerBlock, SpectralComponent)

    def test_transformer_block_initialization(self):
        """Test TransformerBlock initialization."""

        class MockMixing(MixingLayer):
            def __init__(self, hidden_dim: int):
                super().__init__(hidden_dim=hidden_dim)

            def forward(self, x):
                return x


            def get_spectral_properties(self) -> dict[str, Any]:
                return {'transform_type': 'identity'}

        mixing = MockMixing(hidden_dim=256)
        ffn = nn.Sequential(
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, 256)
        )

        block = TransformerBlock(
            mixing_layer=mixing,
            ffn=ffn,
            dropout=0.1
        )

        assert block.mixing_layer is mixing
        assert block.ffn is ffn
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)
        assert isinstance(block.dropout, nn.Dropout)

    def test_transformer_block_forward(self):
        """Test TransformerBlock forward pass."""

        class SimpleMixing(MixingLayer):
            def __init__(self, hidden_dim: int):
                super().__init__(hidden_dim=hidden_dim)
                self.linear = nn.Linear(hidden_dim, hidden_dim)

            def forward(self, x):
                return self.linear(x)


            def get_spectral_properties(self) -> dict[str, Any]:
                return {'transform_type': 'linear'}

        mixing = SimpleMixing(hidden_dim=128)
        ffn = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 128)
        )

        block = TransformerBlock(
            mixing_layer=mixing,
            ffn=ffn,
            dropout=0.0  # Disable dropout for deterministic testing
        )

        x = torch.randn(2, 10, 128)
        output = block(x)
        assert output.shape == x.shape

        # Check that residual connections are working
        # Output should not be identical to input
        assert not torch.allclose(output, x)


class TestBaseModel:
    """Test the BaseModel abstract base class."""

    def test_abstract_base_class(self):
        """Test that BaseModel needs required arguments."""
        # BaseModel is not abstract, but requires arguments
        with pytest.raises(TypeError, match="missing 2 required positional arguments"):
            BaseModel()

    def test_concrete_model(self):
        """Test concrete implementation of BaseModel."""

        class ConcreteModel(BaseModel):
            def __init__(self, hidden_dim: int = 256, num_layers: int = 2):
                super().__init__(
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    max_seq_length=512,
                    vocab_size=None,
                    num_classes=None,
                    dropout=0.0
                )
                self.blocks = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim)
                    for _ in range(num_layers)
                ])

            def forward(
                self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
                # Simple forward implementation
                for layer in self.blocks:
                    x = layer(x)
                return x


        model = ConcreteModel(hidden_dim=128, num_layers=3)
        assert isinstance(model, BaseModel)
        assert model.num_layers == 3

        # Test forward pass
        x = torch.randn(2, 10, 128)
        output = model(x)
        assert isinstance(output, torch.Tensor)
        assert output.shape == x.shape


class TestComponentRegistry:
    """Test the ComponentRegistry class."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        reg = ComponentRegistry()
        assert 'transform' in reg._components
        assert 'mixing' in reg._components
        assert 'attention' in reg._components
        assert 'block' in reg._components
        assert 'model' in reg._components

    def test_register_and_get_component(self):
        """Test registering and retrieving components."""
        reg = ComponentRegistry()

        class TestTransform:
            pass

        reg.register('transform', 'test', TestTransform)
        retrieved = reg.get('transform', 'test')
        assert retrieved is TestTransform

    def test_register_invalid_category(self):
        """Test registering with invalid category."""
        reg = ComponentRegistry()

        with pytest.raises(ValueError, match="Unknown category"):
            reg.register('invalid_category', 'test', object)

    def test_get_invalid_component(self):
        """Test getting non-existent component."""
        reg = ComponentRegistry()

        with pytest.raises(ValueError, match="Unknown transform"):
            reg.get('transform', 'nonexistent')

    def test_list_components(self):
        """Test listing components in a category."""
        reg = ComponentRegistry()

        class Transform1:
            pass

        class Transform2:
            pass

        reg.register('transform', 'transform1', Transform1)
        reg.register('transform', 'transform2', Transform2)

        components = reg.list('transform')
        assert 'transform1' in components
        assert 'transform2' in components

    def test_clear_registry(self):
        """Test clearing the registry."""
        reg = ComponentRegistry()

        class TestModel:
            pass

        reg.register('model', 'test_model', TestModel)
        assert 'test_model' in reg.list('model')

        reg.clear()
        assert 'test_model' not in reg.list('model')
        assert len(reg.list('model')) == 0


class TestRegistryFunctions:
    """Test the global registry functions."""

    def test_register_component_decorator(self):
        """Test the register_component decorator."""
        # Store original state
        original_models = registry.list('model').copy()

        @register_component('model', 'test_decorated_model')
        class DecoratedModel:
            pass

        # Check it was registered
        assert 'test_decorated_model' in registry.list('model')

        # Clean up
        registry._components['model'] = {
            k: v for k, v in registry._components['model'].items()
            if k in original_models
        }

    def test_get_component_function(self):
        """Test the get_component function."""
        # Register a test component
        original_transforms = registry.list('transform').copy()

        class TestTransform:
            pass

        registry.register('transform', 'test_get', TestTransform)

        # Test get_component
        retrieved = get_component('transform', 'test_get')
        assert retrieved is TestTransform

        # Clean up
        registry._components['transform'] = {
            k: v for k, v in registry._components['transform'].items()
            if k in original_transforms
        }

    def test_create_component_function(self):
        """Test the create_component function."""
        # Register a test component with constructor
        original_mixings = registry.list('mixing').copy()

        class TestMixing:
            def __init__(self, hidden_dim: int = 256):
                self.hidden_dim = hidden_dim

        registry.register('mixing', 'test_create', TestMixing)

        # Test create_component
        instance = create_component('mixing', 'test_create', hidden_dim=512)
        assert isinstance(instance, TestMixing)
        assert instance.hidden_dim == 512

        # Clean up
        registry._components['mixing'] = {
            k: v for k, v in registry._components['mixing'].items()
            if k in original_mixings
        }

    def test_list_components_function(self):
        """Test the list_components function."""
        # Should list existing components
        models = list_components('model')
        assert isinstance(models, list)

        # Should contain registered models
        assert 'fnet' in models
        assert 'gfnet' in models
        assert 'afno' in models


class TestTypes:
    """Test type definitions from core.types module."""

    def test_tensor_type(self):
        """Test Tensor type alias."""
        # Tensor should accept torch.Tensor
        tensor: Tensor = torch.randn(2, 3, 4)
        assert isinstance(tensor, torch.Tensor)

    def test_complex_tensor_type(self):
        """Test ComplexTensor type alias."""
        # ComplexTensor should accept complex tensors
        real = torch.randn(2, 3, 4)
        imag = torch.randn(2, 3, 4)
        complex_tensor: ComplexTensor = torch.complex(real, imag)
        assert complex_tensor.is_complex()


class TestRegistryIntegration:
    """Test registry integration with actual components."""

    def test_registered_transforms(self):
        """Test that transforms are properly registered."""
        transforms = list_components('transform')

        # Check core transforms are registered
        assert 'fft1d' in transforms
        assert 'fft2d' in transforms
        assert 'rfft' in transforms
        assert 'dct' in transforms
        assert 'dst' in transforms
        assert 'hadamard' in transforms
        assert 'dwt1d' in transforms
        assert 'dwt2d' in transforms

    def test_registered_models(self):
        """Test that models are properly registered."""
        models = list_components('model')

        # Check core models are registered
        assert 'fnet' in models
        assert 'fnet_encoder' in models
        assert 'gfnet' in models
        assert 'gfnet_encoder' in models
        assert 'afno' in models
        assert 'afno_encoder' in models

    def test_create_transform_from_registry(self):
        """Test creating transforms from registry."""
        # Create FFT transform
        fft = create_component('transform', 'fft1d', norm='ortho')
        x = torch.randn(2, 10, 128)
        y = fft.transform(x)
        assert y.shape == x.shape
        assert y.is_complex()

        # Create DCT transform
        dct = create_component('transform', 'dct')
        x = torch.randn(2, 10, 128)
        y = dct.transform(x)
        assert y.shape == x.shape
        assert not y.is_complex()

    def test_create_model_from_registry(self):
        """Test creating models from registry."""
        # Create FNet model
        fnet = create_component(
            'model', 'fnet',
            hidden_dim=128,
            num_layers=2,
            max_sequence_length=256
        )

        x = torch.randn(2, 10, 128)
        output = fnet(inputs_embeds=x)
        # FNet returns a tensor, not a ModelOutput object
        assert isinstance(output, torch.Tensor)
        assert output.shape == x.shape
