r"""Component registry system for dynamic component registration and retrieval.

This module implements a registry pattern that allows spectral transformer
components to be registered at runtime and retrieved by name. The registry system
enables modular composition of different spectral transforms, mixing layers, attention
mechanisms, and complete models without tight coupling.

The registry supports metadata storage, configuration-driven instantiation, and
category-based organization. This design facilitates experimentation with different
component combinations and makes the library easily extensible with custom implementations.

Classes
-------
ComponentRegistry
    Central registry for storing and retrieving component classes.

Functions
---------
register_component(category, name, metadata=None)
    Decorator to register component classes in the global registry.
get_component(category, name)
    Retrieve a registered component class by category and name.
create_component(category, name, **kwargs)
    Create an instance of a registered component with parameters.
list_components(category=None)
    List available components in a category or all categories.

Examples
--------
Registering a custom component:

>>> from spectrans.core.registry import register_component
>>> from spectrans.layers.mixing.base import MixingLayer
>>> @register_component('mixing', 'my_custom_mixer')
... class CustomMixer(MixingLayer):
...     def forward(self, x):
...         return x  # Custom implementation

Using the registry to create components:

>>> from spectrans.core.registry import create_component, list_components
>>> # List available transforms
>>> list_components('transform')
['fourier', 'cosine', 'hadamard', 'wavelet']
>>> # Create a Fourier transform instance
>>> fft = create_component('transform', 'fourier', dim=-1)

Configuration-driven component creation:

>>> from spectrans.core.registry import registry
>>> config = {'type': 'fourier', 'params': {'dim': -1}}
>>> transform = registry.create_from_config('transform', config)

Notes
-----
Registry Architecture:

The registry implements several design patterns:
1. **Singleton Pattern**: Global registry instance for system-wide access
2. **Factory Pattern**: Component creation through factory methods
3. **Registry Pattern**: Dynamic component discovery and instantiation
4. **Decorator Pattern**: Clean component registration via decorators

Component Categories:
- 'transform': Spectral transforms (FFT, DCT, DWT, Hadamard)
- 'mixing': Token mixing layers (FourierMixing, GlobalFilter, AFNO)
- 'attention': Spectral attention mechanisms (SpectralAttention, LST)
- 'block': Complete transformer blocks combining mixing + FFN
- 'model': Full model implementations (FNet, GFNet, etc.)
- 'kernel': Kernel functions for attention approximation
- 'operator': Neural operators (FNO layers)

The registry supports metadata storage for each component, enabling rich
component descriptions and configuration schemas.

Thread Safety: The registry is not thread-safe. Component registration should
occur during module initialization, before concurrent access.

See Also
--------
spectrans.core.base : Base classes for components stored in registry
spectrans.core.types : Type definitions for registry operations
"""

from collections.abc import Callable
from typing import Any

from .types import ComponentClass, ComponentType, ConfigDict, RegistryDict


class ComponentRegistry:
    """Registry for dynamically registering and retrieving components.

    This registry allows for flexible component registration and retrieval,
    enabling users to easily extend the library with custom implementations.

    Attributes
    ----------
    _components : RegistryDict
        Dictionary mapping component categories to their registered components.
    _metadata : dict[str, dict[str, dict[str, Any]]]
        Dictionary storing metadata about registered components.
    """

    def __init__(self) -> None:
        self._components: RegistryDict = {
            'transform': {},
            'mixing': {},
            'attention': {},
            'block': {},
            'model': {},
            'kernel': {},
            'operator': {},
        }

        # Store metadata about components
        self._metadata: dict[str, dict[str, dict[str, Any]]] = {
            category: {} for category in self._components
        }

    def register(
        self,
        category: ComponentType,
        name: str,
        component: ComponentClass,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a component.

        Parameters
        ----------
        category : ComponentType
            Category of the component (e.g., 'transform', 'mixing').
        name : str
            Name to register the component under.
        component : ComponentClass
            The component class to register.
        metadata : dict[str, Any] | None, default=None
            Optional metadata about the component.

        Raises
        ------
        ValueError
            If the category is unknown or component name already exists.
        """
        if category not in self._components:
            raise ValueError(
                f"Unknown category: {category}. "
                f"Available categories: {list(self._components.keys())}"
            )

        if name in self._components[category]:
            raise ValueError(
                f"Component '{name}' already registered in category '{category}'"
            )

        self._components[category][name] = component

        if metadata is not None:
            self._metadata[category][name] = metadata

    def get(self, category: ComponentType, name: str) -> ComponentClass:
        """Get a registered component.

        Parameters
        ----------
        category : ComponentType
            Category of the component.
        name : str
            Name of the component.

        Returns
        -------
        ComponentClass
            The registered component class.

        Raises
        ------
        ValueError
            If the category or component name is not found.
        """
        if category not in self._components:
            raise ValueError(
                f"Unknown category: {category}. "
                f"Available categories: {list(self._components.keys())}"
            )

        if name not in self._components[category]:
            available = list(self._components[category].keys())
            raise ValueError(
                f"Unknown {category}: '{name}'. "
                f"Available {category}s: {available}"
            )

        return self._components[category][name]

    def list(self, category: ComponentType | None = None) -> list[str] | dict[str, list[str]]:
        """List registered components.

        Parameters
        ----------
        category : ComponentType | None, default=None
            Category to list components for. If None, lists all categories.

        Returns
        -------
        list[str] | dict[str, list[str]]
            If category is specified, returns list of component names.
            Otherwise, returns dict mapping categories to component names.
        """
        if category is not None:
            if category not in self._components:
                raise ValueError(
                    f"Unknown category: {category}. "
                    f"Available categories: {list(self._components.keys())}"
                )
            return list(self._components[category].keys())

        return {
            cat: list(comps.keys())
            for cat, comps in self._components.items()
        }

    def get_metadata(
        self,
        category: ComponentType,
        name: str,
    ) -> dict[str, Any] | None:
        """Get metadata for a registered component.

        Parameters
        ----------
        category : ComponentType
            Category of the component.
        name : str
            Name of the component.

        Returns
        -------
        dict[str, Any] | None
            Metadata dictionary if available, None otherwise.
        """
        return self._metadata.get(category, {}).get(name)

    def create(
        self,
        category: ComponentType,
        name: str,
        **kwargs: Any,
    ) -> Any:
        """Create an instance of a registered component.

        Parameters
        ----------
        category : ComponentType
            Category of the component.
        name : str
            Name of the component.
        **kwargs : Any
            Keyword arguments to pass to the component constructor.

        Returns
        -------
        Any
            Instance of the component.
        """
        component_class = self.get(category, name)
        return component_class(**kwargs)

    def create_from_config(
        self,
        category: ComponentType,
        config: ConfigDict,
    ) -> Any:
        """Create a component instance from a configuration dictionary.

        Parameters
        ----------
        category : ComponentType
            Category of the component.
        config : ConfigDict
            Configuration dictionary with 'type' and optional 'params' keys.

        Returns
        -------
        Any
            Instance of the component.

        Raises
        ------
        ValueError
            If 'type' key is missing from config.
        """
        if 'type' not in config:
            raise ValueError("Configuration must contain 'type' key")

        name = config['type']
        raw_params: int | float | str | bool | list[int | float | str | bool] | dict[str, int | float | str | bool | list[int | float | str | bool]] = config.get('params', {})
        if isinstance(raw_params, dict):
            params: dict[str, int | float | str | bool | list[int | float | str | bool]] = raw_params
        else:
            params = {}

        return self.create(category, name, **params)  # type: ignore[arg-type]

    def __contains__(self, item: tuple[ComponentType, str]) -> bool:
        """Check if a component is registered.

        Parameters
        ----------
        item : tuple[ComponentType, str]
            Tuple of (category, name) to check.

        Returns
        -------
        bool
            True if the component is registered.
        """
        category, name = item
        return (
            category in self._components and
            name in self._components[category]
        )

    def clear(self, category: ComponentType | None = None) -> None:
        """Clear registered components.

        Parameters
        ----------
        category : ComponentType | None, default=None
            Category to clear. If None, clears all categories.
        """
        if category is not None:
            if category not in self._components:
                raise ValueError(f"Unknown category: {category}")
            self._components[category].clear()
            self._metadata[category].clear()
        else:
            for cat in self._components:
                self._components[cat].clear()
                self._metadata[cat].clear()


# Global registry instance
registry = ComponentRegistry()


def register_component(
    category: ComponentType,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Callable[[ComponentClass], ComponentClass]:
    """Decorator for registering components.

    Parameters
    ----------
    category : ComponentType
        Category to register the component under.
    name : str
        Name to register the component as.
    metadata : dict[str, Any] | None, default=None
        Optional metadata about the component.

    Returns
    -------
    Callable[[ComponentClass], ComponentClass]
        Decorator function.

    Examples
    --------
    >>> @register_component('transform', 'my_fft')
    ... class MyFFT(SpectralTransform):
    ...     pass
    """
    def decorator(cls: ComponentClass) -> ComponentClass:
        registry.register(category, name, cls, metadata)
        return cls
    return decorator


def get_component(category: ComponentType, name: str) -> ComponentClass:
    """Get a registered component class.

    Parameters
    ----------
    category : ComponentType
        Category of the component.
    name : str
        Name of the component.

    Returns
    -------
    ComponentClass
        The registered component class.
    """
    return registry.get(category, name)


def create_component(
    category: ComponentType,
    name: str,
    **kwargs: Any,
) -> Any:
    """Create an instance of a registered component.

    Parameters
    ----------
    category : ComponentType
        Category of the component.
    name : str
        Name of the component.
    **kwargs : Any
        Keyword arguments for the component constructor.

    Returns
    -------
    Any
        Instance of the component.
    """
    return registry.create(category, name, **kwargs)


def list_components(
    category: ComponentType | None = None,
) -> list[str] | dict[str, list[str]]:
    """List available components.

    Parameters
    ----------
    category : ComponentType | None, default=None
        Category to list. If None, lists all categories.

    Returns
    -------
    list[str] | dict[str, list[str]]
        Component names or dict of categories to names.
    """
    return registry.list(category)


# Export public API
__all__: list[str] = [
    'ComponentRegistry',
    'create_component',
    'get_component',
    'list_components',
    'register_component',
    'registry',
]
