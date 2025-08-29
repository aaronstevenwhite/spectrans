"""Component registry system for dynamic component registration and retrieval."""

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
    _metadata : dict
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
        params = config.get('params', {})

        return self.create(category, name, **params)

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
