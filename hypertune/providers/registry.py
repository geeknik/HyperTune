"""
Registry for managing LLM providers
"""

from typing import Dict, Type, List
from .base import BaseProvider


class ProviderRegistry:
    """
    Registry for managing available LLM providers
    """
    
    _providers: Dict[str, Type[BaseProvider]] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """
        Register a provider class
        
        Args:
            name: Provider name
            provider_class: Provider class that inherits from BaseProvider
        """
        if not issubclass(provider_class, BaseProvider):
            raise ValueError(f"Provider class {provider_class} must inherit from BaseProvider")
        
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def get_provider_class(cls, name: str) -> Type[BaseProvider]:
        """
        Get a provider class by name
        
        Args:
            name: Provider name
            
        Returns:
            Provider class
            
        Raises:
            ValueError: If provider is not registered
        """
        provider_name = name.lower()
        if provider_name not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(f"Provider '{name}' not found. Available providers: {available}")
        
        return cls._providers[provider_name]
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered provider names
        
        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a provider is registered
        
        Args:
            name: Provider name
            
        Returns:
            True if provider is registered, False otherwise
        """
        return name.lower() in cls._providers
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a provider
        
        Args:
            name: Provider name
            
        Raises:
            ValueError: If provider is not registered
        """
        provider_name = name.lower()
        if provider_name not in cls._providers:
            raise ValueError(f"Provider '{name}' is not registered")
        
        del cls._providers[provider_name]
    
    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered providers
        """
        cls._providers.clear()