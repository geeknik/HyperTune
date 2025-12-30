"""
Factory for creating LLM provider instances
"""

from typing import Optional, List
from .base import BaseProvider
from .registry import ProviderRegistry


class ProviderFactory:
    """
    Factory class for creating LLM provider instances
    """
    
    @staticmethod
    def create_provider(provider_name: str, model: Optional[str] = None) -> BaseProvider:
        """
        Create a provider instance
        
        Args:
            provider_name: Name of the provider to create
            model: Optional model name to use with the provider
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider is not registered
        """
        provider_class = ProviderRegistry.get_provider_class(provider_name)
        return provider_class(model=model)
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """
        Get list of available provider names
        
        Returns:
            List of provider names
        """
        return ProviderRegistry.list_providers()
    
    @staticmethod
    def get_provider_info(provider_name: str) -> dict:
        """
        Get information about a specific provider
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Dictionary with provider information
            
        Raises:
            ValueError: If provider is not registered
        """
        provider = ProviderFactory.create_provider(provider_name)
        return provider.get_provider_info()
    
    @staticmethod
    def list_all_provider_info() -> dict:
        """
        Get information about all registered providers
        
        Returns:
            Dictionary with all provider information
        """
        all_info = {}
        for provider_name in ProviderRegistry.list_providers():
            try:
                all_info[provider_name] = ProviderFactory.get_provider_info(provider_name)
            except Exception as e:
                all_info[provider_name] = {'error': str(e)}
        
        return all_info