"""
Provider module for HyperTune - supports multiple LLM providers
"""

from .factory import ProviderFactory
from .registry import ProviderRegistry
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .openrouter_provider import OpenRouterProvider

# Register all available providers
ProviderRegistry.register('openai', OpenAIProvider)
ProviderRegistry.register('anthropic', AnthropicProvider)
ProviderRegistry.register('gemini', GeminiProvider)
ProviderRegistry.register('openrouter', OpenRouterProvider)

__all__ = [
    'ProviderFactory',
    'ProviderRegistry',
    'OpenAIProvider',
    'AnthropicProvider',
    'GeminiProvider',
    'OpenRouterProvider'
]