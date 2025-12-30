#!/usr/bin/env python3
"""
Minimal test script to verify provider registration without importing core module
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_provider_modules():
    """Test that all provider modules can be imported and registered"""

    try:
        # Test importing provider modules directly
        print("Testing Provider Modules")
        print("=" * 40)

        # Test base provider
        from hypertune.providers.base import BaseProvider

        print("✅ BaseProvider imported successfully")

        # Test registry
        from hypertune.providers.registry import ProviderRegistry

        print("✅ ProviderRegistry imported successfully")

        # Test factory
        from hypertune.providers.factory import ProviderFactory

        print("✅ ProviderFactory imported successfully")

        # Test individual providers
        from hypertune.providers.openai_provider import OpenAIProvider

        print("✅ OpenAIProvider imported successfully")

        from hypertune.providers.anthropic_provider import AnthropicProvider

        print("✅ AnthropicProvider imported successfully")

        from hypertune.providers.gemini_provider import GeminiProvider

        print("✅ GeminiProvider imported successfully")

        from hypertune.providers.openrouter_provider import OpenRouterProvider

        print("✅ OpenRouterProvider imported successfully")

        # Test provider registration
        print("\nTesting Provider Registration")
        print("=" * 40)

        # Check registered providers
        registered = ProviderRegistry.list_providers()
        print(f"Registered providers: {registered}")

        # Verify all expected providers are registered
        expected_providers = ["openai", "anthropic", "gemini", "openrouter"]
        for provider in expected_providers:
            if provider in registered:
                print(f"✅ {provider} is registered")
            else:
                print(f"❌ {provider} is NOT registered")

        print("\n" + "=" * 40)
        print("All provider tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_provider_modules()
    sys.exit(0 if success else 1)
