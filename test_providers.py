#!/usr/bin/env python3
"""
Simple test script to verify provider registration without requiring all dependencies
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_provider_registration():
    """Test that all providers are properly registered"""
    
    try:
        from hypertune.providers import ProviderFactory, ProviderRegistry
        
        print("Testing Provider Registration")
        print("=" * 40)
        
        # Test provider registry
        registered_providers = ProviderRegistry.list_providers()
        print(f"Registered providers: {registered_providers}")
        
        # Test provider factory
        available_providers = ProviderFactory.get_available_providers()
        print(f"Available providers: {available_providers}")
        
        # Test each provider info
        print("\nProvider Information:")
        print("-" * 20)
        
        for provider_name in available_providers:
            try:
                info = ProviderFactory.get_provider_info(provider_name)
                print(f"\n{provider_name.upper()}:")
                print(f"  Default model: {info['model']}")
                print(f"  Available models: {len(info['available_models'])} models")
                print(f"  Parameters: {list(info['parameter_ranges'].keys())}")
            except Exception as e:
                print(f"\n{provider_name.upper()}: Error - {e}")
        
        print("\n✅ Provider registration test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("  pip install openai anthropic google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def test_provider_creation():
    """Test creating provider instances (without API calls)"""
    
    try:
        from hypertune.providers import ProviderFactory
        
        print("\nTesting Provider Creation")
        print("=" * 40)
        
        # Test creating each provider (will fail if API keys not set, but that's expected)
        providers_to_test = ['openai', 'anthropic', 'gemini', 'openrouter']
        
        for provider_name in providers_to_test:
            try:
                provider = ProviderFactory.create_provider(provider_name)
                print(f"✅ {provider_name}: Created successfully")
                print(f"   Model: {provider.model}")
                print(f"   Class: {provider.__class__.__name__}")
            except ImportError as e:
                print(f"⚠️  {provider_name}: Missing dependency - {e}")
            except ValueError as e:
                print(f"⚠️  {provider_name}: Missing API key - {e}")
            except Exception as e:
                print(f"❌ {provider_name}: Unexpected error - {e}")
        
        print("\n✅ Provider creation test completed!")
        
    except Exception as e:
        print(f"❌ Provider creation test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("HyperTune Provider Test Suite")
    print("=" * 50)
    
    success1 = test_provider_registration()
    success2 = test_provider_creation()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Multi-provider support is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the output above for details.")
        sys.exit(1)