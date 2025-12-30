#!/usr/bin/env python3
"""Example usage of HyperTune with multiple providers."""

from hypertune.core import HyperTune
from hypertune.providers import ProviderFactory


def test_providers():
    """Test all available providers with a simple prompt."""
    prompt = "Explain the concept of machine learning in simple terms."
    iterations = 3

    print("HyperTune Multi-Provider Example")
    print("=" * 50)
    print(f"Prompt: '{prompt}'")
    print(f"Iterations per provider: {iterations}")
    print()

    for provider_name in ProviderFactory.get_available_providers():
        print(f"Testing {provider_name.upper()} provider:")
        print("-" * 30)

        try:
            ht = HyperTune(prompt, iterations, provider=provider_name)

            provider_info = ProviderFactory.get_provider_info(provider_name)
            print(f"Model: {provider_info['model']}")

            results = ht.run()

            if results:
                best_result = results[0]
                print(f"Best score: {best_result['total_score']:.2f}")
                print(f"Response: {best_result['text'][:200]}...")
                print(f"Hyperparameters: {best_result['hyperparameters']}")
            else:
                print("No results generated")

        except (ValueError, RuntimeError, ImportError) as e:
            print(f"Error with {provider_name}: {e}")

        print()


def list_providers():
    """List all available providers and their models."""
    print("Available LLM Providers:")
    print("=" * 50)

    all_info = ProviderFactory.list_all_provider_info()
    for provider_name, info in all_info.items():
        if "error" in info:
            print(f"\n{provider_name.upper()}: {info['error']}")
            continue

        print(f"\n{provider_name.upper()}:")
        print(f"  Default model: {info['model']}")
        print("  Available models:")
        for model in info["available_models"][:5]:
            print(f"    - {model}")
        if len(info["available_models"]) > 5:
            print(f"    ... and {len(info['available_models']) - 5} more")

        print("  Parameter ranges:")
        for param, ranges in info["parameter_ranges"].items():
            print(f"    {param}: {ranges['min']} - {ranges['max']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_providers()
    else:
        test_providers()
