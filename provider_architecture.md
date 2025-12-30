# Provider-Agnostic Architecture for HyperTune

## Overview
To support multiple LLM providers (OpenAI, Anthropic Claude, Google Gemini, OpenRouter), we'll implement a provider pattern that abstracts the differences between APIs while maintaining the existing functionality.

## Architecture Components

### 1. Base Provider Interface (`hypertune/providers/base.py`)
Abstract base class that defines the common interface for all providers:
- `generate(prompt, **hyperparameters)`: Generate text response
- `get_default_model()`: Get the default model for the provider
- `get_available_models()`: Get list of available models
- `validate_hyperparameters(**hyperparameters)`: Validate provider-specific parameters

### 2. Provider Implementations (`hypertune/providers/`)
- `openai_provider.py`: OpenAI GPT provider
- `anthropic_provider.py`: Anthropic Claude provider  
- `gemini_provider.py`: Google Gemini provider
- `openrouter_provider.py`: OpenRouter provider

### 3. Provider Factory (`hypertune/providers/factory.py`)
Factory class to create provider instances:
- `create_provider(provider_name, model=None)`: Create provider instance
- `get_available_providers()`: List all available providers

### 4. Provider Registry (`hypertune/providers/registry.py`)
Registry to manage available providers and their configurations:
- `register_provider(name, provider_class)`: Register a new provider
- `get_provider_class(name)`: Get provider class by name
- `list_providers()`: List all registered providers

### 5. Updated Core (`hypertune/core.py`)
Modified HyperTune class to use providers:
- Accept provider name and model in constructor
- Use provider factory to create appropriate provider
- Pass hyperparameters to provider's generate method

## Parameter Mapping Strategy

Since different providers support different hyperparameters, we'll implement a mapping system:

### Common Parameters (supported by all providers)
- `temperature`: Controls randomness (0.0 to 1.0)
- `top_p`: Nucleus sampling (0.0 to 1.0)
- `max_tokens`: Maximum response length

### Provider-Specific Parameters
- OpenAI: `frequency_penalty`, `presence_penalty`
- Google Gemini: `top_k`
- Anthropic: No additional penalties
- OpenRouter: Same as OpenAI (compatible API)

### Parameter Handling
1. Accept all possible parameters in the interface
2. Each provider validates and maps parameters to its API
3. Ignore unsupported parameters with warnings
4. Apply default values for missing parameters

## Implementation Flow

1. User specifies provider and optional model via CLI
2. HyperTune creates provider using factory
3. For each iteration:
   - Generate random hyperparameters
   - Provider validates and maps parameters
   - Provider calls its specific API
   - Response is standardized and returned
4. Scoring remains unchanged (works on text output)

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `GOOGLE_API_KEY`: Google API key
- `OPENROUTER_API_KEY`: OpenRouter API key

### Provider Configuration
Each provider will have default models and parameter ranges:
- OpenAI: "gpt-4o", temperature: 0.1-1.0, top_p: 0.1-1.0
- Anthropic: "claude-3-5-sonnet-20241022", temperature: 0.0-1.0, top_p: 0.0-1.0
- Gemini: "gemini-1.5-pro", temperature: 0.0-2.0, top_p: 0.0-1.0
- OpenRouter: "anthropic/claude-3.5-sonnet", same as OpenAI

## Benefits of This Architecture

1. **Extensibility**: Easy to add new providers by implementing the base interface
2. **Consistency**: Uniform interface across all providers
3. **Flexibility**: Provider-specific parameters can be utilized when available
4. **Maintainability**: Each provider is isolated in its own module
5. **Backward Compatibility**: Existing OpenAI functionality remains unchanged

## File Structure
```
hypertune/
├── providers/
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── registry.py
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── gemini_provider.py
│   └── openrouter_provider.py
├── core.py (modified)
└── ...