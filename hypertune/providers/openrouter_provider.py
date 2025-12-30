"""
OpenRouter provider for HyperTune
"""

import os
from typing import Dict, Any, List
import warnings
from .base import BaseProvider
from openai import OpenAI


class OpenRouterProvider(BaseProvider):
    """
    OpenRouter provider - provides access to hundreds of LLM models
    """
    
    def __init__(self, model: str = None):
        """
        Initialize OpenRouter provider
        
        Args:
            model: Model to use (default: anthropic/claude-3.5-sonnet)
        """
        super().__init__(model)
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
    
    def generate(self, prompt: str, **hyperparameters) -> str:
        """
        Generate response using OpenRouter API
        
        Args:
            prompt: Input prompt
            **hyperparameters: Generation parameters
            
        Returns:
            Generated text response
        """
        validated_params = self.validate_hyperparameters(**hyperparameters)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=validated_params.get('temperature', 0.7),
                top_p=validated_params.get('top_p', 0.9),
                max_tokens=validated_params.get('max_tokens', 1024),
                frequency_penalty=validated_params.get('frequency_penalty', 0.0),
                presence_penalty=validated_params.get('presence_penalty', 0.0)
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {str(e)}")
    
    def get_default_model(self) -> str:
        """
        Get default OpenRouter model
        
        Returns:
            Default model name
        """
        return "anthropic/claude-3.5-sonnet"
    
    def get_available_models(self) -> List[str]:
        """
        Get list of popular OpenRouter models
        Note: OpenRouter has hundreds of models, this is a curated list
        
        Returns:
            List of model names
        """
        return [
            # Anthropic models
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            
            # OpenAI models
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-4-turbo",
            "openai/gpt-3.5-turbo",
            
            # Google models
            "google/gemini-pro-1.5",
            "google/gemini-flash-1.5",
            
            # Meta models
            "meta-llama/llama-3.1-405b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            
            # Mistral models
            "mistralai/mistral-large",
            "mistralai/mistral-medium",
            "mistralai/mistral-small",
            
            # Other popular models
            "cohere/command-r-plus",
            "perplexity/llama-3.1-sonar-large-128k-online",
            "qwen/qwen-2.5-72b-instruct"
        ]
    
    def _validate_provider_specific_params(self, **hyperparameters) -> Dict[str, Any]:
        """
        Validate OpenRouter-specific parameters (same as OpenAI)
        
        Args:
            **hyperparameters: Input hyperparameters
            
        Returns:
            Validated OpenRouter-specific parameters
        """
        validated = {}
        
        # frequency_penalty: -2.0 to 2.0
        if 'frequency_penalty' in hyperparameters:
            freq_penalty = hyperparameters['frequency_penalty']
            if not isinstance(freq_penalty, (int, float)) or freq_penalty < -2.0 or freq_penalty > 2.0:
                warnings.warn(f"Invalid frequency_penalty value: {freq_penalty}. Using default 0.0")
                validated['frequency_penalty'] = 0.0
            else:
                validated['frequency_penalty'] = freq_penalty
        
        # presence_penalty: -2.0 to 2.0
        if 'presence_penalty' in hyperparameters:
            pres_penalty = hyperparameters['presence_penalty']
            if not isinstance(pres_penalty, (int, float)) or pres_penalty < -2.0 or pres_penalty > 2.0:
                warnings.warn(f"Invalid presence_penalty value: {pres_penalty}. Using default 0.0")
                validated['presence_penalty'] = 0.0
            else:
                validated['presence_penalty'] = pres_penalty
        
        return validated
    
    def get_parameter_ranges(self) -> Dict[str, Dict[str, float]]:
        """
        Get parameter ranges for OpenRouter
        
        Returns:
            Dictionary with parameter ranges
        """
        ranges = super().get_parameter_ranges()
        ranges.update({
            'frequency_penalty': {'min': -2.0, 'max': 2.0},
            'presence_penalty': {'min': -2.0, 'max': 2.0}
        })
        return ranges