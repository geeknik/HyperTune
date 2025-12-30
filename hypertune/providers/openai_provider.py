"""
OpenAI provider for HyperTune
"""

import os
from typing import Dict, Any, List
import warnings
from .base import BaseProvider
from openai import OpenAI


class OpenAIProvider(BaseProvider):
    """
    OpenAI GPT provider
    """
    
    def __init__(self, model: str = None):
        """
        Initialize OpenAI provider
        
        Args:
            model: Model to use (default: gpt-4o)
        """
        super().__init__(model)
        self.client = OpenAI()
    
    def generate(self, prompt: str, **hyperparameters) -> str:
        """
        Generate response using OpenAI API
        
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
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    def get_default_model(self) -> str:
        """
        Get default OpenAI model
        
        Returns:
            Default model name
        """
        return "gpt-4o"
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available OpenAI models
        
        Returns:
            List of model names
        """
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ]
    
    def _validate_provider_specific_params(self, **hyperparameters) -> Dict[str, Any]:
        """
        Validate OpenAI-specific parameters
        
        Args:
            **hyperparameters: Input hyperparameters
            
        Returns:
            Validated OpenAI-specific parameters
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
        Get parameter ranges for OpenAI
        
        Returns:
            Dictionary with parameter ranges
        """
        ranges = super().get_parameter_ranges()
        ranges.update({
            'frequency_penalty': {'min': -2.0, 'max': 2.0},
            'presence_penalty': {'min': -2.0, 'max': 2.0}
        })
        return ranges