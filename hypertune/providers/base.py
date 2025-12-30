"""
Base provider interface for all LLM providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import warnings


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialize the provider with a specific model

        Args:
            model: The model to use. If None, uses the provider's default model.
        """
        self.model = model or self.get_default_model()

    @abstractmethod
    def generate(self, prompt: str, **hyperparameters) -> str:
        """
        Generate a response for the given prompt

        Args:
            prompt: The input prompt
            **hyperparameters: Provider-specific hyperparameters

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """
        Get the default model for this provider

        Returns:
            Default model name
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider

        Returns:
            List of model names
        """
        pass

    def validate_hyperparameters(self, **hyperparameters) -> Dict[str, Any]:
        """
        Validate and normalize hyperparameters for this provider

        Args:
            **hyperparameters: Input hyperparameters

        Returns:
            Validated and normalized hyperparameters
        """
        validated = {}

        # Common parameters
        if 'temperature' in hyperparameters:
            temp = hyperparameters['temperature']
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                warnings.warn(f"Invalid temperature value: {temp}. Using default 0.7")
                validated['temperature'] = 0.7
            else:
                validated['temperature'] = temp

        if 'top_p' in hyperparameters:
            top_p = hyperparameters['top_p']
            if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
                warnings.warn(f"Invalid top_p value: {top_p}. Using default 0.9")
                validated['top_p'] = 0.9
            else:
                validated['top_p'] = top_p

        if 'max_tokens' in hyperparameters:
            max_tokens = hyperparameters['max_tokens']
            if not isinstance(max_tokens, int) or max_tokens < 1:
                warnings.warn(f"Invalid max_tokens value: {max_tokens}. Using default 1024")
                validated['max_tokens'] = 1024
            else:
                validated['max_tokens'] = max_tokens

        # Provider-specific validation
        validated.update(self._validate_provider_specific_params(**hyperparameters))

        return validated

    def _validate_provider_specific_params(self, **hyperparameters) -> Dict[str, Any]:
        """
        Validate provider-specific hyperparameters
        Override in subclasses as needed

        Args:
            **hyperparameters: Input hyperparameters

        Returns:
            Dictionary of validated provider-specific parameters
        """
        return {}

    def get_parameter_ranges(self) -> Dict[str, Dict[str, float]]:
        """
        Get the valid ranges for hyperparameters

        Returns:
            Dictionary with parameter names and their min/max values
        """
        return {
            'temperature': {'min': 0.0, 'max': 2.0},
            'top_p': {'min': 0.0, 'max': 1.0},
            'max_tokens': {'min': 1, 'max': 8192}
        }

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider

        Returns:
            Dictionary with provider information
        """
        return {
            'name': self.__class__.__name__,
            'model': self.model,
            'available_models': self.get_available_models(),
            'parameter_ranges': self.get_parameter_ranges()
        }
