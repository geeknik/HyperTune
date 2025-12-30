"""
Anthropic Claude provider for HyperTune
"""

import os
from typing import Dict, Any, List
import warnings
from .base import BaseProvider

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude provider
    """

    def __init__(self, model: str = None):
        """
        Initialize Anthropic provider

        Args:
            model: Model to use (default: claude-3-5-sonnet-20241022)
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            )

        super().__init__(model)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        self.client = Anthropic(api_key=api_key)

    def generate(self, prompt: str, **hyperparameters) -> str:
        """
        Generate response using Anthropic Claude API

        Args:
            prompt: Input prompt
            **hyperparameters: Generation parameters

        Returns:
            Generated text response
        """
        validated_params = self.validate_hyperparameters(**hyperparameters)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=validated_params.get("max_tokens", 1024),
                temperature=validated_params.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")

    def get_default_model(self) -> str:
        """
        Get default Anthropic model

        Returns:
            Default model name
        """
        return "claude-sonnet-4.5"

    def get_available_models(self) -> List[str]:
        """
        Get list of available Anthropic models

        Returns:
            List of model names
        """
        return [
            # Claude 4.5 family
            "claude-opus-4.5",
            "claude-sonnet-4.5",
            "claude-haiku-4.5",
            # Claude 3.5 family (legacy)
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            # Claude 3 family (legacy)
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

    def get_parameter_ranges(self) -> Dict[str, Dict[str, float]]:
        """
        Get parameter ranges for Anthropic Claude

        Returns:
            Dictionary with parameter ranges
        """
        ranges = super().get_parameter_ranges()
        # Anthropic temperature range is 0.0 to 1.0
        ranges["temperature"] = {"min": 0.0, "max": 1.0}
        return ranges
