"""
Google Gemini provider for HyperTune
"""

import os
from typing import Dict, Any, List
import warnings
from .base import BaseProvider

try:
    import google.generativeai as genai

    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False


class GeminiProvider(BaseProvider):
    """
    Google Gemini provider
    """

    def __init__(self, model: str = None):
        """
        Initialize Google Gemini provider

        Args:
            model: Model to use (default: gemini-1.5-pro)
        """
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError(
                "google-generativeai package is required. Install with: pip install google-generativeai"
            )

        super().__init__(model)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model)

    def generate(self, prompt: str, **hyperparameters) -> str:
        """
        Generate response using Google Gemini API

        Args:
            prompt: Input prompt
            **hyperparameters: Generation parameters

        Returns:
            Generated text response
        """
        validated_params = self.validate_hyperparameters(**hyperparameters)

        try:
            # Create generation config
            generation_config = genai.types.GenerationConfig(
                temperature=validated_params.get("temperature", 0.7),
                top_p=validated_params.get("top_p", 0.9),
                top_k=validated_params.get("top_k", 40),
                max_output_tokens=validated_params.get("max_tokens", 1024),
            )

            response = self.model.generate_content(
                prompt, generation_config=generation_config
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Google Gemini API error: {str(e)}")

    def get_default_model(self) -> str:
        """
        Get default Google Gemini model

        Returns:
            Default model name
        """
        return "gemini-2.5-pro"

    def get_available_models(self) -> List[str]:
        """
        Get list of available Google Gemini models

        Returns:
            List of model names
        """
        return [
            # Gemini 3 family
            "gemini-3-pro",
            "gemini-3-flash",
            # Gemini 2.5 family
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            # Gemini 1.5 family (legacy)
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]

    def _validate_provider_specific_params(self, **hyperparameters) -> Dict[str, Any]:
        """
        Validate Google Gemini-specific parameters

        Args:
            **hyperparameters: Input hyperparameters

        Returns:
            Validated Gemini-specific parameters
        """
        validated = {}

        # top_k: 1 to 40
        if "top_k" in hyperparameters:
            top_k = hyperparameters["top_k"]
            if not isinstance(top_k, int) or top_k < 1 or top_k > 40:
                warnings.warn(f"Invalid top_k value: {top_k}. Using default 40")
                validated["top_k"] = 40
            else:
                validated["top_k"] = top_k

        return validated

    def get_parameter_ranges(self) -> Dict[str, Dict[str, float]]:
        """
        Get parameter ranges for Google Gemini

        Returns:
            Dictionary with parameter ranges
        """
        ranges = super().get_parameter_ranges()
        # Gemini temperature range is 0.0 to 2.0
        ranges["temperature"] = {"min": 0.0, "max": 2.0}
        ranges["top_k"] = {"min": 1, "max": 40}
        return ranges
