"""
OpenRouter provider for HyperTune
"""

import os
import re
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
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/geeknik/HyperTune",
                "X-Title": "HyperTune",
            },
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

        extra_body = {}
        if "top_k" in validated_params:
            extra_body["top_k"] = validated_params["top_k"]
        if "repetition_penalty" in validated_params:
            extra_body["repetition_penalty"] = validated_params["repetition_penalty"]
        if "min_p" in validated_params:
            extra_body["min_p"] = validated_params["min_p"]
        if "top_a" in validated_params:
            extra_body["top_a"] = validated_params["top_a"]

        temperature = validated_params.get("temperature", 0.7)

        return self._call_api(prompt, validated_params, extra_body, temperature)

    def _call_api(
        self,
        prompt: str,
        validated_params: Dict[str, Any],
        extra_body: Dict[str, Any],
        temperature: float,
        retry: bool = True,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=validated_params.get("top_p", 0.9),
                max_tokens=validated_params.get("max_tokens", 1024),
                frequency_penalty=validated_params.get("frequency_penalty", 0.0),
                presence_penalty=validated_params.get("presence_penalty", 0.0),
                extra_body=extra_body if extra_body else None,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if retry and "temperature must be within" in error_str:
                match = re.search(
                    r"temperature must be within \[[\d.]+,\s*([\d.]+)\]", error_str
                )
                if match:
                    max_temp = float(match.group(1))
                    clamped_temp = min(temperature, max_temp)
                    warnings.warn(
                        f"Provider temperature limit exceeded. Clamping {temperature} to {clamped_temp}"
                    )
                    return self._call_api(
                        prompt, validated_params, extra_body, clamped_temp, retry=False
                    )
            raise RuntimeError(f"OpenRouter API error: {error_str}")

    def get_default_model(self) -> str:
        """
        Get default OpenRouter model

        Returns:
            Default model name
        """
        return "anthropic/claude-sonnet-4.5"

    def get_available_models(self) -> List[str]:
        """
        Get list of popular OpenRouter models
        Note: OpenRouter has hundreds of models, this is a curated list

        Returns:
            List of model names
        """
        return [
            # Anthropic models
            "anthropic/claude-opus-4.5",
            "anthropic/claude-sonnet-4.5",
            "anthropic/claude-haiku-4.5",
            "anthropic/claude-3.5-sonnet",
            # OpenAI models
            "openai/gpt-5.2",
            "openai/gpt-5.2-pro",
            "openai/gpt-5",
            "openai/gpt-5-mini",
            "openai/gpt-5-nano",
            "openai/gpt-4.1",
            "openai/gpt-4o",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            # Google models
            "google/gemini-3-pro",
            "google/gemini-3-flash",
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
            # Meta models
            "meta-llama/llama-4-405b-instruct",
            "meta-llama/llama-4-70b-instruct",
            "meta-llama/llama-3.1-405b-instruct",
            # Mistral models
            "mistralai/mistral-large-2",
            "mistralai/mistral-medium-2",
            "mistralai/codestral",
            # Other popular models
            "cohere/command-r-plus-2",
            "perplexity/sonar-pro",
            "qwen/qwen-3-72b-instruct",
            "deepseek/deepseek-r1",
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
        if "frequency_penalty" in hyperparameters:
            freq_penalty = hyperparameters["frequency_penalty"]
            if (
                not isinstance(freq_penalty, (int, float))
                or freq_penalty < -2.0
                or freq_penalty > 2.0
            ):
                warnings.warn(
                    f"Invalid frequency_penalty value: {freq_penalty}. Using default 0.0"
                )
                validated["frequency_penalty"] = 0.0
            else:
                validated["frequency_penalty"] = freq_penalty

        # presence_penalty: -2.0 to 2.0
        if "presence_penalty" in hyperparameters:
            pres_penalty = hyperparameters["presence_penalty"]
            if (
                not isinstance(pres_penalty, (int, float))
                or pres_penalty < -2.0
                or pres_penalty > 2.0
            ):
                warnings.warn(
                    f"Invalid presence_penalty value: {pres_penalty}. Using default 0.0"
                )
                validated["presence_penalty"] = 0.0
            else:
                validated["presence_penalty"] = pres_penalty

        # top_k: 0 or above (0 disables)
        if "top_k" in hyperparameters:
            top_k = hyperparameters["top_k"]
            if not isinstance(top_k, int) or top_k < 0:
                warnings.warn(f"Invalid top_k value: {top_k}. Skipping parameter.")
            else:
                validated["top_k"] = top_k

        # repetition_penalty: 0.0 to 2.0
        if "repetition_penalty" in hyperparameters:
            rep_penalty = hyperparameters["repetition_penalty"]
            if (
                not isinstance(rep_penalty, (int, float))
                or rep_penalty < 0.0
                or rep_penalty > 2.0
            ):
                warnings.warn(
                    f"Invalid repetition_penalty value: {rep_penalty}. Skipping parameter."
                )
            else:
                validated["repetition_penalty"] = rep_penalty

        # min_p: 0.0 to 1.0
        if "min_p" in hyperparameters:
            min_p = hyperparameters["min_p"]
            if not isinstance(min_p, (int, float)) or min_p < 0.0 or min_p > 1.0:
                warnings.warn(f"Invalid min_p value: {min_p}. Skipping parameter.")
            else:
                validated["min_p"] = min_p

        # top_a: 0.0 to 1.0
        if "top_a" in hyperparameters:
            top_a = hyperparameters["top_a"]
            if not isinstance(top_a, (int, float)) or top_a < 0.0 or top_a > 1.0:
                warnings.warn(f"Invalid top_a value: {top_a}. Skipping parameter.")
            else:
                validated["top_a"] = top_a

        return validated

    def get_parameter_ranges(self) -> Dict[str, Dict[str, float]]:
        ranges = super().get_parameter_ranges()
        ranges.update(
            {
                "frequency_penalty": {"min": -2.0, "max": 2.0},
                "presence_penalty": {"min": -2.0, "max": 2.0},
                "top_k": {"min": 0, "max": 100},
                "repetition_penalty": {"min": 0.0, "max": 2.0},
                "min_p": {"min": 0.0, "max": 1.0},
                "top_a": {"min": 0.0, "max": 1.0},
            }
        )
        return ranges
