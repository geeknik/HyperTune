from unittest.mock import MagicMock, patch

import pytest


class TestProviderBranchCoverage:
    @patch("hypertune.providers.openai_provider.OpenAI")
    def test_openai_presence_penalty_validation_paths(self, _mock_openai):
        from hypertune.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider()

        valid = provider._validate_provider_specific_params(presence_penalty=1.5)
        assert valid["presence_penalty"] == 1.5

        invalid = provider._validate_provider_specific_params(presence_penalty=3.0)
        assert invalid["presence_penalty"] == 0.0

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_openrouter_non_temperature_error_is_wrapped(self, mock_openai):
        from hypertune.providers.openrouter_provider import OpenRouterProvider

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("network failed")
        mock_openai.return_value = mock_client

        provider = OpenRouterProvider()
        with pytest.raises(RuntimeError, match="OpenRouter API error"):
            provider.generate("test prompt")

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_openrouter_frequency_and_presence_validation_paths(self, _mock_openai):
        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()

        valid = provider._validate_provider_specific_params(
            frequency_penalty=0.5, presence_penalty=0.25
        )
        assert valid["frequency_penalty"] == 0.5
        assert valid["presence_penalty"] == 0.25

        invalid = provider._validate_provider_specific_params(
            frequency_penalty=5.0, presence_penalty=-3.0
        )
        assert invalid["frequency_penalty"] == 0.0
        assert invalid["presence_penalty"] == 0.0

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("hypertune.providers.anthropic_provider.Anthropic")
    def test_anthropic_generate_error_is_wrapped(self, mock_anthropic):
        from hypertune.providers.anthropic_provider import AnthropicProvider

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("provider error")
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider()
        with pytest.raises(RuntimeError, match="Anthropic API error"):
            provider.generate("test")

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("hypertune.providers.anthropic_provider.Anthropic")
    def test_anthropic_parameter_ranges_override_temperature(self, _mock_anthropic):
        from hypertune.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()
        ranges = provider.get_parameter_ranges()
        assert ranges["temperature"] == {"min": 0.0, "max": 1.0}

    def test_anthropic_import_error_when_package_unavailable(self):
        from hypertune.providers import anthropic_provider

        with patch.object(anthropic_provider, "ANTHROPIC_AVAILABLE", False):
            with pytest.raises(ImportError, match="anthropic package is required"):
                anthropic_provider.AnthropicProvider()

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
    @patch("hypertune.providers.gemini_provider.types")
    @patch("hypertune.providers.gemini_provider.genai")
    def test_gemini_generate_success_path(self, mock_genai, mock_types):
        from hypertune.providers import gemini_provider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Gemini response"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client
        mock_types.GenerateContentConfig.return_value = MagicMock()

        with patch.object(gemini_provider, "GOOGLE_AI_AVAILABLE", True):
            provider = gemini_provider.GeminiProvider()
            result = provider.generate("test", temperature=0.4, top_k=10, max_tokens=128)

        assert result == "Gemini response"
        mock_types.GenerateContentConfig.assert_called_once()
        mock_client.models.generate_content.assert_called_once()

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
    @patch("hypertune.providers.gemini_provider.types")
    @patch("hypertune.providers.gemini_provider.genai")
    def test_gemini_generate_error_is_wrapped(self, mock_genai, mock_types):
        from hypertune.providers import gemini_provider

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("provider error")
        mock_genai.Client.return_value = mock_client
        mock_types.GenerateContentConfig.return_value = MagicMock()

        with patch.object(gemini_provider, "GOOGLE_AI_AVAILABLE", True):
            provider = gemini_provider.GeminiProvider()
            with pytest.raises(RuntimeError, match="Google Gemini API error"):
                provider.generate("test")

    def test_gemini_import_error_when_package_unavailable(self):
        from hypertune.providers import gemini_provider

        with patch.object(gemini_provider, "GOOGLE_AI_AVAILABLE", False):
            with pytest.raises(ImportError, match="google-genai package is required"):
                gemini_provider.GeminiProvider()

    def test_gemini_missing_api_key_raises(self):
        from hypertune.providers import gemini_provider

        with patch.object(gemini_provider, "GOOGLE_AI_AVAILABLE", True):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                    gemini_provider.GeminiProvider()
