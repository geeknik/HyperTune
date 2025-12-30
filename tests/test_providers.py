import os
import pytest
from unittest.mock import patch, MagicMock
import warnings


class TestOpenAIProvider:
    @patch("hypertune.providers.openai_provider.OpenAI")
    def test_init_default_model(self, mock_openai):
        from hypertune.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider()
        assert provider.model == "gpt-4o"

    @patch("hypertune.providers.openai_provider.OpenAI")
    def test_init_custom_model(self, mock_openai):
        from hypertune.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(model="gpt-3.5-turbo")
        assert provider.model == "gpt-3.5-turbo"

    @patch("hypertune.providers.openai_provider.OpenAI")
    def test_generate(self, mock_openai):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        from hypertune.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider()
        result = provider.generate("Test prompt", temperature=0.7)

        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch("hypertune.providers.openai_provider.OpenAI")
    def test_generate_api_error(self, mock_openai):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        from hypertune.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider()

        with pytest.raises(RuntimeError, match="OpenAI API error"):
            provider.generate("Test prompt")

    @patch("hypertune.providers.openai_provider.OpenAI")
    def test_get_available_models(self, mock_openai):
        from hypertune.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider()
        models = provider.get_available_models()
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "gpt-3.5-turbo" in models

    @patch("hypertune.providers.openai_provider.OpenAI")
    def test_get_parameter_ranges(self, mock_openai):
        from hypertune.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider()
        ranges = provider.get_parameter_ranges()
        assert "frequency_penalty" in ranges
        assert "presence_penalty" in ranges
        assert ranges["frequency_penalty"]["min"] == -2.0
        assert ranges["frequency_penalty"]["max"] == 2.0

    @patch("hypertune.providers.openai_provider.OpenAI")
    def test_validate_frequency_penalty_valid(self, mock_openai):
        from hypertune.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider()
        result = provider._validate_provider_specific_params(frequency_penalty=1.0)
        assert result["frequency_penalty"] == 1.0

    @patch("hypertune.providers.openai_provider.OpenAI")
    def test_validate_frequency_penalty_invalid(self, mock_openai):
        from hypertune.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider._validate_provider_specific_params(frequency_penalty=5.0)
            assert result["frequency_penalty"] == 0.0


class TestOpenRouterProvider:
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_init_default_model(self, mock_openai):
        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        assert provider.model == "anthropic/claude-3.5-sonnet"

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_get_available_models(self, mock_openai):
        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        models = provider.get_available_models()
        assert len(models) > 0
        assert any("openai" in m for m in models)

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_generate(self, mock_openai):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OpenRouter response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        result = provider.generate("Test prompt")

        assert result == "OpenRouter response"

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            from hypertune.providers.openrouter_provider import OpenRouterProvider

            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                OpenRouterProvider()


class TestAnthropicProvider:
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("hypertune.providers.anthropic_provider.Anthropic")
    def test_init_default_model(self, mock_anthropic):
        from hypertune.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()
        assert provider.model == "claude-3-5-sonnet-20241022"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("hypertune.providers.anthropic_provider.Anthropic")
    def test_get_available_models(self, mock_anthropic):
        from hypertune.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()
        models = provider.get_available_models()
        assert any("claude" in m for m in models)

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("hypertune.providers.anthropic_provider.Anthropic")
    def test_generate(self, mock_anthropic):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Anthropic response"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        from hypertune.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()
        result = provider.generate("Test prompt")

        assert result == "Anthropic response"

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            from hypertune.providers.anthropic_provider import AnthropicProvider

            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AnthropicProvider()


class TestGeminiProvider:
    @pytest.fixture(autouse=True)
    def setup_gemini_mock(self):
        self.mock_genai = MagicMock()
        self.mock_genai.GenerativeModel.return_value = MagicMock()
        self.mock_genai.types.GenerationConfig.return_value = MagicMock()

    def test_get_default_model(self):
        from hypertune.providers.gemini_provider import GeminiProvider

        assert GeminiProvider.get_default_model(None) == "gemini-1.5-pro"

    def test_get_available_models_static(self):
        from hypertune.providers.gemini_provider import GeminiProvider

        provider_class = GeminiProvider
        instance = MagicMock(spec=GeminiProvider)
        models = GeminiProvider.get_available_models(instance)
        assert any("gemini" in m for m in models)

    def test_get_parameter_ranges_includes_top_k(self):
        from hypertune.providers.gemini_provider import GeminiProvider

        instance = MagicMock(spec=GeminiProvider)
        instance.get_parameter_ranges = GeminiProvider.get_parameter_ranges
        ranges = GeminiProvider.get_parameter_ranges(instance)
        assert "top_k" in ranges

    def test_validate_top_k_valid(self):
        from hypertune.providers.gemini_provider import GeminiProvider

        instance = MagicMock(spec=GeminiProvider)
        result = GeminiProvider._validate_provider_specific_params(instance, top_k=20)
        assert result["top_k"] == 20

    def test_validate_top_k_invalid(self):
        from hypertune.providers.gemini_provider import GeminiProvider

        instance = MagicMock(spec=GeminiProvider)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = GeminiProvider._validate_provider_specific_params(
                instance, top_k=100
            )
            assert result["top_k"] == 40
