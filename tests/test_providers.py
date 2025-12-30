import os
import pytest
from unittest.mock import patch, MagicMock
import warnings


class TestOpenAIProvider:
    @patch("hypertune.providers.openai_provider.OpenAI")
    def test_init_default_model(self, mock_openai):
        from hypertune.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider()
        assert provider.model == "gpt-5"

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
        assert "gpt-5" in models
        assert "gpt-5.2" in models
        assert "gpt-4o" in models

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
        assert provider.model == "anthropic/claude-sonnet-4.5"

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

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_get_parameter_ranges_includes_new_params(self, mock_openai):
        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        ranges = provider.get_parameter_ranges()
        assert "top_k" in ranges
        assert "repetition_penalty" in ranges
        assert "min_p" in ranges
        assert "top_a" in ranges

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_validate_top_k_valid(self, mock_openai):
        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        result = provider._validate_provider_specific_params(top_k=50)
        assert result["top_k"] == 50

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_validate_top_k_invalid(self, mock_openai):
        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider._validate_provider_specific_params(top_k=-5)
            assert "top_k" not in result

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_validate_repetition_penalty_valid(self, mock_openai):
        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        result = provider._validate_provider_specific_params(repetition_penalty=1.5)
        assert result["repetition_penalty"] == 1.5

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_validate_repetition_penalty_invalid(self, mock_openai):
        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider._validate_provider_specific_params(repetition_penalty=3.0)
            assert "repetition_penalty" not in result

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_validate_min_p_valid(self, mock_openai):
        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        result = provider._validate_provider_specific_params(min_p=0.1)
        assert result["min_p"] == 0.1

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_validate_min_p_invalid(self, mock_openai):
        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider._validate_provider_specific_params(min_p=1.5)
            assert "min_p" not in result

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_validate_top_a_valid(self, mock_openai):
        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        result = provider._validate_provider_specific_params(top_a=0.5)
        assert result["top_a"] == 0.5

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_validate_top_a_invalid(self, mock_openai):
        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider._validate_provider_specific_params(top_a=-0.1)
            assert "top_a" not in result

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_generate_with_extra_params(self, mock_openai):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        provider.generate(
            "Test", top_k=50, min_p=0.1, repetition_penalty=1.2, top_a=0.3
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"]["top_k"] == 50
        assert call_kwargs["extra_body"]["min_p"] == 0.1
        assert call_kwargs["extra_body"]["repetition_penalty"] == 1.2
        assert call_kwargs["extra_body"]["top_a"] == 0.3

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    @patch("hypertune.providers.openrouter_provider.OpenAI")
    def test_temperature_clamping_on_provider_error(self, mock_openai):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response after retry"

        error_msg = (
            'Error code: 400 - {"error":{"message":"Provider returned error",'
            '"metadata":{"raw":"{\\"error\\":{\\"param\\":\\"temperature must be '
            'within [0, 1.5]\\"}}","provider_name":"Xiaomi"}}}'
        )
        mock_client.chat.completions.create.side_effect = [
            Exception(error_msg),
            mock_response,
        ]
        mock_openai.return_value = mock_client

        from hypertune.providers.openrouter_provider import OpenRouterProvider

        provider = OpenRouterProvider()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.generate("Test", temperature=1.8)
            assert result == "Response after retry"
            assert len(w) == 1
            assert "Clamping 1.8 to 1.5" in str(w[0].message)

        assert mock_client.chat.completions.create.call_count == 2
        second_call = mock_client.chat.completions.create.call_args_list[1][1]
        assert second_call["temperature"] == 1.5


class TestAnthropicProvider:
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("hypertune.providers.anthropic_provider.Anthropic")
    def test_init_default_model(self, mock_anthropic):
        from hypertune.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()
        assert provider.model == "claude-sonnet-4.5"

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

        assert GeminiProvider.get_default_model(None) == "gemini-2.5-pro"

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
