import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypertune.providers.base import BaseProvider
from hypertune.providers.registry import ProviderRegistry


class MockProvider(BaseProvider):
    def generate(self, prompt: str, **hyperparameters) -> str:
        return f"Mock response for: {prompt}"

    def get_default_model(self) -> str:
        return "mock-model-v1"

    def get_available_models(self):
        return ["mock-model-v1", "mock-model-v2"]


@pytest.fixture
def mock_provider_class():
    return MockProvider


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture(autouse=True)
def reset_registry():
    original_providers = ProviderRegistry._providers.copy()
    yield
    ProviderRegistry._providers = original_providers


@pytest.fixture
def mock_openai_client():
    with patch("hypertune.providers.openai_provider.OpenAI") as mock:
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test OpenAI response"
        mock_instance.chat.completions.create.return_value = mock_response
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_anthropic_client():
    with patch("hypertune.providers.anthropic_provider.Anthropic") as mock:
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Test Anthropic response"
        mock_instance.messages.create.return_value = mock_response
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_text():
    return "This is a sample text. It contains multiple sentences. Each sentence adds complexity to the analysis."


@pytest.fixture
def sample_prompt():
    return "Explain the concept of machine learning in simple terms."
