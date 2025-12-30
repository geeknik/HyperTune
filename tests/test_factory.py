import pytest
from unittest.mock import patch, MagicMock
from hypertune.providers.factory import ProviderFactory
from hypertune.providers.registry import ProviderRegistry


class TestProviderFactory:
    def test_create_provider(self, mock_provider_class):
        ProviderRegistry.register("mock", mock_provider_class)
        provider = ProviderFactory.create_provider("mock")
        assert provider is not None
        assert provider.model == "mock-model-v1"

    def test_create_provider_with_custom_model(self, mock_provider_class):
        ProviderRegistry.register("mock", mock_provider_class)
        provider = ProviderFactory.create_provider("mock", model="custom-model")
        assert provider.model == "custom-model"

    def test_create_nonexistent_provider_raises(self):
        with pytest.raises(ValueError):
            ProviderFactory.create_provider("nonexistent")

    def test_get_available_providers(self, mock_provider_class):
        ProviderRegistry.register("available1", mock_provider_class)
        ProviderRegistry.register("available2", mock_provider_class)
        providers = ProviderFactory.get_available_providers()
        assert "available1" in providers
        assert "available2" in providers

    def test_get_provider_info(self, mock_provider_class):
        ProviderRegistry.register("info_test", mock_provider_class)
        info = ProviderFactory.get_provider_info("info_test")
        assert "model" in info
        assert "available_models" in info
        assert "parameter_ranges" in info
        assert info["model"] == "mock-model-v1"

    def test_list_all_provider_info(self, mock_provider_class):
        ProviderRegistry.clear()
        ProviderRegistry.register("provider_a", mock_provider_class)
        ProviderRegistry.register("provider_b", mock_provider_class)
        all_info = ProviderFactory.list_all_provider_info()
        assert "provider_a" in all_info
        assert "provider_b" in all_info

    def test_list_all_provider_info_handles_errors(self, mock_provider_class):
        class FailingProvider(mock_provider_class):
            def get_provider_info(self):
                raise RuntimeError("Info retrieval failed")

        ProviderRegistry.clear()
        ProviderRegistry.register("failing", FailingProvider)
        all_info = ProviderFactory.list_all_provider_info()
        assert "failing" in all_info
        assert "error" in all_info["failing"]
