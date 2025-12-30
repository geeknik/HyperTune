import pytest
from hypertune.providers.registry import ProviderRegistry
from hypertune.providers.base import BaseProvider


class TestProviderRegistry:
    def test_register_valid_provider(self, mock_provider_class):
        ProviderRegistry.register("test_provider", mock_provider_class)
        assert ProviderRegistry.is_registered("test_provider")

    def test_register_invalid_provider_raises(self):
        class NotAProvider:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseProvider"):
            ProviderRegistry.register("invalid", NotAProvider)

    def test_register_case_insensitive(self, mock_provider_class):
        ProviderRegistry.register("TestProvider", mock_provider_class)
        assert ProviderRegistry.is_registered("testprovider")
        assert ProviderRegistry.is_registered("TESTPROVIDER")

    def test_get_provider_class(self, mock_provider_class):
        ProviderRegistry.register("test", mock_provider_class)
        retrieved = ProviderRegistry.get_provider_class("test")
        assert retrieved is mock_provider_class

    def test_get_nonexistent_provider_raises(self):
        with pytest.raises(ValueError, match="not found"):
            ProviderRegistry.get_provider_class("nonexistent")

    def test_list_providers(self, mock_provider_class):
        initial_count = len(ProviderRegistry.list_providers())
        ProviderRegistry.register("new_provider", mock_provider_class)
        providers = ProviderRegistry.list_providers()
        assert len(providers) == initial_count + 1
        assert "new_provider" in providers

    def test_is_registered_returns_false_for_unknown(self):
        assert not ProviderRegistry.is_registered("unknown_provider")

    def test_unregister_provider(self, mock_provider_class):
        ProviderRegistry.register("to_remove", mock_provider_class)
        assert ProviderRegistry.is_registered("to_remove")
        ProviderRegistry.unregister("to_remove")
        assert not ProviderRegistry.is_registered("to_remove")

    def test_unregister_nonexistent_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            ProviderRegistry.unregister("does_not_exist")

    def test_clear_removes_all(self, mock_provider_class):
        ProviderRegistry.register("provider1", mock_provider_class)
        ProviderRegistry.register("provider2", mock_provider_class)
        ProviderRegistry.clear()
        assert len(ProviderRegistry.list_providers()) == 0
