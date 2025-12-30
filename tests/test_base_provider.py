import pytest
import warnings
from hypertune.providers.base import BaseProvider


class ConcreteProvider(BaseProvider):
    def generate(self, prompt: str, **hyperparameters) -> str:
        return f"Response: {prompt}"

    def get_default_model(self) -> str:
        return "concrete-v1"

    def get_available_models(self):
        return ["concrete-v1", "concrete-v2"]


class TestBaseProvider:
    def test_init_with_default_model(self):
        provider = ConcreteProvider()
        assert provider.model == "concrete-v1"

    def test_init_with_custom_model(self):
        provider = ConcreteProvider(model="custom")
        assert provider.model == "custom"

    def test_validate_temperature_valid(self):
        provider = ConcreteProvider()
        result = provider.validate_hyperparameters(temperature=0.5)
        assert result["temperature"] == 0.5

    def test_validate_temperature_invalid_warns(self):
        provider = ConcreteProvider()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.validate_hyperparameters(temperature=5.0)
            assert len(w) == 1
            assert "Invalid temperature" in str(w[0].message)
            assert result["temperature"] == 0.7

    def test_validate_temperature_negative_warns(self):
        provider = ConcreteProvider()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.validate_hyperparameters(temperature=-1)
            assert result["temperature"] == 0.7

    def test_validate_top_p_valid(self):
        provider = ConcreteProvider()
        result = provider.validate_hyperparameters(top_p=0.8)
        assert result["top_p"] == 0.8

    def test_validate_top_p_invalid_warns(self):
        provider = ConcreteProvider()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.validate_hyperparameters(top_p=1.5)
            assert len(w) == 1
            assert result["top_p"] == 0.9

    def test_validate_max_tokens_valid(self):
        provider = ConcreteProvider()
        result = provider.validate_hyperparameters(max_tokens=512)
        assert result["max_tokens"] == 512

    def test_validate_max_tokens_invalid_warns(self):
        provider = ConcreteProvider()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.validate_hyperparameters(max_tokens=0)
            assert len(w) == 1
            assert result["max_tokens"] == 1024

    def test_validate_multiple_params(self):
        provider = ConcreteProvider()
        result = provider.validate_hyperparameters(
            temperature=0.7, top_p=0.9, max_tokens=256
        )
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["max_tokens"] == 256

    def test_get_parameter_ranges(self):
        provider = ConcreteProvider()
        ranges = provider.get_parameter_ranges()
        assert "temperature" in ranges
        assert "top_p" in ranges
        assert "max_tokens" in ranges
        assert ranges["temperature"]["min"] == 0.0
        assert ranges["temperature"]["max"] == 2.0

    def test_get_provider_info(self):
        provider = ConcreteProvider()
        info = provider.get_provider_info()
        assert info["name"] == "ConcreteProvider"
        assert info["model"] == "concrete-v1"
        assert "concrete-v1" in info["available_models"]
        assert "parameter_ranges" in info

    def test_validate_provider_specific_params_default(self):
        provider = ConcreteProvider()
        result = provider._validate_provider_specific_params(custom_param="value")
        assert result == {}
