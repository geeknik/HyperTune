import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestHyperTune:
    @patch("hypertune.core.ProviderFactory")
    def test_init_default_provider(self, mock_factory):
        mock_provider = MagicMock()
        mock_factory.create_provider.return_value = mock_provider

        from hypertune.core import HyperTune

        ht = HyperTune("Test prompt", iterations=5)

        assert ht.prompt == "Test prompt"
        assert ht.iterations == 5
        mock_factory.create_provider.assert_called_once_with("openai", None)

    @patch("hypertune.core.ProviderFactory")
    def test_init_custom_provider(self, mock_factory):
        mock_provider = MagicMock()
        mock_factory.create_provider.return_value = mock_provider

        from hypertune.core import HyperTune

        ht = HyperTune(
            "Test", iterations=3, provider="anthropic", model="claude-3-haiku"
        )

        mock_factory.create_provider.assert_called_once_with(
            "anthropic", "claude-3-haiku"
        )

    @patch("hypertune.core.ProviderFactory")
    def test_generate(self, mock_factory):
        mock_provider = MagicMock()
        mock_provider.get_parameter_ranges.return_value = {
            "temperature": {"min": 0.0, "max": 1.0},
            "top_p": {"min": 0.0, "max": 1.0},
            "max_tokens": {"min": 1, "max": 4096},
        }
        mock_provider.generate.return_value = "Generated response"
        mock_factory.create_provider.return_value = mock_provider

        from hypertune.core import HyperTune

        ht = HyperTune("Test prompt", iterations=3)
        results = ht.generate()

        assert len(results) == 3
        assert all(r["text"] == "Generated response" for r in results)
        assert all("hyperparameters" in r for r in results)

    @patch("hypertune.core.ProviderFactory")
    def test_generate_handles_errors(self, mock_factory):
        mock_provider = MagicMock()
        mock_provider.get_parameter_ranges.return_value = {
            "temperature": {"min": 0.0, "max": 1.0},
        }
        mock_provider.generate.side_effect = Exception("API Error")
        mock_factory.create_provider.return_value = mock_provider

        from hypertune.core import HyperTune

        ht = HyperTune("Test", iterations=2)
        results = ht.generate()

        assert len(results) == 0

    @patch("hypertune.core.ProviderFactory")
    def test_evaluate_coherence_single_sentence(self, mock_factory):
        mock_factory.create_provider.return_value = MagicMock()

        from hypertune.core import HyperTune

        ht = HyperTune("Test", iterations=1)
        score = ht.evaluate_coherence("Single sentence.")

        assert score == 1.0

    @patch("hypertune.core.ProviderFactory")
    def test_evaluate_coherence_multiple_sentences(self, mock_factory):
        mock_factory.create_provider.return_value = MagicMock()

        from hypertune.core import HyperTune

        ht = HyperTune("Test", iterations=1)
        text = "Machine learning is a field of AI. It uses algorithms to learn patterns. Deep learning is a subset of machine learning."
        score = ht.evaluate_coherence(text)

        assert 0 <= score <= 1

    @patch("hypertune.core.ProviderFactory")
    def test_evaluate_relevance(self, mock_factory):
        mock_factory.create_provider.return_value = MagicMock()

        from hypertune.core import HyperTune

        ht = HyperTune("machine learning", iterations=1)

        relevant_text = "Machine learning is a method of data analysis."
        irrelevant_text = "The weather today is sunny and warm."

        relevant_score = ht.evaluate_relevance(relevant_text, "machine learning")
        irrelevant_score = ht.evaluate_relevance(irrelevant_text, "machine learning")

        assert relevant_score > irrelevant_score

    @patch("hypertune.core.ProviderFactory")
    def test_evaluate_complexity(self, mock_factory):
        mock_factory.create_provider.return_value = MagicMock()

        from hypertune.core import HyperTune

        ht = HyperTune("Test", iterations=1)

        simple_text = "This is a test. It is short."
        complex_text = "The implementation of sophisticated algorithms necessitates comprehensive understanding of computational paradigms and mathematical foundations."

        simple_score = ht.evaluate_complexity(simple_text)
        complex_score = ht.evaluate_complexity(complex_text)

        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1

    @patch("hypertune.core.ProviderFactory")
    def test_score_results(self, mock_factory):
        mock_factory.create_provider.return_value = MagicMock()

        from hypertune.core import HyperTune

        ht = HyperTune("machine learning", iterations=1)

        results = [
            {
                "text": "Machine learning uses algorithms. It learns from data.",
                "hyperparameters": {"temp": 0.7},
            },
            {
                "text": "Dogs are great pets. They are loyal.",
                "hyperparameters": {"temp": 0.5},
            },
        ]

        scored = ht.score(results)

        assert len(scored) == 2
        assert all("total_score" in r for r in scored)
        assert all("coherence_score" in r for r in scored)
        assert all("relevance_score" in r for r in scored)
        assert all("complexity_score" in r for r in scored)
        assert scored[0]["total_score"] >= scored[1]["total_score"]

    @patch("hypertune.core.ProviderFactory")
    def test_run_end_to_end(self, mock_factory):
        mock_provider = MagicMock()
        mock_provider.get_parameter_ranges.return_value = {
            "temperature": {"min": 0.0, "max": 1.0},
            "top_p": {"min": 0.0, "max": 1.0},
            "max_tokens": {"min": 1, "max": 1024},
        }
        mock_provider.generate.return_value = (
            "Test response with multiple sentences. This helps with scoring."
        )
        mock_factory.create_provider.return_value = mock_provider

        from hypertune.core import HyperTune

        ht = HyperTune("Test prompt", iterations=2)
        results = ht.run()

        assert len(results) == 2
        assert results[0]["total_score"] >= results[1]["total_score"]
