"""
Tests for hypertune/scoring.py
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestPerplexityScore:
    def test_perplexity_score_returns_none(self):
        from hypertune.scoring import perplexity_score

        result = perplexity_score("This is a test sentence.")
        assert result is None

    def test_perplexity_score_empty_text(self):
        from hypertune.scoring import perplexity_score

        result = perplexity_score("")
        assert result is None


class TestSemanticCoherence:
    @patch("hypertune.scoring.SentenceTransformer")
    @patch("hypertune.scoring.nltk")
    def test_semantic_coherence_multiple_sentences(self, mock_nltk, mock_transformer):
        mock_nltk.sent_tokenize.return_value = [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
        ]

        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.array(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]]
        )

        from hypertune.scoring import semantic_coherence

        result = semantic_coherence("First sentence. Second sentence. Third sentence.")

        mock_nltk.sent_tokenize.assert_called_once()
        mock_model.encode.assert_called_once()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @patch("hypertune.scoring.SentenceTransformer")
    @patch("hypertune.scoring.nltk")
    def test_semantic_coherence_two_sentences(self, mock_nltk, mock_transformer):
        mock_nltk.sent_tokenize.return_value = ["Hello world.", "Goodbye world."]

        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.8, 0.2]])

        from hypertune.scoring import semantic_coherence

        result = semantic_coherence("Hello world. Goodbye world.")

        assert isinstance(result, float)

    @patch("hypertune.scoring.SentenceTransformer")
    @patch("hypertune.scoring.nltk")
    def test_semantic_coherence_identical_sentences(self, mock_nltk, mock_transformer):
        mock_nltk.sent_tokenize.return_value = ["Same sentence.", "Same sentence."]

        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        from hypertune.scoring import semantic_coherence

        result = semantic_coherence("Same sentence. Same sentence.")

        assert result == pytest.approx(1.0, abs=0.01)


class TestFactualAccuracy:
    def test_factual_accuracy_returns_none(self):
        from hypertune.scoring import factual_accuracy

        result = factual_accuracy("The sky is blue.")
        assert result is None

    def test_factual_accuracy_empty_text(self):
        from hypertune.scoring import factual_accuracy

        result = factual_accuracy("")
        assert result is None
