"""
Tests for hypertune/scoring.py
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestPerplexityScore:
    def test_perplexity_score_returns_float(self):
        from hypertune.scoring import perplexity_score

        result = perplexity_score("This is a test sentence with multiple words.")
        assert result is not None
        assert isinstance(result, float)
        assert result > 0

    def test_perplexity_score_empty_text(self):
        from hypertune.scoring import perplexity_score

        result = perplexity_score("")
        assert result is None

    def test_perplexity_score_single_word(self):
        from hypertune.scoring import perplexity_score

        result = perplexity_score("hello")
        assert result is None

    def test_perplexity_score_repetitive_text_lower(self):
        from hypertune.scoring import perplexity_score

        repetitive = "the the the the the the the the"
        varied = "the quick brown fox jumps over lazy dog"

        score_repetitive = perplexity_score(repetitive)
        score_varied = perplexity_score(varied)

        assert score_repetitive is not None
        assert score_varied is not None
        assert score_repetitive < score_varied


class TestSemanticCoherence:
    @patch("hypertune.scoring.nltk")
    def test_semantic_coherence_single_sentence(self, mock_nltk):
        mock_nltk.sent_tokenize.return_value = ["Only one sentence."]

        from hypertune.scoring import semantic_coherence

        result = semantic_coherence("Only one sentence.")
        assert result == 1.0

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
    @patch("hypertune.scoring.SentenceTransformer")
    @patch("hypertune.scoring.nltk")
    def test_factual_accuracy_returns_float(self, mock_nltk, mock_transformer):
        mock_nltk.sent_tokenize.return_value = [
            "The sky is blue.",
            "Water is wet.",
        ]

        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.8, 0.2], [0.7, 0.3]])

        from hypertune.scoring import factual_accuracy

        result = factual_accuracy("The sky is blue. Water is wet.")
        assert result is not None
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_factual_accuracy_empty_text(self):
        from hypertune.scoring import factual_accuracy

        result = factual_accuracy("")
        assert result is None

    def test_factual_accuracy_single_sentence(self):
        from hypertune.scoring import factual_accuracy

        result = factual_accuracy("Just one sentence here")
        assert result is None

    @patch("hypertune.scoring.SentenceTransformer")
    @patch("hypertune.scoring.nltk")
    def test_factual_accuracy_consistent_text(self, mock_nltk, mock_transformer):
        mock_nltk.sent_tokenize.return_value = [
            "Dogs are animals.",
            "Cats are animals.",
            "Pets are animals.",
        ]

        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        mock_model.encode.return_value = np.array(
            [
                [0.9, 0.1, 0.0],
                [0.85, 0.15, 0.0],
                [0.88, 0.12, 0.0],
            ]
        )

        from hypertune.scoring import factual_accuracy

        result = factual_accuracy(
            "Dogs are animals. Cats are animals. Pets are animals."
        )
        assert result is not None
        assert result > 0.5
