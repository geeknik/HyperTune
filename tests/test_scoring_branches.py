import builtins
from unittest.mock import patch

import numpy as np
import pytest


def test_get_embedding_model_raises_when_sentence_transformers_missing():
    from hypertune import scoring

    scoring.SentenceTransformer = None
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing dependency")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(RuntimeError, match="sentence-transformers is required"):
            scoring.get_embedding_model()


def test_semantic_coherence_returns_zero_for_blank_text():
    from hypertune.scoring import semantic_coherence

    assert semantic_coherence("   ") == 0.0


@patch("hypertune.scoring.ensure_nltk_resources")
@patch("hypertune.scoring.nltk")
@patch("hypertune.scoring.get_embedding_model")
def test_semantic_coherence_returns_one_when_embeddings_underflow(
    mock_get_model, mock_nltk, _mock_ensure
):
    from hypertune.scoring import semantic_coherence

    mock_nltk.sent_tokenize.return_value = ["a", "b"]
    mock_model = mock_get_model.return_value
    mock_model.encode.return_value = np.array([[1.0, 0.0]])

    assert semantic_coherence("a. b.") == 1.0


@patch("hypertune.scoring.ensure_nltk_resources")
@patch("hypertune.scoring.nltk")
@patch("hypertune.scoring.get_embedding_model")
def test_factual_accuracy_returns_none_when_no_similarity_pairs(
    mock_get_model, mock_nltk, _mock_ensure
):
    from hypertune.scoring import factual_accuracy

    mock_nltk.sent_tokenize.return_value = ["a", "b"]
    mock_model = mock_get_model.return_value
    mock_model.encode.return_value = np.array([[1.0, 0.0]])

    assert factual_accuracy("a. b.") is None
