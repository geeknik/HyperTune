from unittest.mock import MagicMock

import pytest


def test_database_insert_and_query_roundtrip():
    from hypertune.database import Database

    db = Database(":memory:")
    db.insert_result("prompt", "output", 0.7, 0.9, 0.8)

    row = db.conn.execute(
        "SELECT prompt, output, temperature, top_p, score FROM results"
    ).fetchone()
    assert row == ("prompt", "output", 0.7, 0.9, 0.8)
    assert db.get_best_params("prompt") is None


def test_hyperpredictor_train_and_predict_delegate_to_model():
    from hypertune.predictor import HyperPredictor

    predictor = HyperPredictor()
    predictor.model = MagicMock()
    predictor.model.predict.return_value = [0.42]

    X = [[1, 2, 3], [4, 5, 6]]
    y = [0.1, 0.2]
    predictor.train(X, y)
    result = predictor.predict([[7, 8, 9]])

    predictor.model.fit.assert_called_once_with(X, y)
    predictor.model.predict.assert_called_once_with([[7, 8, 9]])
    assert result == [0.42]


def test_ensure_nltk_resources_raises_for_unsupported_resource():
    from hypertune import nltk_utils

    with pytest.raises(ValueError, match="Unsupported NLTK resource"):
        nltk_utils.ensure_nltk_resources(("unsupported_resource",))


def test_ensure_nltk_resources_raises_for_missing_resources(monkeypatch):
    from hypertune import nltk_utils

    nltk_utils._VALIDATED_RESOURCES.clear()

    def always_missing(_path):
        raise LookupError("missing")

    monkeypatch.setattr(nltk_utils.nltk.data, "find", always_missing)

    with pytest.raises(RuntimeError, match="Missing NLTK resources: punkt"):
        nltk_utils.ensure_nltk_resources(("punkt",))


def test_ensure_nltk_resources_caches_successful_checks(monkeypatch):
    from hypertune import nltk_utils

    nltk_utils._VALIDATED_RESOURCES.clear()
    calls = []

    def fake_find(path):
        calls.append(path)
        return True

    monkeypatch.setattr(nltk_utils.nltk.data, "find", fake_find)

    nltk_utils.ensure_nltk_resources(("punkt",))
    nltk_utils.ensure_nltk_resources(("punkt",))

    assert calls == ["tokenizers/punkt"]
