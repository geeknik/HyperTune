"""Helpers for validating required NLTK resources at runtime."""

from typing import Iterable

import nltk

_RESOURCE_PATHS = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
}
_VALIDATED_RESOURCES = set()


def ensure_nltk_resources(resources: Iterable[str]) -> None:
    """Ensure required NLTK resources are installed before tokenization."""
    missing = []
    for resource in resources:
        if resource in _VALIDATED_RESOURCES:
            continue

        path = _RESOURCE_PATHS.get(resource)
        if path is None:
            raise ValueError(f"Unsupported NLTK resource: {resource}")

        try:
            nltk.data.find(path)
            _VALIDATED_RESOURCES.add(resource)
        except LookupError:
            missing.append(resource)

    if missing:
        missing_resources = ", ".join(missing)
        install_cmd = "python -m nltk.downloader " + " ".join(missing)
        raise RuntimeError(
            f"Missing NLTK resources: {missing_resources}. Install them with `{install_cmd}`."
        )
