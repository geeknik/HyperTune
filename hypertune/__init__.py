"""HyperTune package exports with lazy imports."""

from typing import Any

__all__ = ["HyperTune", "Database", "HyperPredictor"]


def __getattr__(name: str) -> Any:
    if name == "HyperTune":
        from .core import HyperTune

        return HyperTune
    if name == "Database":
        from .database import Database

        return Database
    if name == "HyperPredictor":
        from .predictor import HyperPredictor

        return HyperPredictor
    raise AttributeError(f"module 'hypertune' has no attribute '{name}'")
