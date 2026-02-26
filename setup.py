"""HyperTune package setup configuration."""

from setuptools import setup, find_packages

setup(
    name="hypertune",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "openai",
        "anthropic",
        "google-genai",
        "click",
        "nltk",
        "scikit-learn",
        "sentence-transformers",
        "matplotlib",
        "seaborn",
        "tabulate",
        "pandas",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "hypertune=cli:main",
        ],
    },
)
