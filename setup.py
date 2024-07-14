# setup.py

from setuptools import setup, find_packages

setup(
    name="hypertune",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'openai',
        'click',
        'nltk',
        'scikit-learn',
        'sentence-transformers',
    ],
    entry_points={
        'console_scripts': [
            'hypertune=cli:run',
        ],
    },
)
