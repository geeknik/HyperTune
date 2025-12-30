import math
from collections import Counter
from typing import Optional

import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def perplexity_score(text: str) -> Optional[float]:
    """
    Calculate a pseudo-perplexity score using unigram entropy.

    Lower scores indicate more predictable/fluent text.
    Returns None for empty or very short text.

    Args:
        text: Input text to score

    Returns:
        Perplexity score (float) or None if text is too short
    """
    if not text or not text.strip():
        return None

    tokens = nltk.word_tokenize(text.lower())
    if len(tokens) < 2:
        return None

    word_counts = Counter(tokens)
    total_words = len(tokens)

    entropy = 0.0
    for count in word_counts.values():
        prob = count / total_words
        entropy -= prob * math.log2(prob)

    return 2**entropy


def semantic_coherence(text: str) -> float:
    """
    Calculate semantic coherence between consecutive sentences.

    Args:
        text: Input text with multiple sentences

    Returns:
        Average cosine similarity between consecutive sentences
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = nltk.sent_tokenize(text)
    embeddings = model.encode(sentences)
    scores = [
        cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        for i in range(len(embeddings) - 1)
    ]
    return sum(scores) / len(scores)


def factual_accuracy(text: str) -> Optional[float]:
    """
    Estimate factual consistency using internal contradiction detection.

    This is a heuristic based on semantic self-consistency, not true fact-checking.
    Higher scores indicate more internally consistent statements.
    Returns None for empty text or single-sentence text.

    Args:
        text: Input text to analyze

    Returns:
        Consistency score (0.0-1.0) or None if insufficient text
    """
    if not text or not text.strip():
        return None

    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return None

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            similarities.append(sim)

    if not similarities:
        return None

    avg_sim = sum(similarities) / len(similarities)
    return float(max(0.0, min(1.0, avg_sim)))
