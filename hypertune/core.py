import math
import random
import re
import string
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from .providers import ProviderFactory

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def strip_markdown(text: str) -> str:
    """Remove markdown formatting for cleaner text analysis."""
    text = re.sub(r"\[\[\d+\]\]\([^)]+\)", "", text)  # [[1]](url)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url) -> text
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\|[^\n]+\|", "", text)  # table rows
    text = re.sub(r"\|-+\|", "", text)  # table separators
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # *italic*
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-=*]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def compute_quality_penalty(text: str) -> float:
    """Detect degenerate LLM outputs. Returns 1.0 for good text, <1.0 for garbage."""
    if not text or len(text.strip()) < 10:
        return 0.0

    clean_text = strip_markdown(text)
    if len(clean_text) < 20:
        return 0.3 if len(text) > 100 else 1.0

    chars = list(clean_text)
    char_counts = Counter(chars)
    most_common_char, most_common_count = char_counts.most_common(1)[0]
    repetition_ratio = most_common_count / len(chars)

    if repetition_ratio > 0.4 and most_common_char in ",.;!?'\"()-_\n\t ":
        return max(0.0, 1.0 - repetition_ratio * 2)

    # 8+ consecutive identical non-formatting chars (e.g., "aaaaaaaa")
    repeat_sequences = re.findall(r"([^-=*_#|\s])\1{7,}", clean_text)
    if len(repeat_sequences) > 2:
        return 0.2

    words = clean_text.split()
    if len(words) >= 10:
        word_counts = Counter(w.lower() for w in words)
        _, top_count = word_counts.most_common(1)[0]
        if top_count / len(words) > 0.5:
            return 0.3

    unique_chars = len(set(chars))
    if unique_chars < 25 and len(chars) > 100:
        return 0.2

    return 1.0


class HyperTune:
    def __init__(self, prompt, iterations, provider="openai", model=None):
        """
        Initialize HyperTune with specified provider

        Args:
            prompt: The prompt to generate responses for
            iterations: Number of iterations to run
            provider: LLM provider to use (openai, anthropic, gemini, openrouter)
            model: Specific model to use (optional, uses provider default if None)
        """
        self.prompt = prompt
        self.iterations = iterations
        self.stop_words = set(stopwords.words("english"))
        self.provider = ProviderFactory.create_provider(provider, model)

    def generate(self):
        """
        Generate responses using the configured provider

        Returns:
            List of results with text and hyperparameters
        """
        results = []
        parameter_ranges = self.provider.get_parameter_ranges()

        for _ in range(self.iterations):
            # Generate random hyperparameters within provider's valid ranges
            hyperparameters = {}

            # Temperature
            if "temperature" in parameter_ranges:
                temp_range = parameter_ranges["temperature"]
                hyperparameters["temperature"] = round(
                    random.uniform(temp_range["min"], temp_range["max"]), 2
                )

            # Top_p
            if "top_p" in parameter_ranges:
                top_p_range = parameter_ranges["top_p"]
                hyperparameters["top_p"] = round(
                    random.uniform(top_p_range["min"], top_p_range["max"]), 2
                )

            # Provider-specific parameters
            if "frequency_penalty" in parameter_ranges:
                freq_range = parameter_ranges["frequency_penalty"]
                hyperparameters["frequency_penalty"] = round(
                    random.uniform(freq_range["min"], freq_range["max"]), 2
                )

            if "presence_penalty" in parameter_ranges:
                pres_range = parameter_ranges["presence_penalty"]
                hyperparameters["presence_penalty"] = round(
                    random.uniform(pres_range["min"], pres_range["max"]), 2
                )

            if "top_k" in parameter_ranges:
                top_k_range = parameter_ranges["top_k"]
                hyperparameters["top_k"] = random.randint(
                    int(top_k_range["min"]), int(top_k_range["max"])
                )

            # Max tokens
            if "max_tokens" in parameter_ranges:
                max_tokens_range = parameter_ranges["max_tokens"]
                hyperparameters["max_tokens"] = random.randint(
                    int(max_tokens_range["min"]),
                    min(1024, int(max_tokens_range["max"])),
                )

            # Generate response using provider
            try:
                response_text = self.provider.generate(self.prompt, **hyperparameters)
                results.append(
                    {"text": response_text, "hyperparameters": hyperparameters}
                )
            except Exception as e:
                print(f"Error generating response: {e}")
                # Continue with next iteration
                continue

        return results

    def score(self, results):
        scored_results = []
        for result in results:
            quality_penalty = compute_quality_penalty(result["text"])
            coherence_score = self.evaluate_coherence(result["text"])
            relevance_score = self.evaluate_relevance(result["text"], self.prompt)
            complexity_score = self.evaluate_complexity(result["text"])

            raw_score = (
                coherence_score * 0.4 + relevance_score * 0.4 + complexity_score * 0.2
            )
            total_score = raw_score * quality_penalty

            scored_results.append(
                {
                    "text": result["text"],
                    "total_score": total_score,
                    "coherence_score": coherence_score,
                    "relevance_score": relevance_score,
                    "complexity_score": complexity_score,
                    "quality_penalty": quality_penalty,
                    "hyperparameters": result["hyperparameters"],
                }
            )
        return sorted(scored_results, key=lambda x: x["total_score"], reverse=True)

    def evaluate_coherence(self, text):
        if not text or not text.strip():
            return 0.0

        clean_text = strip_markdown(text)
        sentences = sent_tokenize(clean_text)
        if len(sentences) < 2:
            return 1.0

        model = get_embedding_model()
        embeddings = model.encode(sentences)

        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)

        if not similarities:
            return 0.0

        raw_score = float(np.mean(similarities))
        return self._calibrate_coherence(raw_score)

    def _calibrate_coherence(self, raw_score):
        baseline = 0.15
        ceiling = 0.70
        if raw_score <= baseline:
            return 0.0
        if raw_score >= ceiling:
            return 1.0
        return (raw_score - baseline) / (ceiling - baseline)

    def evaluate_relevance(self, text, prompt):
        if not text or not text.strip() or not prompt or not prompt.strip():
            return 0.0

        clean_text = strip_markdown(text)
        model = get_embedding_model()
        embeddings = model.encode([prompt, clean_text])
        raw_score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        return self._calibrate_relevance(raw_score)

    def _calibrate_relevance(self, raw_score):
        baseline = 0.2
        ceiling = 0.75
        if raw_score <= baseline:
            return 0.0
        if raw_score >= ceiling:
            return 1.0
        return (raw_score - baseline) / (ceiling - baseline)

    def evaluate_complexity(self, text):
        if not text or not text.strip():
            return 0.0

        words = word_tokenize(text.lower())
        words = [word for word in words if word not in string.punctuation]

        if not words:
            return 0.0

        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0

        avg_word_length = np.mean([len(word) for word in words])
        word_length_score = self._sigmoid_normalize(avg_word_length, midpoint=5, k=0.8)

        unique_ratio = len(set(words)) / len(words)
        vocab_score = self._sigmoid_normalize(unique_ratio, midpoint=0.4, k=8)

        avg_sentence_length = np.mean(
            [len(word_tokenize(sentence)) for sentence in sentences]
        )
        sentence_score = self._sigmoid_normalize(
            avg_sentence_length, midpoint=15, k=0.15
        )

        return word_length_score * 0.25 + vocab_score * 0.35 + sentence_score * 0.40

    def _sigmoid_normalize(self, value, midpoint, k):
        return 1 / (1 + math.exp(-k * (value - midpoint)))

    def run(self):
        results = self.generate()
        scored_results = self.score(results)
        return scored_results
