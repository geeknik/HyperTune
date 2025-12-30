import math
import random
import string

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
            coherence_score = self.evaluate_coherence(result["text"])
            relevance_score = self.evaluate_relevance(result["text"], self.prompt)
            complexity_score = self.evaluate_complexity(result["text"])
            total_score = (
                coherence_score * 0.4 + relevance_score * 0.4 + complexity_score * 0.2
            )
            scored_results.append(
                {
                    "text": result["text"],
                    "total_score": total_score,
                    "coherence_score": coherence_score,
                    "relevance_score": relevance_score,
                    "complexity_score": complexity_score,
                    "hyperparameters": result["hyperparameters"],
                }
            )
        return sorted(scored_results, key=lambda x: x["total_score"], reverse=True)

    def evaluate_coherence(self, text):
        if not text or not text.strip():
            return 0.0

        sentences = sent_tokenize(text)
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
        baseline = 0.3
        ceiling = 0.85
        if raw_score <= baseline:
            return 0.0
        if raw_score >= ceiling:
            return 1.0
        return (raw_score - baseline) / (ceiling - baseline)

    def evaluate_relevance(self, text, prompt):
        if not text or not text.strip() or not prompt or not prompt.strip():
            return 0.0

        model = get_embedding_model()
        embeddings = model.encode([prompt, text])
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
