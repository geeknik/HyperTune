import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import random
from .providers import ProviderFactory

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


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
            return 0

        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0

        sentence_vectors = []
        for sentence in sentences:
            words = [
                w.lower()
                for w in word_tokenize(sentence)
                if w.lower() not in self.stop_words
            ]
            sentence_vectors.append(" ".join(words))

        non_empty_vectors = [v for v in sentence_vectors if v.strip()]
        if len(non_empty_vectors) < 2:
            return 0

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(sentence_vectors)
            coherence_scores = []
            for i in range(len(sentences) - 1):
                coherence_scores.append(
                    cosine_similarity(tfidf_matrix[i], tfidf_matrix[i + 1])[0][0]
                )
            return np.mean(coherence_scores) if coherence_scores else 0
        except ValueError:
            return 0

    def evaluate_relevance(self, text, prompt):
        if not text or not text.strip() or not prompt or not prompt.strip():
            return 0

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([prompt, text])
            return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        except ValueError:
            return 0

    def evaluate_complexity(self, text):
        if not text or not text.strip():
            return 0

        words = word_tokenize(text.lower())
        words = [word for word in words if word not in string.punctuation]

        if not words:
            return 0

        avg_word_length = np.mean([len(word) for word in words])
        unique_words_ratio = len(set(words)) / len(words)
        sentences = sent_tokenize(text)

        if not sentences:
            return 0

        avg_sentence_length = np.mean(
            [len(word_tokenize(sentence)) for sentence in sentences]
        )
        norm_word_length = min(avg_word_length / 10, 1)
        norm_sentence_length = min(avg_sentence_length / 30, 1)
        complexity_score = (
            norm_word_length * 0.3
            + unique_words_ratio * 0.3
            + norm_sentence_length * 0.4
        )
        return complexity_score

    def run(self):
        results = self.generate()
        scored_results = self.score(results)
        return scored_results
