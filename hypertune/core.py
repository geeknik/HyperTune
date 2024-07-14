import openai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import random

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class HyperTune:
    def __init__(self, prompt, iterations):
        self.prompt = prompt
        self.iterations = iterations
        self.stop_words = set(stopwords.words('english'))

    def generate(self):
        results = []
        for _ in range(self.iterations):
            temperature = round(random.uniform(0.1, 1.0), 2)
            top_p = round(random.uniform(0.1, 1.0), 2)
            frequency_penalty = round(random.uniform(0.0, 2.0), 2)
            presence_penalty = round(random.uniform(0.0, 2.0), 2)

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": self.prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            results.append({
                'text': response['choices'][0]['message']['content'],
                'hyperparameters': {
                    'temperature': temperature,
                    'top_p': top_p,
                    'frequency_penalty': frequency_penalty,
                    'presence_penalty': presence_penalty
                }
            })
        return results

    def score(self, results):
        scored_results = []
        for result in results:
            coherence_score = self.evaluate_coherence(result['text'])
            relevance_score = self.evaluate_relevance(result['text'], self.prompt)
            complexity_score = self.evaluate_complexity(result['text'])
            total_score = (coherence_score * 0.4 +
                           relevance_score * 0.4 +
                           complexity_score * 0.2)
            scored_results.append({
                'text': result['text'],
                'total_score': total_score,
                'coherence_score': coherence_score,
                'relevance_score': relevance_score,
                'complexity_score': complexity_score,
                'hyperparameters': result['hyperparameters']
            })
        return sorted(scored_results, key=lambda x: x['total_score'], reverse=True)

    def evaluate_coherence(self, text):
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0
        sentence_vectors = []
        for sentence in sentences:
            words = [w.lower() for w in word_tokenize(sentence) if w.lower() not in self.stop_words]
            sentence_vectors.append(' '.join(words))
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentence_vectors)
        coherence_scores = []
        for i in range(len(sentences) - 1):
            coherence_scores.append(cosine_similarity(tfidf_matrix[i], tfidf_matrix[i+1])[0][0])
        return np.mean(coherence_scores)

    def evaluate_relevance(self, text, prompt):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([prompt, text])
        return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    def evaluate_complexity(self, text):
        words = word_tokenize(text.lower())
        words = [word for word in words if word not in string.punctuation]
        avg_word_length = np.mean([len(word) for word in words])
        unique_words_ratio = len(set(words)) / len(words)
        sentences = sent_tokenize(text)
        avg_sentence_length = np.mean([len(word_tokenize(sentence)) for sentence in sentences])
        # Normalize each component
        norm_word_length = min(avg_word_length / 10, 1)  # Assume max avg word length is 10
        # unique_words_ratio is already between 0 and 1
        norm_sentence_length = min(avg_sentence_length / 30, 1)  # Assume max avg sentence length is 30
        complexity_score = (norm_word_length * 0.3 +
                            unique_words_ratio * 0.3 +
                            norm_sentence_length * 0.4)
        return complexity_score  # Already normalized between 0 and 1

    def run(self):
        results = self.generate()
        scored_results = self.score(results)
        return scored_results
