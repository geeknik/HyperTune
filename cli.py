import argparse
from hypertune.core import HyperTune
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import numpy as np
import pandas as pd

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def analyze_results(results):
    all_text = " ".join([result['text'] for result in results])
    words = word_tokenize(all_text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    word_freq = Counter(words)
    common_terms = word_freq.most_common(10)
    
    unique_concepts = set()
    for result in results:
        sentences = nltk.sent_tokenize(result['text'])
        for sentence in sentences:
            if any(term in sentence.lower() for term, _ in common_terms):
                unique_concepts.add(sentence)
    
    return common_terms, list(unique_concepts)

def explain_score(result):
    explanation = f"Total Score: {result['total_score']:.2f}\n"
    explanation += f"  Coherence: {result['coherence_score']:.2f} (40% weight) - Measures how well the sentences flow and connect.\n"
    explanation += f"  Relevance: {result['relevance_score']:.2f} (40% weight) - Measures how closely the response relates to the prompt.\n"
    explanation += f"  Complexity: {result['complexity_score']:.2f} (20% weight) - Measures the sophistication of language used.\n"
    return explanation

def plot_score_comparison(results):
    labels = ['Coherence', 'Relevance', 'Complexity']
    data = np.array([[r['coherence_score'], r['relevance_score'], r['complexity_score']] for r in results[:3]])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.25
    
    for i in range(3):
        ax.bar(x + i*width, data[i], width, label=f'Response {i+1}')
    
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Top 3 Responses')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('score_comparison.png')
    plt.close()

def plot_word_frequency(common_terms):
    words, frequencies = zip(*common_terms)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(words), y=list(frequencies))
    plt.title('Top 10 Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('word_frequency.png')
    plt.close()

def plot_hyperparameter_impact(results):
    df = pd.DataFrame([
        {
            'temperature': r['hyperparameters']['temperature'],
            'top_p': r['hyperparameters']['top_p'],
            'frequency_penalty': r['hyperparameters']['frequency_penalty'],
            'presence_penalty': r['hyperparameters']['presence_penalty'],
            'total_score': r['total_score']
        } for r in results
    ])

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Impact of Hyperparameters on Total Score')

    for i, param in enumerate(['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']):
        ax = axs[i // 2, i % 2]
        sns.scatterplot(data=df, x=param, y='total_score', ax=ax)
        ax.set_title(f'{param.capitalize()} vs Total Score')

    plt.tight_layout()
    plt.savefig('hyperparameter_impact.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="HyperTune CLI")
    parser.add_argument("--prompt", help="Input prompt")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--provider", default="openai",
                       choices=["openai", "anthropic", "gemini", "openrouter"],
                       help="LLM provider to use (default: openai)")
    parser.add_argument("--model", help="Specific model to use (optional, uses provider default)")
    parser.add_argument("--list-providers", action="store_true",
                       help="List all available providers and their models")
    
    args = parser.parse_args()
    
    # List providers if requested
    if args.list_providers:
        from hypertune.providers import ProviderFactory
        print("Available LLM Providers:")
        print("=" * 50)
        
        all_info = ProviderFactory.list_all_provider_info()
        for provider_name, info in all_info.items():
            if 'error' in info:
                print(f"\n{provider_name.upper()}: {info['error']}")
                continue
                
            print(f"\n{provider_name.upper()}:")
            print(f"  Default model: {info['model']}")
            print(f"  Available models:")
            for model in info['available_models']:
                print(f"    - {model}")
            
            print(f"  Parameter ranges:")
            for param, ranges in info['parameter_ranges'].items():
                print(f"    {param}: {ranges['min']} - {ranges['max']}")
        return
    
    # Check if prompt is provided when not listing providers
    if not args.prompt:
        parser.error("--prompt is required when not using --list-providers")

    ht = HyperTune(args.prompt, args.iterations, args.provider, args.model)
    results = ht.run()
    
    print("HyperTune Analysis:")
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Number of iterations: {args.iterations}")
    print(f"Provider: {args.provider}")
    if args.model:
        print(f"Model: {args.model}")
    
    print("\nTop 3 Responses (by score):")
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. Score Breakdown:")
        print(explain_score(result))
        print("Hyperparameters:")
        for param, value in result['hyperparameters'].items():
            print(f"  {param}: {value}")
        print("Response:")
        print(result['text'])
    
    plot_score_comparison(results)
    print("\nScore comparison chart saved as 'score_comparison.png'")
    
    plot_hyperparameter_impact(results)
    print("Hyperparameter impact chart saved as 'hyperparameter_impact.png'")
    
    common_terms, unique_concepts = analyze_results(results)
    
    print("\nKey Concepts and Frequency:")
    table = tabulate(common_terms, headers=['Term', 'Frequency'], tablefmt='grid')
    print(table)
    
    plot_word_frequency(common_terms)
    print("Word frequency chart saved as 'word_frequency.png'")
    
    print("\nUnique Insights:")
    for i, concept in enumerate(unique_concepts, 1):
        print(f"{i}. {concept}")
    
    print("\nHyperparameter Analysis:")
    best_result = results[0]
    print("Best performing hyperparameters:")
    for param, value in best_result['hyperparameters'].items():
        print(f"  {param}: {value}")
    
    print("\nHyperparameter Trends:")
    df = pd.DataFrame([r['hyperparameters'] | {'total_score': r['total_score']} for r in results])
    for param in ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']:
        correlation = df[param].corr(df['total_score'])
        print(f"  {param}: {'Positive' if correlation > 0 else 'Negative'} correlation ({correlation:.2f}) with total score")
    
    print("\nKey Takeaways:")
    print("1. The top response provides the most balanced explanation, scoring well across all metrics.")
    print("2. Common themes across responses include: " + ", ".join([term for term, _ in common_terms[:5]]))
    print("3. The complexity of explanations varies, with some responses using more technical language than others.")
    print("4. All top responses maintain high relevance to the prompt, ensuring focused explanations.")
    print(f"5. The best performing set of hyperparameters achieved a total score of {best_result['total_score']:.2f}")
    
    print("\nRecommendations for Further Tuning:")
    print("1. Experiment with narrower ranges of hyperparameters around the best performing values.")
    print("2. Consider increasing the number of iterations to explore more hyperparameter combinations.")
    print("3. Analyze the trade-offs between different scoring components (coherence, relevance, complexity) and adjust weights if needed.")
    
    print("\nMethodology:")
    print("Scoring is based on three key factors:")
    print("- Coherence (40%): How well the ideas connect and flow")
    print("- Relevance (40%): How closely the response aligns with the prompt")
    print("- Complexity (20%): The sophistication of language and concepts used")
    print("Hyperparameters are randomly selected for each iteration to explore different generation settings.")

if __name__ == "__main__":
    main()
