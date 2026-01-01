# HyperTune

HyperTune is an advanced tool for optimizing and analyzing text generation using multiple LLM providers. It explores various hyperparameter combinations to produce high-quality responses to given prompts, and provides comprehensive analysis of the results.

## Features

- **Multi-Provider Support**: Works with OpenAI, Anthropic Claude, Google Gemini, and OpenRouter
- **Semantic Scoring**: Uses sentence embeddings for accurate coherence and relevance measurement
- **Quality Detection**: Automatically penalizes degenerate outputs (repetitive text, garbage responses)
- **JSON Export**: Save full results with metadata for further analysis
- **Interactive Dashboards**: Generate insightful visualizations of hyperparameter impact
- **Flexible Output**: Control verbosity with truncation and top-N display options

## Supported Providers

### OpenAI
- Models: GPT-5.2, GPT-5.2-pro, GPT-5, GPT-5-mini, GPT-5-nano, GPT-4.1
- Open-weight: gpt-oss-120b, gpt-oss-20b
- Parameters: temperature, top_p, max_tokens, frequency_penalty, presence_penalty

### Anthropic Claude
- Models: Claude Opus 4.5, Claude Sonnet 4.5, Claude Haiku 4.5
- Parameters: temperature, top_p, max_tokens

### Google Gemini
- Models: Gemini 3 Pro, Gemini 3 Flash, Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.5 Flash-Lite
- Parameters: temperature, top_p, max_tokens, top_k

### OpenRouter
- Models: Access to hundreds of models including all OpenAI, Anthropic, Google, Meta, Mistral, and more
- Parameters: temperature, top_p, max_tokens, frequency_penalty, presence_penalty

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/geeknik/hypertune
   cd hypertune
   python -m venv venv && source venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install openai anthropic google-genai scikit-learn nltk matplotlib seaborn pandas sentence-transformers
   ```

3. Set up your API keys as environment variables:
   ```bash
   export OPENAI_API_KEY='your-openai-api-key-here'
   export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
   export GOOGLE_API_KEY='your-google-api-key-here'
   export OPENROUTER_API_KEY='your-openrouter-api-key-here'
   ```

## Usage

### Basic Usage

```bash
python cli.py --prompt "Your prompt here" --iterations 10
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--prompt` | Input prompt for generation (required) |
| `--iterations` | Number of iterations (default: 5) |
| `--provider` | LLM provider: openai, anthropic, gemini, openrouter (default: openai) |
| `--model` | Specific model to use (uses provider default if not specified) |
| `--output FILE` | Save full results to JSON file |
| `--top N` | Number of top results to display (default: 3) |
| `--full` | Show full response text (default: truncated to 500 chars) |
| `--no-charts` | Disable chart generation |
| `--list-providers` | List available providers and models |

### Examples

```bash
# Basic run with OpenAI
python cli.py --prompt "Explain quantum computing" --iterations 10

# Use Anthropic and save results
python cli.py --prompt "Write a poem" --provider anthropic --output results.json

# OpenRouter with specific model, show all responses
python cli.py --prompt "Summarize AI" --provider openrouter --model meta-llama/llama-3.1-70b-instruct --top 10 --full

# Quick run without charts
python cli.py --prompt "Hello world" --iterations 3 --no-charts
```

## Output

### Terminal Output

The CLI displays:
- Score breakdown for top responses (coherence, relevance, complexity, quality penalty)
- Hyperparameters used for each response
- Response text (truncated by default, use `--full` for complete text)
- Score statistics (best, worst, mean, std)
- Best performing hyperparameters

### JSON Export

Use `--output results.json` to save:
```json
{
  "metadata": {
    "timestamp": "2025-01-01T12:00:00",
    "prompt": "Your prompt",
    "provider": "openai",
    "model": "gpt-4",
    "iterations": 10
  },
  "results": [...],
  "summary": {
    "best_score": 0.85,
    "best_hyperparameters": {...},
    "scores_distribution": {"min": 0.2, "max": 0.85, "mean": 0.6}
  }
}
```

### Visualizations

Two charts are generated (disable with `--no-charts`):

**`hypertune_dashboard.png`** - Analysis dashboard with:
- Score distribution histogram
- Stacked score breakdown showing coherence/relevance/complexity contributions
- Parameter correlation heatmap
- Temperature vs score trend line
- Best vs worst response comparison

<img width="2098" height="1630" alt="image" src="https://github.com/user-attachments/assets/7fbd50b2-90e4-4403-b40d-4b99e8ab42cf" />

**`hyperparameter_exploration.png`** - Detailed parameter analysis:
- Scatter plots with trend lines and correlation coefficients
- Box plots showing score variance by parameter range

<img width="2979" height="1183" alt="image" src="https://github.com/user-attachments/assets/569a776c-244b-44b7-ba4e-79421fd8724b" />


## How It Works

1. **Generation**: Multiple responses are generated using random hyperparameter combinations within valid ranges for your chosen provider.

2. **Scoring**: Each response is scored on three dimensions using sentence embeddings (all-MiniLM-L6-v2):
   - **Coherence (40%)**: Semantic similarity between consecutive sentences
   - **Relevance (40%)**: Semantic similarity between response and prompt
   - **Complexity (20%)**: Vocabulary diversity, word length, sentence structure

3. **Quality Filtering**: Degenerate responses (repetitive characters, word spam, low diversity) receive a quality penalty that reduces their total score.

4. **Analysis**: Results are sorted by score and analyzed to identify optimal hyperparameter combinations.

## Contributing

Contributions to HyperTune are welcome! Please feel free to submit a PR.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool interacts with various LLM provider APIs. The authors are not responsible for any misuse or for any offensive content that may be generated.
