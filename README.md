# HyperTune

HyperTune is an advanced tool for optimizing and analyzing text generation using multiple LLM providers. It explores various hyperparameter combinations to produce high-quality responses to given prompts, and provides comprehensive analysis of the results.

## Features

- **Multi-Provider Support**: Works with OpenAI, Anthropic Claude, Google Gemini, and OpenRouter
- Generate multiple responses to a given prompt using different hyperparameter settings
- Score responses based on coherence, relevance, and complexity
- Analyze common themes and unique insights across responses
- Visualize the impact of hyperparameters on response quality
- Provide detailed explanations of scoring and recommendations for further tuning

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
   ```
   git clone https://github.com/geeknik/hypertune
   cd hypertune
   python -m venv venv && source venv/bin/activate
   ```

2. Install the required dependencies:
   ```
   pip install openai anthropic google-generativeai scikit-learn nltk matplotlib seaborn tabulate pandas sentence-transformers
   ```

3. Set up your API keys as environment variables:
   ```
   export OPENAI_API_KEY='your-openai-api-key-here'
   export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
   export GOOGLE_API_KEY='your-google-api-key-here'
   export OPENROUTER_API_KEY='your-openrouter-api-key-here'
   ```

## Usage

### Basic Usage

Run the CLI script with your desired prompt and number of iterations:

```
python cli.py --prompt "Your prompt here" --iterations 10
```

### Provider Selection

Choose a specific LLM provider:

```
python cli.py --prompt "Your prompt here" --provider openai --iterations 10
python cli.py --prompt "Your prompt here" --provider anthropic --iterations 10
python cli.py --prompt "Your prompt here" --provider gemini --iterations 10
python cli.py --prompt "Your prompt here" --provider openrouter --iterations 10
```

### Model Selection

Choose a specific model within a provider:

```
python cli.py --prompt "Your prompt here" --provider openai --model gpt-4o-mini --iterations 10
python cli.py --prompt "Your prompt here" --provider anthropic --model claude-3-haiku-20240307 --iterations 10
python cli.py --prompt "Your prompt here" --provider openrouter --model meta-llama/llama-3.1-70b-instruct --iterations 10
```

### List Available Providers and Models

```
python cli.py --list-providers
```

The script will generate responses, analyze them, and provide detailed output including:

- Top 3 responses with score breakdowns
- Key concepts and their frequencies
- Unique insights from the responses
- Hyperparameter analysis and trends
- Recommendations for further tuning

The script also generates several visualization charts:
- `score_comparison.png`: Comparison of top 3 responses' scores
- `word_frequency.png`: Bar chart of most frequent words
- `hyperparameter_impact.png`: Scatter plots showing the impact of each hyperparameter on the total score

## How It Works

HyperTune uses a combination of natural language processing techniques and machine learning to generate and analyze text responses:

1. It generates multiple responses using your chosen LLM provider with varying hyperparameters.
2. Each response is scored based on coherence, relevance to the prompt, and language complexity.
3. The tool then analyzes the responses collectively to identify common themes, unique insights, and the impact of different hyperparameters.
4. Finally, it provides a comprehensive report with visualizations to help understand the results.

![hyperparameter_impact](https://github.com/user-attachments/assets/73490098-4333-479f-9b3f-094944b34acd)

![word_frequency](https://github.com/user-attachments/assets/e0cd3271-78d3-406e-98a6-b39e75205dbf)

![score_comparison](https://github.com/user-attachments/assets/e6ad5ace-2632-404f-99c7-c50a88d328f7)


## Contributing

Contributions to HyperTune are welcome! Please feel free to submit a PR.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool interacts with various LLM provider APIs. The authors are not responsible for any misuse or for any offensive content that may be generated.
