# HyperTune

HyperTune is an advanced tool for optimizing and analyzing text generation using OpenAI's GPT models. It explores various hyperparameter combinations to produce high-quality responses to given prompts, and provides comprehensive analysis of the results.

## Features

- Generate multiple responses to a given prompt using different hyperparameter settings
- Score responses based on coherence, relevance, and complexity
- Analyze common themes and unique insights across responses
- Visualize the impact of hyperparameters on response quality
- Provide detailed explanations of scoring and recommendations for further tuning

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/geeknik/hypertune
   cd hypertune
   python -m venv venv && source venv/bin/activate
   ```

2. Install the required dependencies:
   ```
   pip install openai scikit-learn nltk matplotlib seaborn tabulate pandas
   ```

3. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

Run the CLI script with your desired prompt and number of iterations:

```
python cli.py --prompt "Your prompt here" --iterations 10
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

1. It generates multiple responses using OpenAI's GPT model with varying hyperparameters.
2. Each response is scored based on coherence, relevance to the prompt, and language complexity.
3. The tool then analyzes the responses collectively to identify common themes, unique insights, and the impact of different hyperparameters.
4. Finally, it provides a comprehensive report with visualizations to help understand the results.

![hyperparameter_impact](https://github.com/user-attachments/assets/73490098-4333-479f-9b3f-094944b34acd)

![word_frequency](https://github.com/user-attachments/assets/e0cd3271-78d3-406e-98a6-b39e75205dbf)

![score_comparison](https://github.com/user-attachments/assets/e6ad5ace-2632-404f-99c7-c50a88d328f7)


## Contributing

Contributions to HyperTune are welcome! Please feel free to submit a PR.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool interacts with OpenAI's GPT models. The authors are not responsible for any misuse or for any offensive content that may be generated.
