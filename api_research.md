# LLM Provider API Research

## OpenAI (Current Implementation)
- Client: `from openai import OpenAI`
- API: `client.chat.completions.create()`
- Model: "gpt-4o"
- Parameters: temperature, top_p, frequency_penalty, presence_penalty
- Authentication: API key via environment variable

## Anthropic Claude
- Documentation: https://platform.claude.com/docs/en/api/overview
- Client: `from anthropic import Anthropic`
- API: `client.messages.create()`
- Models: "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", etc.
- Parameters: temperature, top_p, max_tokens
- Authentication: API key via environment variable
- Request format:
  ```python
  response = client.messages.create(
      model="claude-3-5-sonnet-20241022",
      max_tokens=1024,
      temperature=temperature,
      messages=[
          {"role": "user", "content": prompt}
      ]
  )
  ```

## Google Gemini
- Documentation: https://ai.google.dev/gemini-api/docs
- Client: `import google.generativeai as genai`
- API: `genai.GenerativeModel(model_name).generate_content()`
- Models: "gemini-1.5-pro", "gemini-1.5-flash", etc.
- Parameters: temperature, top_p, top_k, max_output_tokens
- Authentication: API key via environment variable
- Request format:
  ```python
  model = genai.GenerativeModel('gemini-1.5-pro')
  response = model.generate_content(
      prompt,
      generation_config=genai.types.GenerationConfig(
          temperature=temperature,
          top_p=top_p,
          max_output_tokens=max_tokens,
      )
  )
  ```

## OpenRouter
- Documentation: https://openrouter.ai/docs/quickstart
- Client: `from openai import OpenAI` (uses OpenAI-compatible API)
- API: `client.chat.completions.create()`
- Models: Hundreds of models available (e.g., "anthropic/claude-3.5-sonnet", "google/gemini-pro-1.5")
- Parameters: temperature, top_p, max_tokens, frequency_penalty, presence_penalty
- Authentication: API key via environment variable
- Base URL: "https://openrouter.ai/api/v1"
- Request format:
  ```python
  client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key="your-openrouter-api-key"
  )
  response = client.chat.completions.create(
      model="anthropic/claude-3.5-sonnet",
      messages=[{"role": "user", "content": prompt}],
      temperature=temperature,
      top_p=top_p,
      max_tokens=max_tokens
  )
  ```

## Parameter Mapping
| Parameter | OpenAI | Anthropic | Google Gemini | OpenRouter |
|-----------|--------|-----------|---------------|------------|
| temperature | ✓ | ✓ | ✓ | ✓ |
| top_p | ✓ | ✓ | ✓ | ✓ |
| frequency_penalty | ✓ | ✗ | ✗ | ✓ |
| presence_penalty | ✓ | ✗ | ✗ | ✓ |
| max_tokens | ✓ | ✓ (as max_tokens) | ✓ (as max_output_tokens) | ✓ |
| top_k | ✗ | ✗ | ✓ | ✗ |