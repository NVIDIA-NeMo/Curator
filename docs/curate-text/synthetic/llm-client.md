---
description: "Configure LLM clients for synthetic data generation with NVIDIA APIs or custom endpoints"
categories: ["how-to-guides"]
tags: ["llm-client", "openai", "nvidia-api", "configuration"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "text-only"
---

(synthetic-llm-client)=
# LLM Client Configuration

NeMo Curator's synthetic data generation uses OpenAI-compatible clients to communicate with LLM inference servers. This guide covers client configuration, performance tuning, and integration with various endpoints.

## Overview

Two client types are available:

- **`AsyncOpenAIClient`**: Recommended for high-throughput batch processing with concurrent requests
- **`OpenAIClient`**: Synchronous client for simpler use cases or debugging

For most SDG workloads, use `AsyncOpenAIClient` to maximize throughput.

## Basic Configuration

### NVIDIA API Endpoints

```python
from nemo_curator.models.client.openai_client import AsyncOpenAIClient

client = AsyncOpenAIClient(
    api_key="your-nvidia-api-key",  # Or use NVIDIA_API_KEY env var
    base_url="https://integrate.api.nvidia.com/v1",
    max_concurrent_requests=5,
)
```

### Environment Variables

Set your API key as an environment variable to avoid hardcoding credentials:

```bash
export NVIDIA_API_KEY="<your-nvapi-key-here>"
```

The underlying OpenAI client automatically uses the `OPENAI_API_KEY` environment variable if no `api_key` is provided. For NVIDIA APIs, explicitly pass the key:

```python
import os

client = AsyncOpenAIClient(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1",
)
```

## Generation Parameters

Configure LLM generation behavior using `GenerationConfig`:

```python
from nemo_curator.models.client.llm_client import GenerationConfig

config = GenerationConfig(
    max_tokens=2048,
    temperature=0.7,
    top_p=0.95,
    seed=42,  # For reproducibility
)
```

```{list-table} Generation Parameters
:header-rows: 1
:widths: 20 15 15 50

* - Parameter
  - Type
  - Default
  - Description
* - `max_tokens`
  - int
  - 2048
  - Maximum tokens to generate per request
* - `temperature`
  - float
  - 0.0
  - Sampling temperature (0.0-2.0). Higher values increase randomness
* - `top_p`
  - float
  - 0.95
  - Nucleus sampling parameter (0.0-1.0)
* - `top_k`
  - int
  - None
  - Top-k sampling (if supported by the endpoint)
* - `seed`
  - int
  - 0
  - Random seed for reproducibility
* - `stop`
  - str/list
  - None
  - Stop sequences to end generation
* - `stream`
  - bool
  - False
  - Enable streaming (not recommended for batch processing)
* - `n`
  - int
  - 1
  - Number of completions to generate per request
```

## Performance Tuning

### Concurrency vs. Parallelism

The `max_concurrent_requests` parameter controls how many API requests the client can have in-flight simultaneously. This interacts with Ray's distributed workers:

- **Client-level concurrency**: `max_concurrent_requests` limits concurrent API calls per worker
- **Worker-level parallelism**: Ray distributes tasks across multiple workers

```python
# For NVIDIA API endpoints with rate limits
client = AsyncOpenAIClient(
    base_url="https://integrate.api.nvidia.com/v1",
    max_concurrent_requests=3,  # Conservative for cloud APIs
)
```

### Retry Configuration

The client includes automatic retry with exponential backoff for transient errors:

```python
client = AsyncOpenAIClient(
    base_url="https://integrate.api.nvidia.com/v1",
    max_retries=3,        # Number of retry attempts
    base_delay=1.0,       # Base delay in seconds
    timeout=120,          # Request timeout
)
```

The retry logic handles:
- **Rate limit errors (429)**: Automatic backoff with jitter
- **Connection errors**: Retry with exponential delay
- **Transient failures**: Configurable retry attempts

## Using Other OpenAI-Compatible Endpoints

The `AsyncOpenAIClient` works with any OpenAI-compatible API endpoint. Simply configure the `base_url` and `api_key` parameters:

```python
# OpenAI API
client = AsyncOpenAIClient(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",  # Or set OPENAI_API_KEY env var
    max_concurrent_requests=5,
)

# Any OpenAI-compatible endpoint
client = AsyncOpenAIClient(
    base_url="http://your-endpoint/v1",
    api_key="your-api-key",
    max_concurrent_requests=5,
)
```

## Complete Example

```python
import os
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.qa_multilingual_synthetic import QAMultilingualSyntheticStage

# Configure client
client = AsyncOpenAIClient(
    api_key=os.environ.get("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    max_concurrent_requests=5,
    max_retries=3,
    base_delay=1.0,
)

# Configure generation
config = GenerationConfig(
    temperature=0.9,
    top_p=0.95,
    max_tokens=2048,
)

# Use in a pipeline stage
pipeline = Pipeline(name="sdg_example")
pipeline.add_stage(
    QAMultilingualSyntheticStage(
        prompt="Generate a Q&A pair about science in {language}.",
        languages=["English", "French", "German"],
        client=client,
        model_name="meta/llama-3.3-70b-instruct",
        num_samples=100,
        generation_config=config,
    )
)
```

## Troubleshooting

### Rate Limit Errors

If you encounter frequent 429 errors:
1. Reduce `max_concurrent_requests`
2. Increase `base_delay` for longer backoff
3. Consider using a local deployment for high-volume workloads

### Connection Timeouts

For slow networks or high-latency endpoints:
```python
client = AsyncOpenAIClient(
    base_url="...",
    timeout=300,  # Increase from default 120 seconds
)
```

---

## Next Steps

- {ref}`multilingual-qa-tutorial`: Generate multilingual Q&A pairs
- {ref}`nemotron-cc-overview`: Advanced text transformation pipelines

