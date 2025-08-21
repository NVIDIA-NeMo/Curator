---
description: "Generate synthetic data efficiently using asynchronous pipelines for maximum throughput with rate-limited LLM APIs"
categories: ["how-to-guides"]
tags: ["async", "parallel", "performance", "rate-limits", "efficiency", "concurrent-requests"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-gen-data-pipelines-async)=
# Asynchronous Generation Pipeline

This pipeline provides asynchronous alternatives to the synchronous generation pipelines, allowing for more efficient data generation by sending multiple requests in parallel.

## Setup Steps

### Set up the Async OpenAI Client

Configure the OpenAI client to point to NVIDIA's API endpoint:

```python
from openai import AsyncOpenAI

openai_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1", 
    api_key="<insert NVIDIA API key>"
)
```

### Create the NeMo Curator Client Wrapper

Wrap the OpenAI client with NeMo Curator's async client:

```python
from nemo_curator import AsyncOpenAIClient

client = AsyncOpenAIClient(openai_client)
```

### Initialize the Async Generator

Create the generator with concurrency control:

```python
from nemo_curator.synthetic import AsyncNemotronGenerator

generator = AsyncNemotronGenerator(client, max_concurrent_requests=10)
```

### Configure Generation Parameters

Set up your model and generation parameters:

```python
n_macro_topics = 20
model = "mistralai/mixtral-8x7b-instruct-v0.1"
model_kwargs = {
    "temperature": 0.2,
    "top_p": 0.7,
    "max_tokens": 1024,
}
```

### Generate Data Asynchronously

Use await to call the generation method:

```python
responses = await generator.generate_macro_topics(
    n_macro_topics=n_macro_topics, model=model, model_kwargs=model_kwargs
)

print(responses[0])
# Output:
# 1. Climate Change and Sustainable Living
# 2. Space Exploration and the Universe
# ...
```
