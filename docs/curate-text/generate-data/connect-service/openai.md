---
description: "Connect to hosted model endpoints using OpenAI API format including build.nvidia.com and other compatible services"
categories: ["how-to-guides"]
tags: ["openai", "api", "hosted-models", "nvidia-build", "rate-limits"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "text-only"
---

(text-generate-data-connect-service-openai)=
# OpenAI Compatible Services

Connect to hosted model endpoints that implement the OpenAI API format, such as [build.nvidia.com](https://build.nvidia.com/explore/discover).

## Before You Start

### Rate Limits

OpenAI API compatible services typically have rate limits on:

- Number of requests per minute
- Number of tokens per minute
- Total tokens per request

For high-volume data generation, consider using [NeMo Deploy](nemo-deploy) to host your own models without rate limits.

---

## Usage

The following code demonstrates how to connect to build.nvidia.com to query Mixtral 8x7B Instruct:

```python
from openai import OpenAI
from nemo_curator import OpenAIClient

openai_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="<insert NVIDIA API key>",
)
client = OpenAIClient(openai_client)
responses = client.query_model(
    model="mistralai/mixtral-8x7b-instruct-v0.1",
    messages=[
        {
            "role": "user",
            "content": "Write a limerick about the wonders of GPU computing.",
        }
    ],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)
print(responses[0])
# Output:
# A GPU with numbers in flight, Brings joy to programmers late at night.
# With parallel delight, Solving problems, so bright,
# In the realm of computing, it's quite a sight!
```
