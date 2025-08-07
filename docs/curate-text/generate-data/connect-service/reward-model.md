---
description: "Query reward models to score conversations and filter datasets using Nemotron-4 340B and other scoring models"
categories: ["how-to-guides"]
tags: ["reward-models", "scoring", "quality-assessment", "nemotron", "conversation-scoring"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-generate-data-connect-service-reward-models)=
# Reward Models

Reward models score conversations between users and assistants. Instead of generating text responses, they return scores for different quality categories, which can be used to filter datasets.

## Supported Reward Models

NeMo Curator supports several reward models for scoring conversations and filtering datasets. The table below summarizes the supported models, their architecture, and scoring categories:

```{list-table}
:header-rows: 1
:widths: 20 20 30 30
* - Model Name
  - Architecture
  - Scoring Categories
  - Best For
* - nvidia/nemotron-4-340b-reward
  - Nemotron-4 340B
  - Helpfulness, Correctness, Coherence, Complexity, Verbosity
  - Comprehensive conversation quality assessment
* - nvidia/nemotron-3-34b-reward
  - Nemotron-3 34B
  - Helpfulness, Correctness, Coherence
  - Efficient quality assessment for basic filtering
```

## Usage

Here's how to query the Nemotron-4 340b reward model:

```python
from openai import OpenAI
from nemo_curator import OpenAIClient

openai_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="<insert NVIDIA API key>",
)
client = OpenAIClient(openai_client)

model = "nvidia/nemotron-4-340b-reward"

messages = [
    {"role": "user", "content": "I am going to Paris, what should I see?"},
    {
        "role": "assistant",
        "content": "Ah, Paris, the City of Light! There are so many amazing things to see and do in this beautiful city ...",
    },
]

rewards = client.query_reward_model(messages=messages, model=model)
print(rewards)
# {
# "helpfulness": 1.6171875
# "correctness": 1.6484375
# "coherence": 3.3125
# "complexity": 0.546875
# "verbosity": 0.515625
# }
```

## Score Categories

The Nemotron-4 340B reward model provides scores for:

- **Helpfulness**: How well the response addresses the user's needs
- **Correctness**: Factual accuracy of the response
- **Coherence**: Logical flow and consistency of the response
- **Complexity**: Sophistication level of the language and concepts
- **Verbosity**: Appropriate length and detail level

For detailed information about these categories, see the [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704v1). 