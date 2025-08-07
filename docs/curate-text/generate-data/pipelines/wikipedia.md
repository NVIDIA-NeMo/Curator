---
description: "Rewrite documents into Wikipedia-style content improving line spacing, punctuation, and scholarly tone"
categories: ["how-to-guides"]
tags: ["wikipedia", "rewrite", "style-transformation", "scholarly-tone", "formatting", "nemotron-cc"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-gen-data-pipelines-wikipedia)=
# Wikipedia Style Rewrite Pipeline

This pipeline rewrites documents into a style similar to Wikipedia, improving line spacing, punctuation, and scholarly tone. The pipeline uses language models to transform low-quality text into well-formatted, encyclopedia-style content that's more suitable for training datasets.

(text-gen-data-pipelines-wikipedia-before-start)=
## Before You Start

- **LLM Client Setup**: The `NemotronCCGenerator` requires an `LLMClient` instance to interface with language models. Refer to the {ref}`LLM services documentation <text-generate-data-connect-service>` for details on configuring your client with specific model providers.

---

(text-gen-data-pipelines-wikipedia-setup)=
## Setup Steps

(text-gen-data-pipelines-wikipedia-setup-client)=
### Set up the LLM Client

Configure your LLM client (example with OpenAI):

```python
from openai import OpenAI

openai_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="<insert NVIDIA API key>"
)
```

(text-gen-data-pipelines-wikipedia-setup-wrapper)=
### Create the NeMo Curator Client Wrapper

Wrap the client with NeMo Curator's client wrapper:

```python
from nemo_curator import OpenAIClient

client = OpenAIClient(openai_client)
```

(text-gen-data-pipelines-wikipedia-setup-generator)=
### Initialize the Generator

Create the NemotronCCGenerator instance:

```python
from nemo_curator.synthetic import NemotronCCGenerator

generator = NemotronCCGenerator(client)
```

(text-gen-data-pipelines-wikipedia-setup-params)=
### Configure Generation Parameters

Set up your model and generation parameters:

```python
model = "nv-mistralai/mistral-nemo-12b-instruct"
model_kwargs = {
    "temperature": 0.5,
    "top_p": 0.9,
    "max_tokens": 512,
}
```

(text-gen-data-pipelines-wikipedia-usage)=
### Rewrite Documents to Wikipedia Style

Use the generator to transform text into Wikipedia-style content:

:::{dropdown} Python Example
:icon: code-square

```python
document = "The moon is bright. It shines at night."

responses = generator.rewrite_to_wikipedia_style(
    document=document, 
    model=model, 
    model_kwargs=model_kwargs
)

print(responses[0])
# Output:
# The lunar surface has a high albedo, which means it reflects a significant amount of sunlight.
```
:::

:::{note}
The output shown is illustrative. Actual outputs will vary based on the input text and model parameters.
::: 