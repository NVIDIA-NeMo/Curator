(text-gen-data-pipelines-distillation)=
# Distillation Pipeline

This pipeline distills documents to make them more concise, removing redundancy and focusing on key information. The distillation process uses large language models to extract essential content while maintaining the core meaning and important details.

## Before You Start

- **LLM Client Setup**: The `NemotronCCGenerator` requires an `LLMClient` instance to interface with language models. Refer to the [LLM services documentation](text-generate-data-connect-service) for details on configuring your client with specific model providers.

---

## Setup Steps

### Set up the LLM Client

Configure your LLM client (example with OpenAI):

```python
from openai import OpenAI

openai_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="<insert NVIDIA API key>"
)
```

### Create the NeMo Curator Client Wrapper

Wrap the client with NeMo Curator's client wrapper:

```python
from nemo_curator import OpenAIClient

client = OpenAIClient(openai_client)
```

### Initialize the Generator

Create the NemotronCCGenerator instance:

```python
from nemo_curator.synthetic import NemotronCCGenerator

generator = NemotronCCGenerator(client)
```

### Configure Generation Parameters

Set up your model and generation parameters:

```python
model = "nv-mistralai/mistral-nemo-12b-instruct"
document = "The moon is bright. It shines at night. The moon provides light during dark hours. It illuminates the landscape when the sun is not visible."
model_kwargs = {
    "temperature": 0.5,
    "top_p": 0.9,
    "max_tokens": 1600,
}
```

### Generate Distilled Content

Use the generator to distill your document:

```python
responses = generator.distill(
    document=document,
    model=model,
    model_kwargs=model_kwargs
)

print(responses[0])
# Output:
# The moon is bright at night, providing light when the sun is not visible.
```

## Advanced Configuration

Customize the distillation process with additional parameters:

```python
# Use custom prompts and system messages
from nemo_curator.synthetic.prompts import (
    DISTILL_PROMPT_TEMPLATE,
    NEMOTRON_CC_DISTILL_SYSTEM_PROMPT
)

# Configure advanced model parameters
advanced_model_kwargs = {
    "temperature": 0.3,  # Lower temperature for more focused distillation
    "top_p": 0.8,
    "max_tokens": 1200,
    "seed": 42  # For reproducible results
}

# Custom prompt parameters (optional)
custom_prompt_kwargs = {
    "additional_instructions": "Focus on factual information only"
}

responses = generator.distill(
    document=document,
    model=model,
    prompt_template=DISTILL_PROMPT_TEMPLATE,
    system_prompt=NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
    prompt_kwargs=custom_prompt_kwargs,
    model_kwargs=advanced_model_kwargs
)

print(responses[0])
```

## Batch Processing

For processing multiple documents efficiently:

```python
documents = [
    "Long document text one with lots of redundant information...",
    "Another verbose document that needs to be condensed...",
    "Third document with excessive detail that can be simplified..."
]

distilled_results = []
for doc in documents:
    responses = generator.distill(
        document=doc,
        model=model,
        model_kwargs=model_kwargs
    )
    distilled_results.append(responses[0])

# Print results
for i, result in enumerate(distilled_results):
    print(f"Document {i+1} distilled: {result}")
``` 