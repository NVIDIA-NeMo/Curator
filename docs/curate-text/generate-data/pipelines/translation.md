---
description: "Translate text documents using a multi-step LLM pipeline with reflection and improvement"
categories: ["how-to-guides"]
tags: ["translation", "llm", "synthetic-data", "reflection", "improvement"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-gen-data-pipelines-translation)=
# Translation Pipeline

This pipeline translates text documents using a multi-step process with large language models (LLMs), including initial translation, reflection, and improvement. It leverages the NeMo Curator framework and supports both function-based and YAML-based configuration for flexible workflows.

## Before You Start

- **LLM Client Setup**: The `TranslationDataGenerator` requires an `OpenAIClient` instance to interface with language models. See the [LLM services documentation](text-generate-data-connect-service) for details on configuring your client and model provider.

---

## Setup Steps

### Set up the LLM Client

Configure your LLM client (example with OpenAI):

```python
from openai import OpenAI

openai_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="<insert API key>"
)
```

### Create the NeMo Curator Client Wrapper

Wrap the client with NeMo Curator's client wrapper:

```python
from nemo_curator import OpenAIClient

client = OpenAIClient(openai_client)
```

### Initialize the Translation Generator

Create the `TranslationDataGenerator` instance:

```python
from nemo_curator.synthetic.translate import TranslationDataGenerator

generator = TranslationDataGenerator(
    base_url="http://localhost:8000/v1",
    api_key="<insert API key>",
    init_translate_model="gpt-oss:latest",
    reflection_model="gpt-oss:latest",
    improvement_model="gpt-oss:latest",
    hf_tokenizer="openai/gpt-oss-20b",
    hf_token="",
    temperature=1.0,
    top_p=1.0,
    max_tokens=8192,
    max_token_per_chunk=5000,
    source_lang="English",
    target_lang="Traditional Chinese",
    country="Taiwan"
)
```

### YAML-Based Configuration

You can also configure the generator using a YAML file:

```python
# config/translation_config.yaml
# See the provided example in the repository

generator_yaml = TranslationDataGenerator.from_yaml("config/translation_config.yaml")
```

---

## Translation Workflow

### Translate a Single Text

```python
text = "Once upon a time, there were three little pig brothers..."
translations = generator.generate(text)
print(generator.parse_response(translations))
```

### Translate Using YAML Configuration

```python
translations = generator_yaml.generate_from_yaml("config/translation_config.yaml", text)
print(translations)
```

### Batch Translation with DataFrames

Efficiently translate multiple texts in a pandas DataFrame:

```python
import pandas as pd

df = pd.DataFrame({
    "text": [
        "Once upon a time, there were three little pig brothers...",
        "The quick brown fox jumps over the lazy dog."
    ]
})

df_translated = generator_yaml.generate_from_dataframe(df, text_column="text", batch_size=16)
print(df_translated.head())
```

### Asynchronous Batch Translation

For large-scale translation tasks, use the async pipeline:

```python
import asyncio

async def async_translate():
    df_translated = await generator_yaml.async_generate_from_dataframe(df, text_column="text", batch_size=16)
    print("[Async]", df_translated.head())

asyncio.run(async_translate())
```

---

## Advanced Configuration

Customize translation with model parameters and prompts:

```python
# Adjust temperature, top_p, and max_tokens for your use case
custom_generator = TranslationDataGenerator(
    ...,
    temperature=0.7,
    top_p=0.95,
    max_tokens=2048,
    ...
)
```

---

## Pipeline Steps Explained

1. **Initial Translation**: The input text is translated using the specified LLM model.
2. **Reflection**: The initial translation is reviewed and refined, optionally with country-specific context.
3. **Improvement**: The translation is finalized using feedback from the reflection step.

This multi-step approach improves translation quality and contextual accuracy.

---

## Example YAML Configuration

```yaml
base_url: "http://localhost:8000/v1"
api_key: ""
init_translate_model: "gpt-oss:latest"
reflection_model: "gpt-oss:latest"
improvement_model: "gpt-oss:latest"
hf_tokenizer: "openai/gpt-oss-20b"
hf_token: ""
max_token_per_chunk: 5000
temperature: 1.0
top_p: 1.0
max_tokens: 8192
source_lang: "English"
target_lang: "Traditional Chinese"
country: "Taiwan"
```

---

## Comprehensive Translation Workflow

Combine translation with other NeMo Curator synthetic data generation methods for advanced document processing.

```python
def process_document_comprehensive(text, generator):
    """Translate and further process a document."""
    translation = generator.generate(text)
    # You can add additional synthetic data steps here
    return {
        "original_text": text,
        "translated_text": generator.parse_response(translation)
    }

sample_text = "The Amazon rainforest covers 2.1 million square miles across nine countries in South America."
result = process_document_comprehensive(sample_text, generator)
print(result)
```

---

## References

- [NeMo Curator Documentation](https://github.com/NVIDIA/NeMo-Curator)
- [Translation Agent Reference](https://github.com/andrewyng/translation-agent)

---
