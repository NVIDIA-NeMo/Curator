---
description: "Generate multilingual Q&A pairs using LLMs with NeMo Curator's synthetic data pipeline"
categories: ["tutorials"]
tags: ["multilingual", "qa-generation", "synthetic-data", "quickstart"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "text-only"
---

(multilingual-qa-tutorial)=
# Generate Multilingual Q&A Data

This guide shows how to generate synthetic Q&A pairs across multiple languages using NeMo Curator's `QAMultilingualSyntheticStage`. You'll learn to configure an LLM client, create a generation pipeline, and optionally filter the output.


## What You'll Build

A pipeline that:
1. Generates Q&A pairs in multiple languages using an LLM
2. Optionally filters results by language
3. Writes output to JSONL format

## Prerequisites

- **NVIDIA API Key**: Obtain from [NVIDIA Build](https://build.nvidia.com/settings/api-keys)
- **NeMo Curator**: Installed with text extras

```bash
export NVIDIA_API_KEY="<your-nvapi-key-here>"
```

## Quick Start

```python
import os
from nemo_curator.core.client import RayClient
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.qa_multilingual_synthetic import QAMultilingualSyntheticStage
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter

# Initialize Ray
client = RayClient(include_dashboard=False)
client.start()

# Create LLM client
llm_client = AsyncOpenAIClient(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1",
    max_concurrent_requests=5,
)

# Create pipeline
pipeline = Pipeline(name="multilingual_qa")

# Add synthetic generation stage
pipeline.add_stage(
    QAMultilingualSyntheticStage(
        prompt="Generate a Q&A pair about science in {language}.",
        languages=["English", "French", "German", "Spanish"],
        client=llm_client,
        model_name="meta/llama-3.3-70b-instruct",
        num_samples=50,
        generation_config=GenerationConfig(temperature=0.9),
    )
)

# Write output
pipeline.add_stage(JsonlWriter(path="./synthetic_qa/"))

# Run pipeline
results = pipeline.run()

client.stop()
```

## Step-by-Step Guide

### Step 1: Configure the LLM Client

The `AsyncOpenAIClient` enables concurrent API requests for efficient batch generation:

```python
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.models.client.llm_client import GenerationConfig

llm_client = AsyncOpenAIClient(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1",
    max_concurrent_requests=5,  # Adjust based on rate limits
    max_retries=3,              # Retry on transient failures
    base_delay=1.0,             # Backoff delay in seconds
)

# Configure generation parameters
generation_config = GenerationConfig(
    temperature=0.9,   # Higher for more diverse outputs (default: 0.0)
    top_p=0.95,
    max_tokens=2048,
    seed=None,         # Set to None for non-deterministic (default: 0 for reproducibility)
)
```

### Step 2: Define the Prompt Template

The prompt template must include a `{language}` placeholder. The stage randomly selects a language for each sample:

```python
# Simple Q&A prompt
prompt = "Generate a Q&A pair about science in {language}."

# Structured prompt with language prefixes
prompt = """
Generate a short question and a short answer in the general science domain in {language}.
Begin with the language name using the 2-letter code in square brackets,
for example, [EN] for English, [FR] for French, [DE] for German.
"""
```

### Step 3: Create the Pipeline

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.qa_multilingual_synthetic import QAMultilingualSyntheticStage

pipeline = Pipeline(
    name="multilingual_qa_generation",
    description="Generate synthetic Q&A pairs in multiple languages",
)

pipeline.add_stage(
    QAMultilingualSyntheticStage(
        prompt=prompt,
        languages=["English", "French", "German", "Spanish", "Italian"],
        client=llm_client,
        model_name="meta/llama-3.3-70b-instruct",
        num_samples=100,
        generation_config=generation_config,
    )
)
```

### Step 4: Add Language Filtering (Optional)

If your prompt includes language prefixes, you can filter to keep only specific languages:

```python
from nemo_curator.stages.text.filters import DocumentFilter, ScoreFilter


class BeginsWithLanguageFilter(DocumentFilter):
    """Filter documents based on language prefix codes."""

    def __init__(self, languages: list[str]):
        super().__init__()
        self._name = "begins_with_language_filter"
        self.languages = languages

    def score_document(self, text: str) -> float:
        if not self.languages:
            return 1.0
        return 1.0 if text.startswith(tuple(self.languages)) else 0.0

    def keep_document(self, score: float) -> bool:
        return score == 1.0


# Add filter to keep only English outputs
pipeline.add_stage(
    ScoreFilter(
        BeginsWithLanguageFilter(languages=["[EN]"]),
        text_field="text",
    ),
)
```

### Step 5: Configure Output

Write results to JSONL or Parquet format:

```python
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter

# JSONL output
pipeline.add_stage(JsonlWriter(path="./output/synthetic_qa/"))

# Or Parquet output
# pipeline.add_stage(ParquetWriter(path="./output/synthetic_qa/"))
```

### Step 6: Run the Pipeline

```python
from nemo_curator.core.client import RayClient

# Initialize Ray
client = RayClient(include_dashboard=False)
client.start()

# Execute pipeline
print(pipeline.describe())
results = pipeline.run()

# Print results summary
if results:
    for result in results:
        if hasattr(result, "data") and result.data:
            for file_path in result.data:
                print(f"Generated: {file_path}")

client.stop()
```

## CLI Usage

The tutorial script supports command-line arguments:

```bash
cd tutorials/synthetic

# Basic usage
python synthetic_data_generation_example.py --num-samples 50

# Custom languages and model
python synthetic_data_generation_example.py \
    --num-samples 100 \
    --languages English French German \
    --model-name meta/llama-3.3-70b-instruct \
    --temperature 0.9

# Skip language filtering
python synthetic_data_generation_example.py \
    --num-samples 50 \
    --no-filter-languages
```

### Available Arguments

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Argument
  - Default
  - Description
* - `--api-key`
  - env var
  - NVIDIA API key (or set NVIDIA_API_KEY)
* - `--base-url`
  - NVIDIA API
  - Base URL for the API endpoint
* - `--model-name`
  - meta/llama-3.3-70b-instruct
  - Model to use for generation
* - `--languages`
  - English, French, German, Spanish, Italian
  - Languages to generate Q&A pairs for (use full names)
* - `--num-samples`
  - 100
  - Number of samples to generate
* - `--temperature`
  - 0.9
  - Sampling temperature
* - `--output-path`
  - ./synthetic_output
  - Output directory
* - `--no-filter-languages`
  - False
  - Disable language filtering
```

## Sample Output

Generated documents contain a `text` field with the LLM response:

```json
{"text": "[EN] Question: What causes ocean tides? Answer: Ocean tides are primarily caused by the gravitational pull of the Moon and Sun on Earth's water bodies."}
{"text": "[FR] Question: Qu'est-ce que la photosynthèse? Answer: La photosynthèse est le processus par lequel les plantes convertissent la lumière du soleil en énergie."}
{"text": "[DE] Question: Was ist der größte Planet in unserem Sonnensystem? Answer: Jupiter ist der größte Planet in unserem Sonnensystem."}
```

## Tips for Diverse Output

1. **Use higher temperature** (0.7-1.0) for more varied outputs
2. **Avoid fixed seeds** for non-deterministic generation
3. **Include clear instructions** in the prompt for consistent formatting
4. **Filter post-generation** to ensure quality standards

---

## Next Steps

- {ref}`synthetic-llm-client`: Advanced client configuration and performance tuning
- {ref}`nemotron-cc-overview`: Advanced pipelines for text transformation and knowledge extraction

