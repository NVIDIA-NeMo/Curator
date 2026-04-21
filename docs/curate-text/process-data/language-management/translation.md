---
description: "Translate flat and structured text fields with Curator's translation pipeline, quality scoring, and backend integrations"
categories: ["how-to-guides"]
tags: ["translation", "multilingual", "faith", "text-quality", "llm", "nmt"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-translation)=

# Translation

Use NeMo Curator's translation package to translate flat text fields or structured records, such as chat conversations stored under `messages.*.content`.

The translation package is centered on `TranslationPipeline`, which composes segmentation, translation, reassembly, output formatting, and optional evaluation into a reusable text-processing stage.

## Capabilities

- Translate a single text field such as `text`, a nested field path such as `metadata.body`, or wildcard paths such as `messages.*.content`
- Preserve machine-readable payloads, including valid JSON objects and arrays, instead of sending them to the translation model
- Emit translated output in `replaced`, `raw`, or `both` modes
- Capture segment pairs for inspection or downstream evaluation
- Run FAITH scoring on translated text with `FaithEvalFilter`
- Score forward and reverse translation quality with `TextQualityMetricStage`

## Before You Start

- For `backend_type="llm"`, configure an OpenAI-compatible async client. See {ref}`synthetic-llm-client`.
- For non-LLM backends, use one of the built-in backend types: `google`, `aws`, or `nmt`.
- Input data is typically newline-delimited JSON with a `text` field or another field referenced through `text_field`.

## Basic Translation Pipeline

The example below reads JSONL files, translates `messages.*.content` from English to Hindi, preserves translated segment pairs, runs FAITH scoring, and writes the results back to JSONL.

```python
import os

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.translation import TranslationPipeline

client = AsyncOpenAIClient(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1",
    max_concurrent_requests=8,
)

pipeline = Pipeline(name="translate_chat_dataset")
pipeline.add_stage(JsonlReader(file_paths="input/*.jsonl"))
pipeline.add_stage(
    TranslationPipeline(
        client=client,
        model_name="openai/openai/gpt-5.1",
        generation_config=GenerationConfig(max_tokens=2048),
        source_lang="en",
        target_lang="hi",
        text_field="messages.*.content",
        output_field="translated_text",
        output_mode="both",
        preserve_segment_pairs=True,
        reconstruct_messages=True,
        enable_faith_eval=True,
        faith_threshold=2.5,
    )
)
pipeline.add_stage(JsonlWriter(path="translated/"))
results = pipeline.run()
```

## Structured Translation Behavior

Structured translation works directly on nested records.

- `text_field="messages.*.content"` extracts every message content string from the record
- valid JSON objects and arrays are treated as non-translatable content and are preserved verbatim
- when `output_mode="replaced"`, translated values are written back into the original field path
- when `output_mode="raw"` or `output_mode="both"`, Curator also emits translation metadata with whole-text and segmented mappings

This makes the pipeline suitable for chat-style records where natural-language turns should be translated but tool payloads should remain untouched.

## Segmentation and Output Control

`TranslationPipeline` exposes a few important controls:

- `segmentation_mode="coarse"` keeps line-level splitting with code-block awareness
- `segmentation_mode="fine"` uses sentence-level segmentation with structure preservation
- `min_segment_chars` lets you explicitly bypass segmentation for short text instead of relying on a hidden cutoff
- `preserve_segment_pairs=True` stores source/target segment pairs for debugging and evaluation
- `reconstruct_messages=True` rebuilds translated message lists for structured chat-style inputs

## Quality Evaluation

### FAITH Scoring

Enable FAITH scoring inside the translation pipeline when you want model-based adequacy checks on translated output:

```python
TranslationPipeline(
    client=client,
    model_name="openai/openai/gpt-5.1",
    source_lang="en",
    target_lang="de",
    text_field="text",
    enable_faith_eval=True,
    faith_threshold=2.5,
)
```

FAITH scores are merged into the output when `output_mode="raw"` or `output_mode="both"`.

### Round-Trip Metrics

Backtranslation is composed from a second translation pass with reversed languages, followed by `TextQualityMetricStage`:

```python
from nemo_curator.stages.text.translation import TextQualityMetricStage, TranslationPipeline

pipeline.add_stage(
    TranslationPipeline(
        client=client,
        model_name="openai/openai/gpt-5.1",
        source_lang="hi",
        target_lang="en",
        text_field="translated_text",
        output_field="backtranslated_text",
        output_mode="both",
    )
)
pipeline.add_stage(
    TextQualityMetricStage(
        reference_text_field="text",
        hypothesis_text_field="backtranslated_text",
        metrics=[
            {"type": "sacrebleu", "threshold": 20.0},
            {"type": "chrf", "threshold": 40.0},
        ],
    )
)
```

Supported metric types are:

- `sacrebleu`
- `chrf`
- `ter`

## Backend Selection

Use `backend_type` to switch between translation backends:

- `llm`: OpenAI-compatible async client
- `google`: Google translation backend
- `aws`: AWS translation backend
- `nmt`: NMT service backend

For non-LLM backends, pass backend-specific settings through `backend_config`.

## Notes

- The translation package is designed for pipeline execution. Avoid converting large datasets to Pandas on the driver just to orchestrate translation.
- For structured inputs, wildcard paths and nested paths are first-class inputs to the library. You do not need to flatten records manually before calling `TranslationPipeline`.
