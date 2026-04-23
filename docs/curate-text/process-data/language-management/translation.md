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

The translation package is centered on `TranslationStage`, which composes segmentation, translation, reassembly, output formatting, and optional evaluation into a reusable text-processing stage.

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
from nemo_curator.stages.text.translation import TranslationStage

client = AsyncOpenAIClient(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1",
    max_concurrent_requests=8,
)

pipeline = Pipeline(name="translate_chat_dataset")
pipeline.add_stage(JsonlReader(file_paths="input/*.jsonl"))
pipeline.add_stage(
    TranslationStage(
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

- Setting `text_field="messages.*.content"` extracts every message content string from the record
- Valid JSON objects and arrays are treated as non-translatable content and are preserved verbatim
- When `output_mode="replaced"`, translated values are written back into the original field path
- When `output_mode="raw"` or `output_mode="both"`, Curator also emits translation metadata with whole-text and segmented mappings

This makes the pipeline suitable for chat-style records where natural-language turns should be translated but tool payloads should remain untouched.

## Segmentation and Output Control

`TranslationStage` exposes a few important controls:

- `segmentation_mode="coarse"` keeps line-level splitting with code-block awareness
- `segmentation_mode="fine"` uses sentence-level segmentation with structure preservation
- `min_segment_chars` lets you explicitly bypass segmentation for short text rather than relying on an implicit cutoff
- `preserve_segment_pairs=True` stores source/target segment pairs for debugging and evaluation
- `reconstruct_messages=True` rebuilds translated message lists for structured chat-style inputs

## DocumentBatch Walkthrough

`TranslationStage` operates on a `DocumentBatch`, which is a task wrapper around a pandas DataFrame or Arrow table.

The wrapper fields stay mostly constant across stages:

- `task_id`
- `dataset_name`
- `_stage_perf`
- `_metadata`

The main thing that changes is the DataFrame inside `DocumentBatch.data`.

### Worked Example

Assume a single input row and this pipeline configuration:

```python
TranslationStage(
    text_field="text",
    source_lang="en",
    target_lang="hi",
    segmentation_mode="coarse",
    preserve_segment_pairs=True,
    enable_faith_eval=True,
    segment_level=True,
    output_mode="raw",
    merge_scores=True,
)
```

Input row:

~~~text
id | text
7  | Explain grouped-query attention.
   | {"tool":"search","query":"GQA"}
   | ```python
   | print("hello")
   | ```
   | It reduces KV-cache memory.
~~~

### 1. SegmentationStage

The batch changes from one row per document to one row per translatable segment.

New columns:

- `_seg_segments`
- `_seg_metadata`
- `_seg_doc_id`

Example output:

```text
id | _seg_segments
7  | Explain grouped-query attention.
7  | It reduces KV-cache memory.
```

Important details:

- Valid JSON payloads and fenced code blocks are not emitted as translatable segment rows.
- `_seg_doc_id` ties all segment rows back to the same source document.
- `_seg_metadata` is a JSON reconstruction template duplicated across the segment rows for that document.

For coarse segmentation, the metadata contains the original non-translated lines plus placeholders for translatable lines. For fine segmentation, it stores sentence-like units and their separators.

### 2. SegmentTranslationStage

The batch still has one row per segment, but each segment now gets its translated text and per-segment runtime/error data.

New columns:

- `_translated`
- `_translation_time`
- `_translation_error`

Example output:

```text
id | _seg_segments                    | _translated
7  | Explain grouped-query attention. | समूहित-क्वेरी अटेंशन समझाइए।
7  | It reduces KV-cache memory.      | यह KV-cache मेमोरी को कम करता है।
```

### 3. CaptureSegmentPairsStage

When `preserve_segment_pairs=True`, the batch gains:

- `_seg_translation_pairs`

This is a JSON list of `{"src": ..., "tgt": ...}` pairs for the document. The first segment row for a document carries the full list, and later segment rows carry `[]`. This keeps the information available for later scoring and metadata construction without duplicating it on every surviving row after reassembly.

### 4. ReassemblyStage

The batch collapses back to one row per original document by grouping on `_seg_doc_id`.

Removed internal columns:

- `_seg_segments`
- `_seg_metadata`
- `_seg_doc_id`
- `_translated`
- `_translation_time`
- `_translation_error`

Added document-level columns:

- `translated_text`
- `translation_time`
- `translation_errors`
- `_translation_map`
- `_segmented_translation_map`

Example output:

~~~text
id | translated_text
7  | समूहित-क्वेरी अटेंशन समझाइए।
   | {"tool":"search","query":"GQA"}
   | ```python
   | print("hello")
   | ```
   | यह KV-cache मेमोरी को कम करता है।
~~~

Important details:

- `translation_time` is the sum of the segment-level translation times.
- `translation_errors` joins any non-empty segment errors.
- `_translation_map` and `_segmented_translation_map` are helper columns used later to build `translation_metadata`.
- For structured fields such as `messages.*.content`, reassembly writes translations back into the nested structure instead of only returning a flat string.

### 5. FaithEvalFilter

When `enable_faith_eval=True`, FAITH scores are added to the document-level batch.

New columns:

- `faith_fluency`
- `faith_accuracy`
- `faith_idiomaticity`
- `faith_terminology`
- `faith_handling_of_format`
- `faith_avg`
- `faith_parse_failed`
- `faith_segment_scores` when `segment_level=True`

If filtering is enabled, rows below `faith_threshold` may be dropped here.

### 6. FormatTranslationOutputStage

When `output_mode="raw"` or `output_mode="both"`, this stage builds:

- `translation_metadata`

`translation_metadata` contains:

- `target_lang`
- `translation`
- `segmented_translation`

When `output_mode="raw"`, the final translated text column is dropped after metadata is constructed. This is useful when downstream consumers want metadata-rich output without replacing the source text field.

This stage also drops internal helper columns such as:

- `_seg_translation_pairs`
- `_translation_map`
- `_segmented_translation_map`
- `_faith_source_text`
- `_faith_translated_text`

### 7. MergeFaithScoresStage

When `merge_scores=True`, FAITH scores are merged into the existing `translation_metadata` JSON under `faith_scores`.

At that point, the final row contains:

- the original source columns
- translation runtime and error columns
- FAITH score columns
- `translation_metadata`

### Skip/Restore Path

When `skip_translated=True`, the pipeline inserts two additional stages:

- `SkipExistingTranslationsStage`
- `RestoreSkippedRowsStage`

`SkipExistingTranslationsStage` removes rows that already have a non-empty translation column from the DataFrame and stores them temporarily in `DocumentBatch._metadata["_skipped_rows_state"]`.

`RestoreSkippedRowsStage` restores those rows later, fills in any missing score/metadata columns with defaults, and sorts them back into the original row order.

## Quality Evaluation

### FAITH Scoring

Enable FAITH scoring inside the translation pipeline when you want model-based adequacy checks on translated output:

```python
TranslationStage(
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

Backtranslation uses a second translation pass with reversed languages, followed by `TextQualityMetricStage`:

```python
from nemo_curator.stages.text.translation import TextQualityMetricStage, TranslationStage

pipeline.add_stage(
    TranslationStage(
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

- The translation package is designed for pipeline execution. Avoid converting large datasets to pandas on the driver just to orchestrate translation.
- For structured inputs, wildcard paths and nested paths are first-class inputs to the library. You do not need to flatten records manually before calling `TranslationStage`.
