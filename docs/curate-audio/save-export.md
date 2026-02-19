---
description: "Export processed audio data and transcriptions in formats optimized for ASR training and multimodal applications"
categories: ["data-export"]
tags: ["output-formats", "manifests", "jsonl", "metadata", "asr-training"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "audio-only"
---


(audio-save-export)=

# Save & Export Audio Data

Export processed audio data and transcriptions in formats optimized for ASR model training, speech-to-text applications, and downstream analysis workflows.

## Overview

After processing your audio data through NeMo Curator's pipeline, export the results in standardized formats suitable for:

- **ASR Model Training**: JSONL manifests with audio file paths and transcriptions for NeMo ASR training
- **Quality Analysis**: Datasets with WER, duration, and other metrics for evaluation
- **Dataset Distribution**: Curated audio datasets with metadata for sharing or archiving
- **Downstream Processing**: Structured data for integration with other tools and workflows

## Output Formats

NeMo Curator's audio curation pipeline supports JSONL (JSON Lines) format, the standard for NeMo ASR training and audio dataset distribution.

### JSONL Manifests

The primary output format for audio curation is JSONL (JSON Lines), where each line represents one audio sample:

```json
{"audio_filepath": "/data/audio/sample_001.wav", "text": "hello world", "pred_text": "hello world", "wer": 0.0, "duration": 2.1}
{"audio_filepath": "/data/audio/sample_002.wav", "text": "good morning", "pred_text": "good morning", "wer": 0.0, "duration": 1.8}
```
**Format characteristics:**
- One JSON object per line (newline-delimited)
- Human-readable and machine-parseable
- Compatible with NeMo ASR training pipelines
- Easy to process with standard tools (jq, pandas, etc.)

### Metadata Fields

Standard fields included in audio manifests:

| Field | Type | Description |
|-------|------|-------------|
| `audio_filepath` | string | Absolute path to audio file |
| `text` | string | Ground truth transcription |
| `pred_text` | string | ASR model prediction |
| `wer` | float | Word Error Rate percentage |
| `duration` | float | Audio duration in seconds |
| `language` | string | Language identifier (optional) |

:::{note}
Fields marked as "optional" depend on which processing stages you included in your pipeline. At minimum, manifests require `audio_filepath` and either `text` (for ground truth) or `pred_text` (for ASR predictions).
:::

## Export Configuration

::::{tab-set}

:::{tab-item} Basic Export Setup

To export audio curation results, you must first convert `AudioBatch` to `DocumentBatch` format, then use `JsonlWriter`:

```python
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage

# Convert AudioBatch to DocumentBatch for text writer
pipeline.add_stage(AudioToDocumentStage())

# Configure JSONL export
pipeline.add_stage(
    JsonlWriter(
        path="/output/audio_manifests",
        write_kwargs={"force_ascii": False}  # Support Unicode characters
    )
)
```

**Parameters:**
- `path`: Output directory path (absolute or relative)
- `write_kwargs`: Optional dictionary passed to pandas `.to_json()` method
  - `force_ascii=False`: Preserve Unicode characters (recommended for non-English languages)
  - `orient="records"`: Format (default for JSONL)
  - `lines=True`: Write as JSONL (default)

**Note:** `AudioToDocumentStage()` is required before `JsonlWriter` because the writer operates on `DocumentBatch` objects, not `AudioBatch` objects.

### Advanced Export Options

Customize the export behavior with additional parameters:

```python
# Example: Custom JSON formatting
pipeline.add_stage(
    JsonlWriter(
        path="/output/audio_manifests",
        write_kwargs={
            "force_ascii": False,     # Preserve Unicode
            "indent": None,           # No indentation (compact)
            "ensure_ascii": False     # Allow non-ASCII characters
        }
    )
)
```

:::

::::

## Directory Structure

### Standard Output Layout

The `JsonlWriter` creates output files in the specified directory:

```text
/output/audio_manifests/
├── <hash>.jsonl   # Deterministic hash if metadata.source_files present, else UUID
├── <hash>.jsonl
└── ...
```

**File naming:**
- Files are named using deterministic hashes based on partition metadata when available
- File names are generated automatically; you cannot specify individual file names
- Multiple output files may be created depending on data partitioning

**File content:** Each JSONL file contains one or more audio records, with one JSON object per line.


## Quality Control

### Pre-Export Validation

Apply quality filters before exporting to ensure your output dataset meets minimum standards:


```python
from nemo_curator.stages.audio.common import PreserveByValueStage

# Filter by quality thresholds
quality_filters = [
    # Keep samples with WER <= 30%
    PreserveByValueStage(
        input_value_key="wer",
        target_value=30.0,
        operator="le"
    ),
    # Keep samples with duration 0.1-20.0 seconds
    PreserveByValueStage(
        input_value_key="duration", 
        target_value=0.1,
        operator="ge"
    ),
    PreserveByValueStage(
        input_value_key="duration",
        target_value=20.0, 
        operator="le"
    )
]

# Add quality filters before conversion and export
for filter_stage in quality_filters:
    pipeline.add_stage(filter_stage)

# Then convert and export
pipeline.add_stage(AudioToDocumentStage())
pipeline.add_stage(JsonlWriter(path="/output/high_quality_audio"))
```

**Recommended validation steps:**
1. **WER filtering**: Remove samples with poor transcription accuracy
2. **Duration filtering**: Exclude samples that are too short or too long
3. **Completeness check**: Ensure required fields (`audio_filepath`, `text`) are present
4. **Path validation**: Verify audio file paths are accessible for training

## Complete Export Example

Here's a complete pipeline demonstrating audio processing and export:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.resources import Resources

# Create audio curation pipeline with export
pipeline = Pipeline(name="audio_curation_with_export")

# 1. Load audio data
pipeline.add_stage(CreateInitialManifestFleursStage(
    lang="en_us",
    split="validation",
    raw_data_dir="./audio_data"
).with_(batch_size=8))

# 2. Run ASR inference
pipeline.add_stage(InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_large_pc",
    pred_text_key="pred_text"
).with_(resources=Resources(gpus=1.0)))

# 3. Calculate quality metrics
pipeline.add_stage(GetPairwiseWerStage(
    text_key="text",
    pred_text_key="pred_text",
    wer_key="wer"
))

pipeline.add_stage(GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
))

# 4. Apply quality filters
pipeline.add_stage(PreserveByValueStage(
    input_value_key="wer",
    target_value=30.0,
    operator="le"  # Keep WER <= 30%
))

pipeline.add_stage(PreserveByValueStage(
    input_value_key="duration",
    target_value=0.1,
    operator="ge"  # Keep duration >= 0.1s
))

pipeline.add_stage(PreserveByValueStage(
    input_value_key="duration",
    target_value=20.0,
    operator="le"  # Keep duration <= 20s
))

# 5. Convert to DocumentBatch and export
pipeline.add_stage(AudioToDocumentStage())
pipeline.add_stage(JsonlWriter(
    path="./curated_audio_dataset",
    write_kwargs={"force_ascii": False}
))

# Execute pipeline
executor = XennaExecutor()
pipeline.run(executor)
print("Audio curation complete. Results saved to ./curated_audio_dataset/")
```

**Expected output:**
- JSONL files in `./curated_audio_dataset/` directory
- Each file contains filtered, high-quality audio samples
- All samples have WER ≤ 30% and duration between 0.1-20.0 seconds

## Best Practices

- **Use absolute paths**: For `audio_filepath`, use absolute paths to ensure audio files are accessible during training
- **Validate before export**: Apply quality filters before conversion to reduce output size and improve dataset quality
- **Set appropriate thresholds**: Adjust WER and duration thresholds based on your specific use case and domain
- **Preserve metadata**: Include all relevant fields (WER, duration, language) for future analysis and filtering
- **Test with small batches**: Run pipeline on a small subset first to verify output format and quality
- **Document your filters**: Keep track of quality thresholds used for reproducibility

## Related Topics

- **[Text Integration](process-data/text-integration/index.md)** - Convert audio data to document format for export
- **[Quality Assessment](process-data/quality-assessment/index.md)** - Filter audio by WER, duration, and quality metrics
- **[Audio Curation Quickstart](../get-started/audio.md)** - End-to-end audio curation tutorial
- **[FLEURS Dataset](load-data/fleurs-dataset.md)** - Load and process FLEURS audio data
