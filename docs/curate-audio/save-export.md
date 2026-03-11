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

Export processed audio data and transcriptions in formats optimized for ASR model training, audio-and-text applications, and downstream analysis workflows.

## Output Formats

NeMo Curator's audio curation pipeline supports several output formats tailored for different use cases:

### JSONL Manifests

The primary output format for audio curation is JSONL (JSON Lines):

```json
{"audio_filepath": "/data/audio/sample_001.wav", "text": "hello world", "pred_text": "hello world", "wer": 0.0, "duration": 2.1}
{"audio_filepath": "/data/audio/sample_002.wav", "text": "good morning", "pred_text": "good morning", "wer": 0.0, "duration": 1.8}
```

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

## Export Configuration

::::{tab-set}

:::{tab-item} Using JsonlWriter

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

:::

::::

## Directory Structure

### Standard Output Layout

When `source_files` metadata exists, the writer generates deterministic hashed file names. Otherwise, it generates UUID-based names.

```text
/output/audio_manifests/
├── <hash>.jsonl   # Deterministic hash if metadata.source_files present, else UUID
├── <hash>.jsonl
└── ...
```


## Quality Control

### Validation Checks

Before export, check your processed data:

```python
from nemo_curator.stages.audio.common import PreserveByValueStage

# Filter by quality thresholds
quality_filters = [
    # Keep samples with WER <= 50%
    PreserveByValueStage(
        input_value_key="wer",
        target_value=50.0,
        operator="le"
    ),
    # Keep samples with duration 1-30 seconds
    PreserveByValueStage(
        input_value_key="duration", 
        target_value=1.0,
        operator="ge"
    ),
    PreserveByValueStage(
        input_value_key="duration",
        target_value=30.0, 
        operator="le"
    )
]

for filter_stage in quality_filters:
    pipeline.add_stage(filter_stage)
```
