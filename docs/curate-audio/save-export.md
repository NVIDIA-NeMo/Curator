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

The primary output format for audio curation is JSONL (JSON Lines), compatible with NeMo Framework training workflows:

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

:::{tab-item} Custom Export Options

```python
# Export with custom formatting and selected fields
writer = JsonlWriter(
    path="/output/processed_audio",
    write_kwargs={
        "force_ascii": False,
        "double_precision": 2,
        "indent": None  # Compact format
    },
    # Include only these columns in the output
    fields=["audio_filepath", "text", "pred_text", "wer", "duration"]
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

### Organized by Language

```python
# Language-specific output directories
for lang in ["en_us", "es_419", "hy_am"]:
    pipeline.add_stage(
        JsonlWriter(
            path=f"/output/manifests/{lang}",
            write_kwargs={"force_ascii": False}
        )
    )
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

### Export Statistics

Refer to related workflows for post-export analysis and quality reporting:

- Audio analysis and dataset characterization: refer to Audio Analysis workflow.
- Quality assessment and WER/CER metrics: refer to Quality Assessment workflow.

## Integration with Training Workflows

### NeMo Framework Integration

Exported manifests are directly compatible with NeMo Framework training:

```python
# Use in NeMo ASR training config
train_manifest: "/output/audio_manifests/train_manifest.jsonl"
validation_manifest: "/output/audio_manifests/dev_manifest.jsonl" 
test_manifest: "/output/audio_manifests/test_manifest.jsonl"
```

### Cross-Modal Training

```python
# Export for multimodal training (audio + text)
multimodal_export = {
    "audio_path": "audio_filepath",
    "transcription": "text", 
    "predicted_transcription": "pred_text",
    "quality_score": "wer",
    "metadata": {
        "duration": "duration",
        "language": "language"
    }
}
```

## Performance Considerations

::::{tab-set}

:::{tab-item} Large Dataset Export

For datasets with millions of audio files:

```python
# Use compression and control batch sizes upstream
writer = JsonlWriter(
    path="/output/large_dataset",
    write_kwargs={"compression": "gzip"}
)

# Tip: reduce upstream batch sizes when converting/writing
pipeline.add_stage(AudioToDocumentStage().with_(batch_size=1))
```

:::

:::{tab-item}  Storage Optimization

```python
# Optimize for storage efficiency using writer options
writer = JsonlWriter(
    path="/output/processed_audio",
    write_kwargs={
        "compression": "gzip",
        "double_precision": 2,
        "indent": None,
    },
    # Select only needed columns
    fields=["audio_filepath", "text", "pred_text", "wer", "duration"]
)
```

:::

::::

## Troubleshooting

### Common Export Issues

::::{tab-set}
:::{tab-item} Large file sizes

Use compression and field filtering

```python
JsonlWriter(write_kwargs={"compression": "gzip"})
```

:::

:::{tab-item} Unicode errors

Ensure proper encoding

```python
JsonlWriter(write_kwargs={"force_ascii": False})
```

:::

:::{tab-item} Memory issues

 Reduce upstream batch sizes

```python
# Example: write smaller batches by adjusting upstream stage
AudioToDocumentStage().with_(batch_size=1)
```

:::

::::
