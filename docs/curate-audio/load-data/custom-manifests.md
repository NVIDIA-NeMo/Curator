---
description: "Create and load custom audio manifests in JSONL format for NeMo Curator audio processing"
categories: ["data-loading"]
tags: ["custom-manifests", "jsonl", "audio-metadata"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "audio-only"
---

(audio-load-data-custom-manifests)=
# Create and Load Custom Audio Manifests

Create and load custom audio manifests in JSONL format for your speech datasets. This guide covers the required manifest format and how to load manifests into NeMo Curator pipelines.

## Manifest Format

NeMo Curator uses JSONL (JSON Lines) format for audio manifests, with one JSON object per line:

```json
{"audio_filepath": "/data/audio/sample_001.wav", "text": "hello world", "duration": 2.1}
{"audio_filepath": "/data/audio/sample_002.wav", "text": "good morning", "duration": 1.8}
{"audio_filepath": "/data/audio/sample_003.wav", "text": "how are you", "duration": 2.3}
```

:::{note}
NeMo Curator does not provide a generic TSV reader stage. You must convert your data to JSONL format before loading, or use dataset-specific importers like the FLEURS manifest creator.
:::

## Required Fields

Every audio manifest entry must include:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `audio_filepath` | string | Absolute or relative path to audio file | `/data/audio/sample.wav` |
| `text` | string | Ground truth transcription | `"hello world"` |

## Optional Fields

Additional fields that can enhance processing:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `duration` | float | Audio duration in seconds | `2.1` |
| `language` | string | Language identifier | `"en_us"` |
| `speaker_id` | string | Speaker identifier | `"speaker_001"` |
| `sample_rate` | int | Audio sample rate in Hz | `16000` |

## Creating Custom Manifests

You'll need to create your own manifest files using your preferred tools. Here's a simple Python example:

```python
import json

# Example: Create a manifest from a list of audio files
audio_data = [
    {"audio_filepath": "/data/audio/sample1.wav", "text": "hello world"},
    {"audio_filepath": "/data/audio/sample2.wav", "text": "good morning"},
    {"audio_filepath": "/data/audio/sample3.wav", "text": "how are you"}
]

# Write JSONL manifest
with open("my_audio_manifest.jsonl", "w") as f:
    for entry in audio_data:
        f.write(json.dumps(entry) + "\n")
```

## Loading Manifests in Pipelines

### Using JsonlReader

Load your custom manifest using the built-in `JsonlReader`:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage

# Create pipeline
pipeline = Pipeline(name="custom_audio_processing")

# Load custom manifest (produces DocumentBatch)
pipeline.add_stage(
    JsonlReader(
        file_paths="my_audio_manifest.jsonl",
        fields=["audio_filepath", "text"]
    )
)

# ASR inference (consumes DocumentBatch, produces AudioBatch)
pipeline.add_stage(
    InferenceAsrNemoStage(
        model_name="nvidia/stt_en_fastconformer_hybrid_large_pc",
        filepath_key="audio_filepath",
        pred_text_key="pred_text"
    )
)

# Calculate WER between ground truth and prediction
pipeline.add_stage(
    GetPairwiseWerStage(
        text_key="text", 
        pred_text_key="pred_text", 
        wer_key="wer"
    )
)
```

### Validation

Audio file validation happens automatically during pipeline processing:

```python
# Validation occurs when stages process AudioBatch objects
# Files are checked for existence during ASR inference
# Invalid files generate warnings but don't stop processing

# Pipeline stages handle validation automatically
if audio_batch.validate():
    print("All audio files exist")
else:
    print("Some audio files are missing")
```

## Example: Complete Workflow

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter

def create_custom_audio_pipeline(manifest_path: str, output_path: str) -> Pipeline:
    """Create pipeline for custom audio manifest processing."""
    
    pipeline = Pipeline(name="custom_audio_processing")
    
    # Load custom manifest
    pipeline.add_stage(JsonlReader(
        file_paths=manifest_path,
        fields=["audio_filepath", "text"]
    ))
    
    # ASR processing
    pipeline.add_stage(InferenceAsrNemoStage(
        model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
    ))
    
    # Quality assessment
    pipeline.add_stage(GetPairwiseWerStage())
    pipeline.add_stage(GetAudioDurationStage(
        audio_filepath_key="audio_filepath",
        duration_key="duration"
    ))
    
    # Filter by quality (keep WER <= 40%)
    pipeline.add_stage(PreserveByValueStage(
        input_value_key="wer",
        target_value=40.0,
        operator="le"
    ))
    
    # Export results
    pipeline.add_stage(AudioToDocumentStage())
    pipeline.add_stage(JsonlWriter(path=output_path))
    
    return pipeline

# Usage
pipeline = create_custom_audio_pipeline(
    manifest_path="my_audio_manifest.jsonl",
    output_path="processed_results"
)
```

## Related Topics

- **[Audio Processing Overview](../../index.md)** - Complete audio processing workflow
- **[FLEURS Dataset](fleurs-dataset.md)** - Example of automated dataset loading
- **[Local Files](local-files.md)** - Loading audio files from local directories