---
description: "Concepts for constructing manifests and ingesting audio datasets in NeMo Curator"
categories: ["concepts-architecture"]
tags: ["manifests", "ingest", "datasets", "audio"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "audio-only"
---

(about-concepts-audio-manifests-ingest)=

# Dataset Manifests and Ingest

This guide covers the core concepts for ingesting audio data into NeMo Curator using consistent manifests and validation workflows.

## Manifest Structure

Audio manifests in NeMo Curator follow a standardized format for consistent data processing:

**Required Fields**:

- `audio_filepath`: Path to the audio file (absolute or relative)

**Common Optional Fields**:

- `text`: Ground truth transcription or existing transcription
- `duration`: Audio length in seconds
- `language`: Language code (such as "en", "es", "fr")
- `speaker_id`: Speaker identifier for multi-speaker datasets
- Custom metadata fields for domain-specific information

**Creation Methods**:

- **Programmatic Generation**: Use dataset-specific stages like `CreateInitialManifestFleursStage`
- **Custom Scripts**: Generate JSONL files with consistent field naming
- **Manual Creation**: Create JSONL manifests for small datasets or specialized use cases

## Data Ingestion and Validation

NeMo Curator provides robust validation mechanisms for audio data ingestion:

**File Existence Validation**:

- `AudioBatch` automatically validates file paths during creation
- Use `validate()` for batch-level validation
- Use `validate_item()` for individual file validation
- Missing files generate warnings but do not stop processing

**Validation Strategy**:

- Check file existence at the start of the pipeline
- Add metadata fields (duration, format) in downstream processing stages
- Use non-blocking validation to maintain processing throughput

## Field Recommendations

**Essential for All Workflows**:

- `audio_filepath`: File path validation and processing

**Recommended for ASR Workflows**:

- `text`: Ground truth for WER calculation and quality assessment
- `language`: Language-specific model selection and validation

**Recommended for Quality Assessment**:

- `duration`: Duration-based filtering and speech rate analysis
- `speaker_id`: Speaker consistency and diversity analysis

**Domain-Specific Fields**:

- Recording quality indicators (studio, phone, outdoor)
- Content type tags (conversational, broadcast, lecture)
- Noise level indicators for quality assessment

## Implementation Examples

**Basic Manifest Creation**:

```python
import json

# Create simple manifest
manifest_data = [
    {
        "audio_filepath": "/path/to/audio1.wav",
        "text": "Hello world",
        "duration": 1.5,
        "language": "en"
    },
    {
        "audio_filepath": "/path/to/audio2.wav", 
        "text": "Good morning",
        "duration": 2.1,
        "language": "en"
    }
]

# Save as JSONL
with open("manifest.jsonl", "w") as f:
    for item in manifest_data:
        f.write(json.dumps(item) + "\n")
```

**AudioBatch Validation**:

```python
from nemo_curator.tasks import AudioBatch

# Create AudioBatch with validation
audio_batch = AudioBatch(
    data=manifest_data,
    filepath_key="audio_filepath"
)

# Validate file existence
is_valid = audio_batch.validate()
print(f"Batch validation: {is_valid}")
```

## Pipeline Integration

**ASR Workflow Preparation**:

- Ensure `audio_filepath` points to valid audio files
- ASR stages automatically add `pred_text` field with predictions
- Include `text` field for WER calculation and quality assessment

**Quality Assessment Preparation**:

- Use `GetAudioDurationStage` to add duration information
- Include existing transcriptions for WER-based filtering
- Add metadata fields for comprehensive quality analysis

**Format Conversion Readiness**:

- Standardize field names across different data sources
- Ensure consistent audio file formats and sample rates
- Validate encoding and accessibility of all audio files
