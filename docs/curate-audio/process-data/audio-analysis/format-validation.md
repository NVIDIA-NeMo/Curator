---
description: "Validate audio file formats and detect corrupted files for robust audio curation pipelines"
categories: ["processors"]
tags: ["audio-validation", "format-support", "error-handling", "quality-control", "file-validation"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "audio-only"
---



(audio-analysis-format-validation)=

# Format Validation

Verify audio file formats and detect corrupted files to ensure robust audio curation pipelines with reliable data quality.

## Overview

The format validation processor verifies audio file integrity, format compatibility, and accessibility before processing. This prevents pipeline failures and ensures consistent data quality across large audio datasets.

:::note
This page shows a custom approach to format validation. Classes such as `AudioFormatValidator` and `AudioValidationFilter` are examples and are not built-in stages. Create these as custom stages using `soundfile` for detection and validation, or adapt the example stage patterns from the overview page.
:::

## Key Features

- **Format Detection**: Automatic audio format detection and validation
- **Corruption Detection**: Identifies corrupted or unreadable audio files
- **Compatibility Checking**: Ensures files are compatible with processing pipeline
- **Error Reporting**: Detailed error logs for debugging and quality control

## Supported Formats

NeMo Curator supports formats compatible with the `soundfile` library (backed by `libsndfile`):

- **WAV** - Uncompressed audio (recommended for high quality)
- **FLAC** - Lossless compression with metadata support
- **OGG** - Open-source compressed format
- **MP3** - Compressed format (requires system dependencies)
- **M4A** - Apple compressed format
- **AIFF** - Apple uncompressed format

## How it Works

The format validation processor performs comprehensive checks on audio files:

:::note
Example: The following code uses `AudioFormatValidator` as a conceptual class. Create a similar custom stage (for example, a `LegacySpeechStage` that calls `sf.info(...)`) to populate validation fields such as `is_valid` and `validation_status` in your batch entries.
:::

```python
# Conceptual example (not built-in)
from nemo_curator.stages.audio import AudioFormatValidator  # example-only

format_validator = AudioFormatValidator()
validated_batch = format_validator(audio_batch)
```

### Validation Process

1. **File Existence**: Checks if audio files exist at specified paths
2. **Format Detection**: Identifies audio format using file headers
3. **Readability Test**: Attempts to read audio file metadata
4. **Compatibility Check**: Verifies format compatibility with pipeline
5. **Corruption Detection**: Tests file integrity and structure

## Configuration Options

### Basic Configuration

```python
# Conceptual configuration (not built-in)
from nemo_curator.stages.audio import AudioFormatValidator  # example-only

format_validator = AudioFormatValidator(
    audio_field="audio_filepath",
    validation_field="is_valid",
    supported_formats=["wav", "flac", "ogg"]
)
```

### Advanced Configuration

```python
# Conceptual advanced configuration (not built-in)
format_validator = AudioFormatValidator(
    audio_field="audio_filepath",
    validation_field="validation_status",
    supported_formats=["wav", "flac", "ogg", "mp3"],
    check_corruption=True,
    check_metadata=True,
    max_file_size_mb=500,
    min_file_size_kb=1,
    report_details=True,
    remove_invalid=False
)
```

## Usage Examples

### Basic Format Validation

```python
from nemo_curator.datasets import AudioDataset

# Conceptual validator (not built-in)
from nemo_curator.stages.audio import AudioFormatValidator

audio_dataset = AudioDataset.from_manifest("audio_manifest.jsonl")
audio_batch = audio_dataset.to_batch()

format_validator = AudioFormatValidator()
validated_batch = format_validator(audio_batch)

valid_count = sum(1 for sample in validated_batch.data if sample.get("is_valid", False))
total_count = len(validated_batch.data)
print(f"Valid files: {valid_count}/{total_count}")
```

### Filtering Invalid Files

```python
# Conceptual example (not built-in)
from nemo_curator.stages.audio import AudioFormatValidator, AudioValidationFilter

format_validator = AudioFormatValidator(remove_invalid=False)
validation_filter = AudioValidationFilter(validation_field="is_valid")

validated_batch = format_validator(audio_batch)
clean_batch = validation_filter(validated_batch)
```

### Detailed Error Reporting

```python
# Conceptual reporting (not built-in)
format_validator = AudioFormatValidator(
    report_details=True,
    validation_field="validation_status"
)

validated_batch = format_validator(audio_batch)

for sample in validated_batch.data:
    status = sample.get("validation_status", {})
    if not status.get("is_valid", False):
        print(f"Invalid file: {sample['audio_filepath']}")
        print(f"Error: {status.get('error_message', 'Unknown error')}")
        print(f"Format: {status.get('detected_format', 'Unknown')}")
```

### Custom Format Support

```python
# Conceptual custom support (not built-in)
format_validator = AudioFormatValidator(
    supported_formats=["wav", "flac", "ogg", "m4a", "aiff"],
    format_converters={
        "m4a": "ffmpeg",
        "aiff": "soundfile"
    }
)
```

## Validation Results

The processor adds validation information to each audio sample:

```json
{
  "audio_filepath": "/path/to/audio.wav",
  "text": "Sample transcription",
  "is_valid": true,
  "validation_status": {
    "is_valid": true,
    "detected_format": "wav",
    "file_size_mb": 2.45,
    "sample_rate": 16000,
    "channels": 1,
    "duration": 12.3,
    "error_message": null
  }
}
```

## Integration with Pipeline

Format validation is typically the first step in audio processing pipelines:

```python
# Example pipeline combining a custom validator with built-in stages
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage

# Custom validator (example-only, not built-in). Implement similar to examples above.
class AudioFormatValidationStage:  # placeholder for your custom stage
    def __call__(self, batch):
        # populate fields like is_valid or validation_status
        return batch

preprocessing_pipeline = [
    AudioFormatValidationStage(),
    GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration"),
    PreserveByValueStage(input_value_key="duration", target_value=1.0, operator="ge"),
    InferenceAsrNemoStage(model_name="nvidia/stt_en_fastconformer_hybrid_large_pc")
]

processed_batch = audio_batch
for stage in preprocessing_pipeline:
    processed_batch = stage(processed_batch)
```

## Performance optimization

### Parallel Validation

```python
# Conceptual parallel configuration (not built-in)
format_validator = AudioFormatValidator(
    parallel_workers=8,
    chunk_size=100,
    use_threading=True
)
```

### Caching Results

```python
# Conceptual caching configuration (not built-in)
format_validator = AudioFormatValidator(
    cache_validation=True,
    cache_size=10000,
    cache_ttl_hours=24
)
```

## Troubleshooting

### Common Issues

Unsupported format errors

```python
import soundfile as sf
print("Available formats:", sf.available_formats())

# Conceptual conversion configuration (not built-in)
format_validator = AudioFormatValidator(
    auto_convert=True,
    conversion_format="wav",
    keep_original=False
)
```

File permission issues

```python
# Conceptual permission handling (not built-in)
format_validator = AudioFormatValidator(
    handle_permissions=True,
    log_permission_errors=True
)
```

Large file handling

```python
# Conceptual large-file settings (not built-in)
format_validator = AudioFormatValidator(
    max_file_size_mb=1000,
    stream_validation=True,
    quick_check=True
)
```

## Error Handling

### Validation Modes

```python
# Conceptual error handling modes (not built-in)
strict_validator = AudioFormatValidator(
    validation_mode="strict",
    stop_on_error=True
)

lenient_validator = AudioFormatValidator(
    validation_mode="lenient",
    log_errors=True,
    fallback_validation=True
)
```

### Custom Error Handling

```python
def custom_error_handler(filepath, error):
    """Custom error handling function (conceptual)"""
    print(f"Validation error for {filepath}: {error}")
    return {"is_valid": False, "error_type": type(error).__name__}

format_validator = AudioFormatValidator(
    error_handler=custom_error_handler,
    continue_on_error=True
)
```

