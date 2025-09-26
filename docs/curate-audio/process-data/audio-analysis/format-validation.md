---
description: "Audio format support and error handling in NeMo Curator audio processing stages"
categories: ["processors"]
tags: ["audio-validation", "format-support", "error-handling", "soundfile"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "reference"
modality: "audio-only"
---

(audio-analysis-format-validation)=

# Audio Format Support

NeMo Curator audio processing stages use the `soundfile` library for audio file handling. Built-in error handling surfaces unreadable or unsupported files during duration calculation.

## Supported Formats

Audio stages support formats compatible with the `soundfile` library (backed by `libsndfile`):

- **WAV**: Uncompressed audio (recommended for high quality)
- **FLAC**: Lossless compression with metadata support
- **OGG**: Open-source compressed format
- **MP3**: Compressed format (availability depends on your system's `libsndfile` build)
- **AIFF**: Apple uncompressed format

Note: AAC/M4A is not supported by default by `soundfile`/`libsndfile`. Prefer WAV or FLAC for consistent cross-platform behavior.

## Built-in Error Handling

### Duration Calculation with Error Handling

The `GetAudioDurationStage` automatically handles corrupted or unreadable files:

```python
from nemo_curator.stages.audio.common import GetAudioDurationStage

# Calculate duration with built-in error handling
duration_stage = GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
)
```

### Error Handling Behavior

When `soundfile`/`libsndfile` cannot read audio files:

- **Duration Calculation**: Returns -1.0 for corrupted/unreadable files
- **ASR Inference**: Will fail with clear error messages for unsupported formats
- **File Validation**: Use duration = -1.0 as an indicator of file issues

```python
from nemo_curator.stages.audio.common import PreserveByValueStage

# Filter out corrupted files (duration = -1.0)
valid_files_filter = PreserveByValueStage(
    input_value_key="duration",
    target_value=0.0,
    operator="gt"  # greater than 0
)
```

## Working Example

Here is a complete pipeline that handles format validation through built-in error handling:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage

# Create pipeline with built-in error handling
pipeline = Pipeline(name="audio_validation")

# 1. Calculate duration (automatically handles format validation)
pipeline.add_stage(GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
))

# 2. Filter out corrupted files (duration = -1.0 indicates issues)
pipeline.add_stage(PreserveByValueStage(
    input_value_key="duration",
    target_value=0.0,
    operator="gt"
))

# 3. Proceed with ASR inference on valid files only
pipeline.add_stage(InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
))
```

## Format Support Check

To check supported formats on your system:

```python
import soundfile as sf

# Check available formats
print("Supported formats:")
for format_name, format_info in sf.available_formats().items():
    print(f"  {format_name}: {format_info}")

# Check specific file
try:
    info = sf.info("your_audio_file.wav")
    print(f"File info: {info}")
except Exception as e:
    print(f"File validation failed: {e}")
```

This approach leverages the built-in error handling of NeMo Curator's audio stages rather than requiring extra format validation steps.
