---
description: "Extract and analyze audio file characteristics including duration calculation, format validation, and metadata extraction"
categories: ["audio-processing"]
tags: ["audio-analysis", "duration-calculation", "format-validation", "metadata-extraction", "file-validation"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "workflow"
modality: "audio-only"
---

(audio-process-data-audio-analysis)=
# Audio Analysis

Extract and analyze audio file characteristics for quality control, metadata generation, and dataset validation. Audio analysis provides essential information about audio files before and during processing.

## How it Works

Audio analysis in NeMo Curator examines audio files to extract:

1. **Duration Information**: Precise timing measurements using `soundfile`
2. **Format Characteristics**: Sample rate, bit depth, channels, and format
3. **Quality Indicators**: File integrity, format compliance, technical quality
4. **Metadata Extraction**: Embedded metadata and file properties

:::{note} NeMo Curator provides duration extraction as a built-in stage (`GetAudioDurationStage`). The format and metadata examples below show how to build custom stages and are not built-in.
:::

## Duration Analysis

### Precise Duration Calculation

```python
from nemo_curator.stages.audio.common import GetAudioDurationStage

# Calculate audio duration for each file
duration_stage = GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
)
```

The duration calculation:

- Uses the `soundfile` library; computes duration as frames ÷ sample rate
- Handles formats supported by `soundfile` (`libsndfile`)
- Returns -1.0 for corrupted or unreadable files
- Calculates: `duration = sample_count / sample_rate`

### Duration-Based Quality Assessment

After calculating durations, you can analyze the results:

### Duration Filtering Example

```python
from nemo_curator.stages.audio.common import PreserveByValueStage

# Keep samples between 1 and 15 seconds
min_duration_filter = PreserveByValueStage(
    input_value_key="duration",
    target_value=1.0,
    operator="ge"
)
max_duration_filter = PreserveByValueStage(
    input_value_key="duration",
    target_value=15.0,
    operator="le"
)
```

Refer to [Duration Filtering](../quality-assessment/duration-filtering.md) for end-to-end examples.

## Format Validation

NeMo Curator's built-in stages handle audio format validation automatically. See [Format Validation](format-validation.md) for details on error handling and supported formats.

### Basic Format Check

```python
import soundfile as sf

# Check if file is readable
try:
    info = sf.info("audio_file.wav")
    print(f"Duration: {info.duration}s, Sample rate: {info.samplerate}Hz")
except Exception as e:
    print(f"File validation failed: {e}")
```

## Complete Analysis Pipeline

Here's a complete working pipeline for audio analysis:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage

# Create analysis pipeline
pipeline = Pipeline(name="audio_analysis")

# 1. Calculate duration (handles format validation automatically)
pipeline.add_stage(GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
))

# 2. Filter by duration (removes corrupted files with duration = -1.0)
pipeline.add_stage(PreserveByValueStage(
    input_value_key="duration",
    target_value=1.0,
    operator="ge"  # >= 1 second
))

pipeline.add_stage(PreserveByValueStage(
    input_value_key="duration", 
    target_value=15.0,
    operator="le"  # <= 15 seconds
))

# 3. Continue with ASR inference on validated files
pipeline.add_stage(InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
))
```

**Source**: `tutorials/audio/fleurs/pipeline.py:47-48`
**Evidence**: Working pipeline demonstrates audio analysis with duration calculation and filtering
