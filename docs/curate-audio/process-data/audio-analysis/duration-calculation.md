---
description: "Calculate precise audio duration using soundfile library for quality assessment and metadata generation"
categories: ["processors"]
tags: ["audio-analysis", "duration", "soundfile", "metadata", "quality-control"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "audio-only"
---

(audio-analysis-duration-calculation)=

# Duration Calculation

Calculate precise audio duration using the `soundfile` library for quality assessment and metadata generation in audio curation pipelines.

## Overview

The `GetAudioDurationStage` extracts precise timing information from audio files using the `soundfile` library. This information is essential for quality filtering, dataset analysis, and ensuring consistent audio lengths for training.

## Key Features

- **High Precision**: Uses `soundfile` for frame-accurate duration calculation
- **Format Support**: Works with all audio formats supported by `soundfile` (WAV, FLAC, OGG, and so on)
- **Error Handling**: Returns -1.0 for corrupted or unreadable files
- **Pipeline Integration**: Designed for use in NeMo Curator processing pipelines

## How It Works

The duration calculation stage reads audio samples and sample rate to determine exact duration:

```python
from nemo_curator.stages.audio.common import GetAudioDurationStage
from nemo_curator.tasks import AudioBatch

# Initialize duration calculator
duration_stage = GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
)

# Process audio data
audio_data = {"audio_filepath": "/path/to/audio.wav", "text": "transcription"}
audio_batch = AudioBatch(data=[audio_data])
result_batch = duration_stage.process(audio_batch)

# Access duration information
duration = result_batch[0].data[0]["duration"]
print(f"Audio duration: {duration:.3f} seconds")
```

### Duration Calculation Process

1. **File Reading**: Uses `soundfile` to read audio samples and sample rate
2. **Frame Counting**: Counts total audio frames from the loaded samples
3. **Duration Calculation**: Computes duration as `frames ÷ sample_rate`
4. **Error Handling**: Sets duration to -1.0 for corrupted files

## Configuration

### Basic Configuration

```python
from nemo_curator.stages.audio.common import GetAudioDurationStage

# Configure duration calculation
duration_stage = GetAudioDurationStage(
    audio_filepath_key="audio_filepath",  # Field containing audio file paths
    duration_key="duration"               # Output field for duration values
)
```

### Custom Field Names

```python
# Use custom field names for your data format
duration_stage = GetAudioDurationStage(
    audio_filepath_key="wav_file_path",   # Custom input field
    duration_key="audio_length_seconds"   # Custom output field
)
```

## Usage Examples

### Basic Duration Calculation

```python
from nemo_curator.stages.audio.common import GetAudioDurationStage
from nemo_curator.tasks import AudioBatch

# Sample audio data
audio_samples = [
    {"audio_filepath": "/path/to/sample1.wav", "text": "Hello world"},
    {"audio_filepath": "/path/to/sample2.wav", "text": "How are you"},
    {"audio_filepath": "/path/to/sample3.wav", "text": "Good morning"}
]

# Create duration calculation stage
duration_stage = GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
)

# Process each sample
for sample in audio_samples:
    audio_batch = AudioBatch(data=[sample])
    result_batch = duration_stage.process(audio_batch)
    
    processed_sample = result_batch[0].data[0]
    print(f"File: {processed_sample['audio_filepath']}")
    print(f"Duration: {processed_sample['duration']:.3f} seconds")
```

### Pipeline Integration

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage

# Create audio processing pipeline
pipeline = Pipeline(name="audio_duration_pipeline")

# Add duration calculation stage
pipeline.add_stage(GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
))

# Add duration-based filtering (1-30 seconds)
pipeline.add_stage(PreserveByValueStage(
    input_value_key="duration",
    target_value=1.0,
    operator="ge"  # greater than or equal
))

pipeline.add_stage(PreserveByValueStage(
    input_value_key="duration",
    target_value=30.0,
    operator="le"  # less than or equal
))
```

### Batch Processing

```python
from nemo_curator.stages.audio.common import GetAudioDurationStage
from nemo_curator.tasks import AudioBatch

# Process multiple samples in a batch
audio_data_list = [
    {"audio_filepath": "/path/to/file1.wav", "text": "Sample 1"},
    {"audio_filepath": "/path/to/file2.wav", "text": "Sample 2"},
    {"audio_filepath": "/path/to/file3.wav", "text": "Sample 3"}
]

# Create batch
audio_batch = AudioBatch(data=audio_data_list)

# Process entire batch
duration_stage = GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
)

# Process returns list of AudioBatch objects
result_batches = duration_stage.process(audio_batch)

# Extract processed data
for batch in result_batches:
    for sample in batch.data:
        print(f"File: {sample['audio_filepath']}")
        print(f"Duration: {sample['duration']:.3f} seconds")
```

## Output Format

The stage adds duration information to each audio sample's metadata:

```json
{
  "audio_filepath": "/path/to/audio.wav",
  "text": "Sample transcription text",
  "duration": 12.345
}
```

For corrupted or unreadable files:

```json
{
  "audio_filepath": "/path/to/corrupted.wav",
  "text": "Sample transcription text", 
  "duration": -1.0
}
```

## Error Handling

The stage handles various error conditions:

### File Not Found

```python
# Non-existent files result in duration = -1.0
sample = {"audio_filepath": "/nonexistent/file.wav", "text": "test"}
audio_batch = AudioBatch(data=[sample])
result = duration_stage.process(audio_batch)
# result[0].data[0]["duration"] == -1.0
```

### Corrupted Audio Files

```python
# Corrupted files are logged and marked with duration = -1.0
# Check logs for specific error messages
import logging
logging.basicConfig(level=logging.WARNING)

# Process will continue with other files
result = duration_stage.process(audio_batch)
```

### Filtering Error Files

```python
from nemo_curator.stages.audio.common import PreserveByValueStage

# Filter out files with calculation errors
error_filter = PreserveByValueStage(
    input_value_key="duration",
    target_value=0.0,
    operator="gt"  # greater than (excludes -1.0 error values)
)
```

## Integration with Quality Assessment

Duration calculation is typically the first step in quality assessment workflows:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage

# Create comprehensive quality pipeline
pipeline = Pipeline(name="audio_quality_assessment")

# Step 1: Calculate durations
pipeline.add_stage(GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
))

# Step 2: Filter by duration range (optimal for ASR training)
pipeline.add_stage(PreserveByValueStage(
    input_value_key="duration",
    target_value=1.0,      # Minimum 1 second
    operator="ge"
))

pipeline.add_stage(PreserveByValueStage(
    input_value_key="duration", 
    target_value=15.0,     # Maximum 15 seconds
    operator="le"
))

# Step 3: Remove error files
pipeline.add_stage(PreserveByValueStage(
    input_value_key="duration",
    target_value=0.0,      # Exclude -1.0 error values
    operator="gt"
))
```

## Performance Considerations

### Memory Usage

- The stage reads audio samples to compute frames
- Memory usage scales with file duration, channels, and data type
- Reduce batch size when processing large files or large batches of files
- For a custom alternative that avoids loading samples, use `soundfile.info` to get `frames` and `samplerate`

### Processing Speed

- Duration calculation is I/O bound and scales with file size
- Network-mounted files may be slower than local storage
- Consider parallel processing for large datasets using Ray

### File System Optimization

```python
# For better performance with large datasets:
# 1. Use local storage when possible
# 2. Ensure sufficient I/O bandwidth
# 3. Consider file system caching
```

## Troubleshooting

### Common Issues

#### Unsupported Audio Formats

```python
# Check supported formats
import soundfile as sf
print("Supported formats:", sf.available_formats())

# Common supported formats: WAV, FLAC, OGG, AIFF
# MP3 support depends on your system's libsndfile build
```

