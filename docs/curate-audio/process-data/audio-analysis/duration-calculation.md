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

Calculate precise audio duration using the soundfile library for quality assessment and metadata generation in audio curation pipelines.

## Overview

The duration calculation processor extracts precise timing information from audio files using the soundfile library. This information is essential for quality filtering, dataset analysis, and ensuring consistent audio lengths for training.

## Key Features

- **High Precision**: Uses soundfile for frame-accurate duration calculation
- **Format Support**: Works with all audio formats supported by soundfile (WAV, FLAC, OGG, etc.)
- **Metadata Integration**: Adds duration information to AudioBatch metadata
- **Batch Processing**: Efficiently processes large datasets with GPU acceleration

## How It Works

The duration calculation processor reads audio file headers and samples to determine exact duration:

```python
import soundfile as sf
from nemo_curator.stages.audio import AudioDurationCalculator

# Initialize duration calculator
duration_calc = AudioDurationCalculator()

# Process AudioBatch to add duration metadata
audio_batch_with_duration = duration_calc(audio_batch)
```

### Duration Calculation Process

1. **File Reading**: Uses soundfile to read audio file metadata
2. **Frame Counting**: Counts total audio frames and sample rate
3. **Duration Calculation**: Computes duration as frames ÷ sample_rate
4. **Metadata Addition**: Adds duration field to AudioBatch metadata

## Configuration Options

### Basic Configuration

```python
from nemo_curator.stages.audio import AudioDurationCalculator

# Default configuration
duration_calc = AudioDurationCalculator(
    audio_field="audio_filepath",  # Field containing audio file paths
    duration_field="duration",     # Output field for duration values
    precision=3                    # Decimal precision for duration
)
```

### Advanced Configuration

```python
# Custom configuration with error handling
duration_calc = AudioDurationCalculator(
    audio_field="audio_filepath",
    duration_field="duration_seconds",
    precision=6,                   # Higher precision for research
    handle_errors="skip",          # Skip files with errors
    validate_files=True,           # Validate file existence
    cache_results=True             # Cache duration calculations
)
```

## Usage Examples

### Basic Duration Calculation

```python
from nemo_curator.datasets import AudioDataset
from nemo_curator.stages.audio import AudioDurationCalculator

# Load audio dataset
audio_dataset = AudioDataset.from_manifest("audio_manifest.jsonl")
audio_batch = audio_dataset.to_batch()

# Calculate durations
duration_calc = AudioDurationCalculator()
audio_batch_with_duration = duration_calc(audio_batch)

# Access duration information
for sample in audio_batch_with_duration.data:
    print(f"File: {sample['audio_filepath']}")
    print(f"Duration: {sample['duration']:.3f} seconds")
```

### Duration-Based Filtering

```python
from nemo_curator.stages.audio import AudioDurationCalculator, AudioDurationFilter

# Calculate durations first
duration_calc = AudioDurationCalculator()
audio_batch_with_duration = duration_calc(audio_batch)

# Filter by duration range (1-30 seconds)
duration_filter = AudioDurationFilter(
    min_duration=1.0,
    max_duration=30.0,
    duration_field="duration"
)

filtered_batch = duration_filter(audio_batch_with_duration)
print(f"Filtered {len(filtered_batch.data)} samples by duration")
```

### Batch Processing with Progress

```python
from nemo_curator.stages.audio import AudioDurationCalculator
from tqdm import tqdm

# Process large dataset with progress tracking
duration_calc = AudioDurationCalculator(show_progress=True)

# Process in chunks for memory efficiency
chunk_size = 1000
total_samples = len(audio_batch.data)

processed_chunks = []
for i in tqdm(range(0, total_samples, chunk_size)):
    chunk = audio_batch.slice(i, i + chunk_size)
    processed_chunk = duration_calc(chunk)
    processed_chunks.append(processed_chunk)

# Combine processed chunks
final_batch = AudioBatch.concat(processed_chunks)
```

## Output Format

The processor adds duration information to each audio sample's metadata:

```json
{
  "audio_filepath": "/path/to/audio.wav",
  "text": "Sample transcription text",
  "duration": 12.345,
  "sample_rate": 16000,
  "channels": 1
}
```

## Performance Considerations

### Memory Optimization

```python
# For large datasets, use streaming processing
duration_calc = AudioDurationCalculator(
    stream_processing=True,    # Process files one at a time
    cache_size=1000,          # Cache recently calculated durations
    parallel_workers=4        # Use multiple workers for I/O
)
```

### Error Handling

```python
# Robust error handling for production pipelines
duration_calc = AudioDurationCalculator(
    handle_errors="log",       # Log errors but continue processing
    fallback_duration=0.0,     # Default duration for failed files
    validate_range=(0.1, 3600) # Valid duration range (0.1s to 1 hour)
)
```

## Integration with Quality Assessment

Duration calculation is often the first step in quality assessment workflows:

```python
from nemo_curator.stages.audio import (
    AudioDurationCalculator,
    AudioDurationFilter,
    AudioQualityAssessment
)

# Complete quality assessment pipeline
pipeline = [
    AudioDurationCalculator(),                    # Calculate durations
    AudioDurationFilter(min_duration=1.0),       # Filter by minimum duration
    AudioQualityAssessment(),                     # Calculate other quality metrics
]

# Process audio batch through pipeline
processed_batch = audio_batch
for stage in pipeline:
    processed_batch = stage(processed_batch)
```

## Troubleshooting

### Common Issues

**File Not Found Errors**
```python
# Validate file paths before processing
duration_calc = AudioDurationCalculator(validate_files=True)
```

**Unsupported Audio Formats**
```python
# Check supported formats
import soundfile as sf
print("Supported formats:", sf.available_formats())

# Use format validation first
from nemo_curator.stages.audio import AudioFormatValidator
format_validator = AudioFormatValidator()
validated_batch = format_validator(audio_batch)
duration_calc = AudioDurationCalculator()
processed_batch = duration_calc(validated_batch)
```

**Memory Issues with Large Files**
```python
# Use streaming for large files
duration_calc = AudioDurationCalculator(
    stream_processing=True,
    max_file_size_mb=100  # Skip files larger than 100MB
)
```

## Related Topics

- **[Format Validation](format-validation.md)** - Validate audio files before duration calculation
- **[Quality Assessment](../quality-assessment/index.md)** - Use duration in quality filtering workflows
- **[Audio Analysis Overview](index.md)** - Complete audio analysis capabilities
- **[ASR Inference](../asr-inference/index.md)** - Transcription processing workflows
