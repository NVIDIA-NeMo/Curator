---
description: "Understanding the AudioBatch data structure for efficient audio file management and validation in NeMo Curator"
categories: ["concepts-architecture"]
tags: ["data-structures", "audiobatch", "audio-validation", "batch-processing", "file-management"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "audio-only"
---

(about-concepts-audio-audio-batch)=
# AudioBatch Data Structure

Understanding the `AudioBatch` data structure, which serves as the core container for audio data throughout NeMo Curator's audio processing pipeline.

## Overview

`AudioBatch` is a specialized data structure that extends NeMo Curator's base `Task` class to handle audio-specific requirements:

- **File Path Management**: Validates audio file existence and accessibility
- **Batch Processing**: Groups multiple audio samples for efficient processing
- **Metadata Handling**: Preserves audio characteristics and processing results
- **Warning Logging**: Logs validation warnings during file existence checks

## Structure and Components

### Basic Structure

```python
from nemo_curator.tasks import AudioBatch

# Create AudioBatch with single audio file
audio_batch = AudioBatch(
    data={
        "audio_filepath": "/path/to/audio.wav",
        "text": "ground truth transcription",
        "duration": 3.2,
        "language": "en"
    },
    filepath_key="audio_filepath",
    task_id="audio_task_001",
    dataset_name="my_speech_dataset"
)

# Create AudioBatch with multiple audio files
audio_batch = AudioBatch(
    data=[
        {
            "audio_filepath": "/path/to/audio1.wav", 
            "text": "first transcription",
            "duration": 2.1
        },
        {
            "audio_filepath": "/path/to/audio2.wav",
            "text": "second transcription", 
            "duration": 3.5
        }
    ],
    filepath_key="audio_filepath"
)
```

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `list[dict]` | List of audio sample dictionaries |
| `filepath_key` | `str` | Key name for audio file paths in data |
| `task_id` | `str` | Unique identifier for the batch |
| `dataset_name` | `str` | Name of the source dataset |
| `num_items` | `int` | Number of audio samples in batch (read-only) |

## Data Validation

### Automatic Validation

`AudioBatch` provides built-in validation for audio data integrity:

```python
# File existence validation
audio_batch = AudioBatch(
    data=[
        {"audio_filepath": "/valid/path/audio.wav", "text": "valid sample"},
        {"audio_filepath": "/invalid/path/missing.wav", "text": "invalid sample"}
    ],
    filepath_key="audio_filepath"
)

# Validate entire batch
is_valid = audio_batch.validate()  # Returns False due to missing file

# Validate individual items
for i, item in enumerate(audio_batch.data):
    item_valid = audio_batch.validate_item(item)
    if not item_valid:
        print(f"Item {i} failed validation")
```

### Validation Behavior

The validation process checks:

1. **File Existence**: Verifies audio files exist at specified paths when `filepath_key` is provided

Note: Validation is invoked during task construction and logs warnings for missing files. It does not enforce required metadata fields (such as `text`) and does not abort processing.

**Warning Handling**: Invalid files generate warnings but don't stop processing:

```python
# Example warning output:
# WARNING: File /missing/audio.wav does not exist
```

## Batch Processing Patterns

### Optimal Batch Sizes

Choose batch sizes based on your hardware and processing requirements:

```python
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage

# GPU processing - larger batches for efficiency
wer_stage_gpu = GetPairwiseWerStage().with_(batch_size=32)

# CPU processing - smaller batches to avoid memory issues  
wer_stage_cpu = GetPairwiseWerStage().with_(batch_size=8)

# Memory-constrained environments
wer_stage_small = GetPairwiseWerStage().with_(batch_size=4)
```

### Dynamic Batching

```python
def create_dynamic_batches(audio_files: list, target_duration: float = 60.0) -> list[AudioBatch]:
    """Create batches based on total audio duration."""
    
    batches = []
    current_batch = []
    current_duration = 0.0
    
    for audio_file in audio_files:
        duration = audio_file.get("duration", 0)
        
        if current_duration + duration > target_duration and current_batch:
            # Create batch when target duration reached
            batches.append(AudioBatch(data=current_batch))
            current_batch = []
            current_duration = 0.0
        
        current_batch.append(audio_file)
        current_duration += duration
    
    # Handle remaining files
    if current_batch:
        batches.append(AudioBatch(data=current_batch))
    
    return batches
```

## Metadata Management

### Standard Metadata Fields

Common fields stored in AudioBatch data:

```python
audio_sample = {
    # Required fields
    "audio_filepath": "/path/to/audio.wav",
    "text": "transcription text",
    
    # Audio characteristics
    "duration": 3.2,
    "sample_rate": 16000,
    "channels": 1,
    
    # Processing results
    "pred_text": "asr prediction",
    "wer": 12.5,
    
    # Dataset metadata
    "language": "en_us",
    "speaker_id": "speaker_001",
    "recording_quality": "studio",
    
    # Custom fields
    "domain": "conversational",
    "noise_level": "low"
}
```

:::{note} Character error rate (CER) is available as a utility function and typically requires a custom stage to compute and store it.
:::

### Metadata Evolution

AudioBatch data evolves through the processing pipeline:

```python
# Initial state (after loading)
initial_data = {
    "audio_filepath": "/audio.wav",
    "text": "ground truth"
}

# After ASR inference
post_asr_data = {
    "audio_filepath": "/audio.wav", 
    "text": "ground truth",
    "pred_text": "asr prediction"  # Added by ASR stage
}

# After quality assessment
post_quality_data = {
    "audio_filepath": "/audio.wav",
    "text": "ground truth", 
    "pred_text": "asr prediction",
    "wer": 15.2,                    # Added by WER stage
    "duration": 3.4                 # Added by duration stage
}
```

## Error Handling

### Graceful Failure Modes

AudioBatch handles various error conditions:

```python
# Missing files
audio_batch = AudioBatch(data=[
    {"audio_filepath": "/missing/file.wav", "text": "sample"}
])
# Validation fails, but processing continues with warnings

# Corrupted audio files  
corrupted_sample = {
    "audio_filepath": "/corrupted/audio.wav",
    "text": "sample text"
}
# Duration calculation returns -1.0 for corrupted files

# Invalid metadata
invalid_sample = {
    "audio_filepath": "/valid/audio.wav",
    # Missing required "text" field
}
# AudioBatch does not enforce required metadata fields. Add a validation stage if required.
```

### Error Recovery Strategies

```python
def robust_audiobatch_creation(raw_data: list) -> AudioBatch:
    """Create AudioBatch with error recovery."""
    
    valid_data = []
    error_count = 0
    
    for item in raw_data:
        try:
            # Validate required fields
            if "audio_filepath" not in item or "text" not in item:
                error_count += 1
                continue
            
            # Validate file existence
            if not os.path.exists(item["audio_filepath"]):
                error_count += 1
                continue
                
            valid_data.append(item)
            
        except Exception as e:
            logger.warning(f"Error processing item: {e}")
            error_count += 1
    
    logger.info(f"Created AudioBatch with {len(valid_data)} valid items, {error_count} errors")
    
    return AudioBatch(
        data=valid_data,
        filepath_key="audio_filepath"
    )
```

## Performance Characteristics

### Memory Usage

AudioBatch memory footprint depends on:

- **Number of samples**: Linear scaling with batch size
- **Metadata complexity**: Additional fields increase memory usage
- **File path lengths**: Longer paths consume more memory
- **Audio file loading**: Files are loaded on-demand, not cached in batch

### Processing Efficiency

**Batch Size Impact**:
```python
# Small batches (1-4 samples)
# - Lower memory usage
# - Higher overhead per sample
# - Better for memory-constrained environments

# Medium batches (8-16 samples)  
# - Balanced memory and performance
# - Good for most use cases
# - Optimal for CPU processing

# Large batches (32+ samples)
# - Higher memory usage
# - Better GPU utilization
# - Optimal for GPU processing with sufficient VRAM
```

## Integration with Processing Stages

### Stage Input/Output

AudioBatch serves as input and output for audio processing stages:

```python
# Stage processing signature
def process(self, task: AudioBatch) -> AudioBatch:
    # Process audio data
    processed_data = []
    
    for item in task.data:
        # Apply processing logic
        processed_item = self.process_audio_item(item)
        processed_data.append(processed_item)
    
    # Return new AudioBatch with processed data
    return AudioBatch(
        data=processed_data,
        filepath_key=task.filepath_key,
        task_id=f"processed_{task.task_id}",
        dataset_name=task.dataset_name
    )
```

### Chaining Stages

```text
# AudioBatch flows through multiple stages
pipeline_flow = [
    "AudioBatch (raw)" →
    "ASR Stage" →
    "AudioBatch (with predictions)" →
    "Quality Stage" →  
    "AudioBatch (with metrics)" →
    "Filter Stage" →
    "AudioBatch (filtered)" →
    "Export Stage"
]
```

## Best Practices

### Batch Creation

1. **Validate Early**: Check file existence during batch creation
2. **Consistent Metadata**: Ensure all samples have required fields
3. **Reasonable Sizes**: Use batch sizes appropriate for your hardware
4. **Error Logging**: Track and log validation failures

### Data Organization

1. **Absolute Paths**: Use absolute file paths for reliability
2. **Consistent Schema**: Maintain consistent field names across batches
3. **Metadata Preservation**: Keep important metadata throughout processing
4. **Version Tracking**: Track data transformations for reproducibility

### Performance Optimization

1. **Batch Size Tuning**: Experiment with different batch sizes
2. **Memory Monitoring**: Watch memory usage during processing
3. **Parallel Processing**: Use multiple workers for large datasets
4. **Checkpointing**: Save intermediate results for long-running pipelines

## Related Topics

- **[ASR Pipeline](asr-pipeline.md)** - Overall pipeline architecture using AudioBatch
- **[Quality Metrics](quality-metrics.md)** - Metrics stored in AudioBatch
- **[Text Integration](text-integration.md)** - Converting AudioBatch to text formats
- **[Audio Processing](../../curate-audio/process-data/index.md)** - Practical AudioBatch usage

