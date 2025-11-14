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

This guide covers the `AudioBatch` data structure, which serves as the core container for audio data throughout NeMo Curator's audio processing pipeline.

## Overview

`AudioBatch` is a specialized data structure that extends NeMo Curator's base `Task` class to handle audio-specific processing requirements:

- **File Path Management**: Automatically validates audio file existence and accessibility
- **Batch Processing**: Groups multiple audio samples for efficient parallel processing
- **Metadata Handling**: Preserves audio characteristics and processing results throughout pipeline stages

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
| `data` | `dict \| list[dict] \| None` | Audio sample data (stored internally as `list[dict]`) |
| `filepath_key` | `str \| None` | Key name for audio file paths in data (optional) |
| `task_id` | `str` | Unique identifier for the batch |
| `dataset_name` | `str` | Name of the source dataset |
| `num_items` | `int` | Number of audio samples in batch (read-only property) |

## Data Validation

### Automatic Validation

`AudioBatch` provides built-in validation for audio data integrity.

## Metadata Management

### Standard Metadata Fields

Common fields stored in AudioBatch data:

```python
audio_sample = {
    # Core fields (user-provided)
    "audio_filepath": "/path/to/audio.wav",
    "text": "transcription text",
    
    # Fields added by processing stages
    "pred_text": "asr prediction",    # Added by ASR inference stages
    "wer": 12.5,                     # Added by GetPairwiseWerStage
    "duration": 3.2,                 # Added by GetAudioDurationStage
    
    # Optional user-provided metadata
    "language": "en_us",
    "speaker_id": "speaker_001",
    
    # Custom fields (examples)
    "domain": "conversational",
    "noise_level": "low"
}
```

```{note}
Character error rate (CER) is available as a utility function and typically requires a custom stage to compute and store it.
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
    # Missing "text" field - needed for WER calculation but not enforced by AudioBatch
}
# AudioBatch does not enforce metadata field requirements. Add a validation stage if required.
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

AudioBatch memory footprint depends on these factors:

- **Number of samples**: Memory usage scales linearly with batch size
- **Metadata complexity**: Additional metadata fields increase memory consumption
- **File path lengths**: Longer file paths consume more memory
- **Audio file loading**: Audio files are loaded on-demand and not cached in the batch

### Processing Efficiency

**Batch Size Impact**:

**Small batches**:
- Lower memory usage
- Higher overhead per sample
- Better for memory-constrained environments

**Medium batches**:
- Balanced memory and performance
- Good for most use cases
- Optimal for CPU processing

**Large batches**:
- Higher memory usage
- Better GPU utilization
- Optimal for GPU processing with sufficient VRAM

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

AudioBatch flows through multiple processing stages, with each stage adding new metadata fields:

```{mermaid}
flowchart TD
    A["AudioBatch (raw)<br/>• audio_filepath<br/>• text"] --> B[ASR Inference Stage]
    B --> C["AudioBatch (with predictions)<br/>• audio_filepath<br/>• text<br/>• pred_text"]
    C --> D[Quality Assessment Stage]
    D --> E["AudioBatch (with metrics)<br/>• audio_filepath<br/>• text<br/>• pred_text<br/>• wer<br/>• duration"]
    E --> F[Filter Stage]
    F --> G["AudioBatch (filtered)<br/>• audio_filepath<br/>• text<br/>• pred_text<br/>• wer<br/>• duration"]
    G --> H[Export Stage]
    H --> I[Output Files]
    
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style E fill:#e8f5e8
    style G fill:#fff3e0
    style I fill:#fce4ec
```
