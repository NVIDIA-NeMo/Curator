---
description: "Comprehensive overview of the automatic speech recognition pipeline architecture and workflow in NeMo Curator"
categories: ["concepts-architecture"]
tags: ["asr-pipeline", "speech-recognition", "architecture", "workflow", "nemo-toolkit"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "concept"
modality: "audio-only"
---

(about-concepts-audio-asr-pipeline)=

# ASR Pipeline Architecture

This guide provides a comprehensive overview of NeMo Curator's automatic speech recognition (ASR) pipeline architecture, covering audio input processing through transcription generation and quality assessment.

## Pipeline Overview

The ASR pipeline in NeMo Curator follows a systematic approach to speech processing:

```{mermaid}
graph TD
    A[Audio Files] --> B[AudioBatch Creation]
    B --> C[ASR Model Loading]
    C --> D[Batch Inference]
    D --> E[Transcription Output]
    E --> F[Quality Assessment]
    F --> G[Filtering & Export]
    
    subgraph "Input Stage"
        A
        B
    end
    
    subgraph "Processing Stage"
        C
        D
        E
    end
    
    subgraph "Assessment Stage"
        F
        G
    end
```

## Core Components

### 1. Audio Input Management

**AudioBatch Structure**: The foundation for audio processing

- Contains audio file paths and associated metadata
- Validates file existence and accessibility automatically
- Supports efficient batch processing for scalability

**Input Validation**: Ensures data integrity before processing

- File path existence checks using `AudioBatch.validate()` and `validate_item()`
- Optional metadata validation added by downstream stages (such as duration and format checks)

### 2. ASR Model Integration

**NeMo Framework Integration**: Leverages state-of-the-art ASR models

- Automatic model downloading and caching for convenience
- GPU-accelerated inference when hardware is available
- Support for multilingual and domain-specific model variants

**Model Management**: Efficient resource utilization

- Lazy loading of models to conserve system memory
- Automatic GPU or CPU device selection based on available resources
- Optimized model-level batching handled within NeMo framework

### 3. Inference Processing

**Batch Processing**: Optimized for high throughput

- Vectorized transcription processing over multiple audio files
- Configurable batch size and resource allocation using `.with_(batch_size=..., resources=Resources(...))`

**Output Generation**: Structured transcription results

- Clean predicted text extraction from NeMo model outputs
- Complete metadata preservation throughout the processing pipeline

## Processing Stages

### Stage 1: Audio Loading and Validation

```python
# AudioBatch creation with validation
audio_batch = AudioBatch(
    data=[
        {
            "audio_filepath": "/path/to/audio.wav",
            "text": "ground truth transcription"
        }
    ],
    filepath_key="audio_filepath"
)

# Automatic validation
is_valid = audio_batch.validate()  # Checks file existence
```

### Stage 2: ASR Model Setup

```python
# Model initialization
asr_stage = InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
)

# GPU/CPU device selection (based on configured resources)
device = asr_stage.check_cuda()

# Model loading
asr_stage.setup()  # Downloads and loads model
```

### Stage 3: Transcription Generation

```python
# Batch transcription
audio_files = ["/path/to/audio1.wav", "/path/to/audio2.wav"]
transcriptions = asr_stage.transcribe(audio_files)

# Output format: ["predicted text 1", "predicted text 2"]
```

### Stage 4: Quality Assessment Integration

```python
# WER calculation
wer_stage = GetPairwiseWerStage(
    text_key="text",
    pred_text_key="pred_text", 
    wer_key="wer"
)

# Duration analysis
duration_stage = GetAudioDurationStage(
    audio_filepath_key="audio_filepath",
    duration_key="duration"
)
```

## Data Flow Architecture

### Input Data Flow

1. **Audio Files** → File system or cloud storage
2. **Manifest Files** → JSONL or TSV format with metadata
3. **AudioBatch Objects** → Validated, structured data containers

### Processing Data Flow

1. **Model Loading** → NeMo ASR model initialization
2. **Batch Creation** → Group audio files for efficient processing  
3. **GPU Processing** → Transcription generation
4. **Result Aggregation** → Combine transcriptions with metadata

### Output Data Flow

1. **Transcription Results** → Predicted text for each audio file
2. **Quality Metrics** → WER, CER, duration, and custom scores
3. **Filtered Datasets** → High-quality audio-text pairs
4. **Export Formats** → JSONL manifests for training workflows

## Performance Characteristics

### Scalability Factors

**Model Selection Impact**:

- Larger models provide better accuracy but require more processing time
- NeMo models support streaming capabilities, though this stage performs offline transcription
- Language-specific models significantly improve accuracy for target languages

**Hardware Utilization**:

- GPU acceleration typically outperforms CPU processing for larger workloads
- Memory requirements scale proportionally with model size and audio input lengths

### Optimization Strategies

**Memory Management**:

```python
# Optimize for memory-constrained environments
asr_stage = InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_small"  # Smaller model
).with_(
    resources=Resources(gpus=0.5)  # Request fractional GPU via executor/backends
)
```

**Throughput Optimization**:

```python
# Optimize for maximum throughput
asr_stage = InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
).with_(
    resources=Resources(gpus=1.0)  # Dedicated GPU
)
```

## Error Handling and Recovery

### Audio Processing Errors

```python
# Validate and filter invalid file paths
audio_batch = AudioBatch(data=audio_data, filepath_key="audio_filepath")

# Filter out entries that do not exist on disk
valid_samples = [item for item in audio_batch.data if audio_batch.validate_item(item)]
```

### Pipeline Recovery

For guidance on resumable processing and recovery at the executor and backend level, refer to [Resumable Processing](../../../reference/infrastructure/resumable-processing.md).

## Integration Points

### Text Processing Integration

The ASR pipeline seamlessly integrates with text processing workflows:

```python
# Audio → Text pipeline
audio_to_text = [
    InferenceAsrNemoStage(),  # Audio → Transcriptions
    AudioToDocumentStage(),   # AudioBatch → DocumentBatch
    # Continue with text processing stages...
]
```
