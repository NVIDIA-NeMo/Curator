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

This guide provides a comprehensive overview of NeMo Curator's automatic speech recognition (ASR) pipeline architecture, from audio input through transcription and quality assessment.

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

- Contains audio file paths and metadata
- Validates file existence and accessibility
- Supports batch processing for efficiency

**Input Validation**: Ensures audio quality before processing

- Path existence checks using `AudioBatch.validate()` and `validate_item()`
- Optional metadata checks added by downstream stages (for example, duration, format)

### 2. ASR Model Integration

**NeMo Framework Integration**: Leverages state-of-the-art ASR models

- Automatic model downloading and caching
- GPU-accelerated inference when available
- Support for multilingual and domain-specific models

**Model Management**: Efficient resource use

- Lazy loading of models to conserve memory
- GPU/CPU device selection based on configured resources
- Model-level batching behavior handled inside NeMo models

### 3. Inference Processing

**Batch Processing**: Optimized for throughput

- Vectorized transcription over a list of files passed to the model
- The NeMo ASR model handles batching; this stage does not expose a batch size control

**Output Generation**: Structured transcription results

- Predicted text extraction from NeMo outputs
- Metadata preservation throughout processing

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

- Larger models can provide better accuracy but may be slower
- Some NeMo models support streaming; this stage performs offline transcription
- Language-specific models improve accuracy for target languages

**Hardware Use**:

- GPU acceleration typically outperforms CPU for larger workloads
- Memory requirements scale with model size and input lengths

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

### Model Loading Failures

```python
try:
    asr_stage.setup()
except RuntimeError as e:
    logger.error(f"ASR model loading failed: {e}")
    # Fallback strategies:
    # 1. Try smaller model
    # 2. Use CPU processing
    # 3. Check network connectivity
```

### Audio Processing Errors

```python
# Validate and filter invalid file paths
audio_batch = AudioBatch(data=audio_data, filepath_key="audio_filepath")

# Filter out entries that do not exist on disk
valid_samples = [item for item in audio_batch.data if audio_batch.validate_item(item)]
```

### Pipeline Recovery

For guidance on resumable processing and recovery at the executor/backends level, refer to [Resumable Processing](../../../reference/infrastructure/resumable-processing.md).

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

### Multi-Modal Integration

Support for audio-visual and audio-text multi-modal workflows:

```python
# Audio + Video processing
multimodal_pipeline = Pipeline(name="audio_video")

# Process audio track
multimodal_pipeline.add_stage(audio_asr_stage)

# Process video frames (separate branch)
multimodal_pipeline.add_stage(video_processing_stage)

# Combine modalities
multimodal_pipeline.add_stage(multimodal_fusion_stage)
```

## Related Concepts

- **[Quality Metrics](quality-metrics.md)** - Understanding WER, CER, and custom metrics
- **[AudioBatch Structure](audio-batch.md)** - Core data structures for audio processing
- **[Text Integration](text-integration.md)** - Combining audio and text workflows
- **[Infrastructure Components](../../reference/infrastructure/index.md)** - Distributed processing foundations
