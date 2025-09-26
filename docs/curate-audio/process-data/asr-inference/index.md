---
description: "Perform automatic speech recognition using NeMo Framework models with GPU acceleration and batch processing"
categories: ["audio-processing"]
tags: ["asr-inference", "nemo-models", "speech-recognition", "gpu-accelerated", "batch-processing"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "audio-only"
---

(audio-process-data-asr-inference)=

# ASR Inference

Perform automatic speech recognition (ASR) on audio files using NeMo Framework models. The ASR inference stage transcribes audio into text, enabling downstream quality assessment and text processing workflows.

## How it Works

The `InferenceAsrNemoStage` processes `AudioBatch` objects by:

1. **Input Validation**: Verifies required attributes and data structure
2. **Model Loading**: Downloads and initializes NeMo ASR models on GPU or CPU
3. **Batch Processing**: Groups audio files for efficient inference
4. **Transcription**: Generates text predictions for each audio file
5. **Output Creation**: Returns `AudioBatch` with original data plus predicted transcriptions

## Basic Usage

### Simple ASR Inference

```python
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.resources import Resources

# Create ASR inference stage
asr_stage = InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_large_pc",
    filepath_key="audio_filepath",
    pred_text_key="pred_text"
)

# Configure for GPU processing
asr_stage = asr_stage.with_(
    resources=Resources(gpus=1.0),
    batch_size=16
)
```

### Multilingual ASR

```python
# Use language-specific models
language_models = {
    "en_us": "nvidia/stt_en_fastconformer_hybrid_large_pc",
    "es_419": "nvidia/stt_es_fastconformer_hybrid_large_pc", 
    "hy_am": "nvidia/stt_hy_fastconformer_hybrid_large_pc",
}

# Create stage for Armenian
armenian_asr = InferenceAsrNemoStage(
    model_name=language_models["hy_am"]
)
```

## Configuration Options

### Model Selection

NeMo Framework provides ready-to-use ASR models for several languages and domains:

```python
# Domain-specific models
models = {
    "general": "nvidia/stt_en_fastconformer_hybrid_large_pc",
    "telephony": "nvidia/stt_en_fastconformer_telephony_large",
    "streaming": "nvidia/stt_en_fastconformer_streaming_large",
}
```

### Resource Configuration

```python
from nemo_curator.stages.resources import Resources

# GPU configuration
asr_stage = asr_stage.with_(
    resources=Resources(
        gpus=1.0,           # Number of GPUs (multi-GPU aware stages)
        cpus=4.0            # CPU cores
    )
)

# Alternatively, request fractional single-GPU memory (do not combine with gpus):
# asr_stage = asr_stage.with_(resources=Resources(cpus=4.0, gpu_memory_gb=16.0))
```

### Batch Processing

```python
# Optimize batch size based on GPU memory
asr_stage = asr_stage.with_(
    batch_size=32  # Larger batches improve GPU utilization
)
```

:::{note} `batch_size` controls the number of tasks the executor groups per call. The ASR stage does not define `process_batch()`; the executor batches tasks.

Within a single `AudioBatch`, `process()` transcribes the file paths together.

:::

## Input Requirements

### AudioBatch Format

Data loading stages create input `AudioBatch` objects that must contain:

```python
# AudioBatch structure (created automatically by loading stages)
# Each item in the batch contains:
{
    "audio_filepath": "/path/to/audio1.wav",
    # Optional: existing metadata
    "duration": 5.2,
    "language": "en"
}
```

### Audio File Requirements

- **Supported Formats**: Determined by the selected NeMo ASR model; refer to the NeMo ASR documentation.
- **Sample Rates**: Typically 16 kHz; refer to the model card for details.
- **Channels**: Mono or stereo; channel handling (for example, down-mixing) depends on the model.
- **Duration**: Long files may require manual chunking before inference.

## Output Structure

The ASR stage adds predicted transcriptions to each audio sample:

```python
# Output AudioBatch structure
{
    "audio_filepath": "/path/to/audio1.wav",
    "pred_text": "this is the predicted transcription",
    "duration": 5.2,  # Preserved from input
    "language": "en"  # Preserved from input
}
```

## Error Handling

### Model Loading Errors

```python
try:
    asr_stage.setup()
except RuntimeError as e:
    print(f"Failed to load ASR model: {e}")
    # Fallback to CPU or different model
```

### Processing Errors

Processing behavior:

- **Input structure validation**: The stage uses `validate_input()` to check required attributes/columns and raises `ValueError` if they are missing.
- **Model loading failures**: `setup()` raises `RuntimeError` if model download or initialization fails.
- **No automatic retries or auto-tuning**: The stage does not perform automatic batch size reduction or network retries.
- **Missing files**: `AudioBatch.validate()` may log file-existence warnings when code creates tasks; the stage does not auto-skip files.

## Performance Optimization

### GPU Memory Management

```python
# For large models or limited GPU memory
asr_stage = InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
).with_(
    batch_size=8,  # Reduce batch size
    resources=Resources(gpus=0.5)  # Share GPU resources
)
```

### Distributed Processing

```python
from nemo_curator.backends.xenna import XennaExecutor

# Configure executor (refer to Pipeline Execution Backends)
executor = XennaExecutor(
    config={
        "execution_mode": "streaming",
        "logging_interval": 60,
        "ignore_failures": False
    }
)
```

## Integration Examples

### Complete Audio-to-Text Pipeline

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage

pipeline = Pipeline(name="audio_to_text")

# ASR inference
pipeline.add_stage(asr_stage)

# Convert to text format
pipeline.add_stage(AudioToDocumentStage())

# Continue with text processing...
```

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

NeMo ASR Models <nemo-models.md>
```
