---
description: "Guide to using NeMo Framework's pretrained ASR models for speech recognition in audio curation pipelines"
categories: ["audio-processing"]
tags: ["nemo-models", "asr-models", "pretrained", "multilingual", "model-selection"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "audio-only"
---

# NeMo ASR Models

Use NeMo Framework's automatic speech recognition models for transcription in your audio curation pipelines. This guide covers basic usage and configuration.

## Model Selection

NeMo Framework provides pre-trained ASR models through the Hugging Face model hub. For the complete list of available models and their specifications, refer to the [NeMo Framework ASR documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/all_chkpt.html).

### Example Model Usage

```python
# Example using a test-verified model
example_model = "nvidia/parakeet-tdt-0.6b-v2"

# For production use, select appropriate models from:
# https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/all_chkpt.html
```

## Basic Usage

### Simple ASR Inference

```python
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.resources import Resources

# Create ASR inference stage with a model from NeMo Framework
asr_stage = InferenceAsrNemoStage(
    model_name="your_chosen_model_name",  # Select from NeMo Framework docs
    filepath_key="audio_filepath",
    pred_text_key="pred_text"
)

# Configure for GPU processing
asr_stage = asr_stage.with_(
    resources=Resources(gpus=1.0),
    batch_size=16
)
```

### Custom Configuration

```python
# Example with custom field names
custom_asr = InferenceAsrNemoStage(
    model_name="your_chosen_model_name",
    filepath_key="custom_audio_path",
    pred_text_key="transcription"
).with_(
    batch_size=32,
    resources=Resources(cpus=4.0, gpus=1.0)
)
```

## Model Caching

Models are automatically downloaded and cached when first loaded:

```python
# Models are cached automatically on first use
asr_stage = InferenceAsrNemoStage(model_name="your_chosen_model_name")

# The setup() method handles model downloading and caching
asr_stage.setup()
```

## Resource Configuration

Configure GPU and CPU resources based on your hardware:

```python
from nemo_curator.stages.resources import Resources

# Single GPU configuration
asr_stage = InferenceAsrNemoStage(
    model_name="your_chosen_model_name"
).with_(
    resources=Resources(
        cpus=4.0,
        gpu_memory_gb=8.0  # Adjust based on your model's requirements
    ),
    batch_size=16
)

# Multi-GPU configuration
multi_gpu_stage = InferenceAsrNemoStage(
    model_name="your_chosen_model_name"
).with_(
    resources=Resources(
        cpus=8.0,
        gpus=2.0  # Use 2 GPUs
    ),
    batch_size=32
)
```

:::{note}
Resource requirements vary by model. Test with your specific model to determine optimal settings.
:::

