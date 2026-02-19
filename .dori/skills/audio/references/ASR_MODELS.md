# NeMo ASR Models

Guide for selecting ASR models for audio transcription in NeMo Curator.

## Verified Models

The following model has been verified in the NeMo Curator test suite:

| Model | Languages | Notes |
|-------|-----------|-------|
| `nvidia/parakeet-tdt-0.6b-v2` | English | Test-verified, recommended for getting started |

## Additional Models

NeMo Framework provides many pre-trained ASR models. For the complete list of available models and their specifications, refer to the [NeMo Framework ASR documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/all_chkpt.html).

Common model families include:
- **Parakeet**: Fast, accurate English models
- **FastConformer**: High-quality models with optional punctuation
- **Canary**: Multilingual support

## Model Architectures

### FastConformer Hybrid

- Best accuracy
- Supports punctuation and capitalization
- GPU: ~16GB recommended
- Speed: ~10x realtime

### FastConformer CTC

- Faster inference
- No punctuation
- GPU: ~12GB recommended
- Speed: ~15x realtime

## GPU Requirements

| Model Size | GPU Memory |
|------------|------------|
| Small (0.6B) | ~4GB |
| Large | 12-16GB recommended |

> **Note**: ASR inference can run on CPU but will be significantly slower.

## Usage in Pipeline

### Python API

```python
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.resources import Resources

stage = InferenceAsrNemoStage(
    model_name="nvidia/parakeet-tdt-0.6b-v2",
    filepath_key="audio_filepath",
    pred_text_key="transcript",
    batch_size=16,
)
```

### Stage Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | required | NeMo ASR model name from HuggingFace |
| `filepath_key` | str | `"audio_filepath"` | Key for audio file path in data |
| `pred_text_key` | str | `"pred_text"` | Output key for transcription |
| `batch_size` | int | `16` | Batch size for inference |

### GPU Configuration

```python
from nemo_curator.stages.resources import Resources

# GPU configuration
stage = InferenceAsrNemoStage(
    model_name="nvidia/parakeet-tdt-0.6b-v2",
    resources=Resources(cpus=1.0, gpu_memory_gb=4.0),
)
```

## Model Selection Tips

1. **Start with verified model** (`nvidia/parakeet-tdt-0.6b-v2`) for testing
2. **Check the NeMo Model Catalog** for production model selection
3. **Match language** to your dataset
4. **Check available GPU memory** before selecting larger models

## References

- [NeMo ASR Model Catalog](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/all_chkpt.html)
- [NeMo Framework Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/)
