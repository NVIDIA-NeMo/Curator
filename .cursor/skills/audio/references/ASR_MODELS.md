# NeMo ASR Models

Guide for selecting ASR models for audio transcription in NeMo Curator.

## Recommended Models

### English

| Model | Architecture | Use Case |
|-------|--------------|----------|
| `nvidia/stt_en_fastconformer_hybrid_large_pc` | FastConformer Hybrid | Best quality, punctuation |
| `nvidia/stt_en_fastconformer_ctc_large` | FastConformer CTC | Fast inference |
| `nvidia/stt_en_conformer_ctc_large` | Conformer CTC | Balanced |

### Other Languages

| Language | Model |
|----------|-------|
| German | `nvidia/stt_de_fastconformer_hybrid_large_pc` |
| Spanish | `nvidia/stt_es_fastconformer_hybrid_large_pc` |
| French | `nvidia/stt_fr_fastconformer_hybrid_large_pc` |
| Armenian | `nvidia/stt_hy_fastconformer_hybrid_large_pc` |
| Multilingual | `nvidia/stt_multilingual_fastconformer_hybrid_large_pc` |

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

### Conformer CTC

- Original architecture
- Widely tested
- GPU: ~12GB recommended

## GPU Requirements

| Model Size | GPU Memory |
|------------|------------|
| Large | 16GB recommended |
| Medium | 8-12GB |
| Small | 4-8GB |

> **Note**: ASR inference can run on CPU but will be significantly slower.
> Use `--gpu-memory-gb 0` in the config generator for CPU-only mode.

## Usage in Pipeline

### YAML Configuration

```yaml
- _target_: nemo_curator.stages.audio.inference.asr_nemo.InferenceAsrNemoStage
  model_name: nvidia/stt_en_fastconformer_hybrid_large_pc
  resources:
    _target_: nemo_curator.stages.resources.Resources
    cpus: 1.0
    gpu_memory_gb: 16.0
```

### Python API

```python
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.resources import Resources

stage = InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_large_pc",
    resources=Resources(cpus=1.0, gpu_memory_gb=16.0),
)
```

### CPU-Only Configuration

```yaml
- _target_: nemo_curator.stages.audio.inference.asr_nemo.InferenceAsrNemoStage
  model_name: nvidia/stt_en_fastconformer_hybrid_large_pc
  # Default resources (CPU only) when resources not specified
```

## Model Selection Tips

1. **Start with FastConformer Hybrid** for best quality
2. **Use CTC variants** for faster processing
3. **Match language** to your dataset
4. **Use multilingual** for mixed-language content
5. **Check available GPU memory** before selecting model size

## References

- [NeMo ASR Model Catalog](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/all_chkpt.html)
