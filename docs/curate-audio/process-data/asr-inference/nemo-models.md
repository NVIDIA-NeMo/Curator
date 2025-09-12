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

Leverage NeMo Framework's state-of-the-art automatic speech recognition models for accurate transcription in your audio curation pipelines. This guide covers model selection, configuration, and optimization strategies.

## Available Models

NeMo Framework provides a comprehensive collection of pre-trained ASR models for various languages, domains, and use cases.

### English Models

::::{tab-set}

:::{tab-item} General Purpose

```python
english_models = {
    "large": "nvidia/stt_en_fastconformer_hybrid_large_pc",      # Best accuracy
    "medium": "nvidia/stt_en_fastconformer_hybrid_medium_pc",    # Balanced
    "small": "nvidia/stt_en_fastconformer_hybrid_small_pc"       # Fastest
}
```

:::

:::{tab-item} Domain-Specific

```python
domain_models = {
    "telephony": "nvidia/stt_en_fastconformer_telephony_large",
    "streaming": "nvidia/stt_en_fastconformer_streaming_large", 
    "noisy": "nvidia/stt_en_fastconformer_noisy_large",
    "multilingual": "nvidia/stt_multilingual_fastconformer_hybrid_large_pc"
}
```

:::

::::

### Multilingual Models

::::{tab-set}

:::{tab-item} Language-Specific Models

```python
language_models = {
    "spanish": "nvidia/stt_es_fastconformer_hybrid_large_pc",
    "german": "nvidia/stt_de_fastconformer_hybrid_large_pc", 
    "french": "nvidia/stt_fr_fastconformer_hybrid_large_pc",
    "italian": "nvidia/stt_it_fastconformer_hybrid_large_pc",
    "armenian": "nvidia/stt_hy_fastconformer_hybrid_large_pc",
    "chinese": "nvidia/stt_zh_fastconformer_hybrid_large_pc"
}
```

:::

:::{tab-item} Multilingual Models

```python
multilingual_models = {
    "general": "nvidia/stt_multilingual_fastconformer_hybrid_large_pc",
    "streaming": "nvidia/stt_multilingual_fastconformer_streaming_large"
}
```

:::

::::

## Model Selection Guide

::::{tab-set}

:::{tab-item} By Accuracy Requirements

**Production Quality** (Highest Accuracy):

```python
# Use large models for best results
production_model = InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
).with_(
    batch_size=8,  # Smaller batches for large models
    resources=Resources(gpus=1.0)
)
```

**Balanced Performance**:

```python
# Medium models for good accuracy with reasonable speed
balanced_model = InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_medium_pc"
).with_(
    batch_size=16,
    resources=Resources(gpus=1.0)
)
```

**High Throughput** (Speed Priority):

```python
# Small models for maximum processing speed
fast_model = InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_small_pc"
).with_(
    batch_size=32,  # Larger batches for small models
    resources=Resources(gpus=1.0)
)
```

:::

:::{tab-item} By Domain

**Telephony and Call Center Data**:

```python
telephony_asr = InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_telephony_large"
)
# Optimized for 8kHz audio, background noise, compression artifacts
```

**Streaming and Real-time Applications**:

```python
streaming_asr = InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_streaming_large"
)
# Optimized for low-latency, incremental processing
```

**Noisy Environments**:

```python
noisy_asr = InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_noisy_large"
)
# Robust to background noise, music, multiple speakers
```

:::

:::{tab-item} By Language

**Single Language Processing**:

```python
def get_language_asr_model(language_code: str) -> str:
    """Get optimal ASR model for specific language."""
    
    language_model_map = {
        "en": "nvidia/stt_en_fastconformer_hybrid_large_pc",
        "es": "nvidia/stt_es_fastconformer_hybrid_large_pc",
        "de": "nvidia/stt_de_fastconformer_hybrid_large_pc",
        "fr": "nvidia/stt_fr_fastconformer_hybrid_large_pc",
        "it": "nvidia/stt_it_fastconformer_hybrid_large_pc",
        "hy": "nvidia/stt_hy_fastconformer_hybrid_large_pc",
        "zh": "nvidia/stt_zh_fastconformer_hybrid_large_pc"
    }
    
    # Fallback to multilingual model
    return language_model_map.get(
        language_code, 
        "nvidia/stt_multilingual_fastconformer_hybrid_large_pc"
    )

# Usage
spanish_asr = InferenceAsrNemoStage(
    model_name=get_language_asr_model("es")
)
```

**Multilingual Processing**:

```python
# Process multiple languages with single model
multilingual_asr = InferenceAsrNemoStage(
    model_name="nvidia/stt_multilingual_fastconformer_hybrid_large_pc"
)
# Supports 100+ languages with good accuracy
```

:::

::::

## Model Configuration

::::{tab-set}

:::{tab-item} Resource Allocation

```python
from nemo_curator.stages.resources import Resources

# GPU configuration for different model sizes
model_resources = {
    "large_model": Resources(
        cpus=4.0,
        gpu_memory_gb=16.0
    ),
    "medium_model": Resources(
        cpus=2.0,
        gpu_memory_gb=8.0
    ),
    "small_model": Resources(
        cpus=1.0,
        gpu_memory_gb=4.0
    )
}

# Apply resource configuration
asr_stage = InferenceAsrNemoStage(
    model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
).with_(
    resources=model_resources["large_model"],
    batch_size=8
)
```

:::

:::{tab-item} Batch Size Optimization

```python
# Model size vs. batch size recommendations
batch_size_guide = {
    "small_model": {
        "gpu_8gb": 64,
        "gpu_16gb": 128,
        "gpu_24gb": 256
    },
    "medium_model": {
        "gpu_8gb": 32,
        "gpu_16gb": 64, 
        "gpu_24gb": 128
    },
    "large_model": {
        "gpu_8gb": 16,
        "gpu_16gb": 32,
        "gpu_24gb": 64
    }
}

def get_optimal_batch_size(model_size: str, gpu_memory: str) -> int:
    """Get optimal batch size for model and hardware combination."""
    return batch_size_guide.get(model_size, {}).get(gpu_memory, 16)
```

:::

::::

## Performance Optimization

::::{tab-set}

:::{tab-item} Model Caching

```python
# Preload models to avoid repeated downloads
def setup_model_cache(model_names: list[str], cache_dir: str = "~/.cache/nemo"):
    """Pre-download and cache NeMo models."""
    
    import nemo.collections.asr as nemo_asr
    
    for model_name in model_names:
        try:
            print(f"Downloading {model_name}...")
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_name,
                map_location="cpu"  # Download only; do not load to GPU
            )
            print(f"Cached {model_name}")
            
        except Exception as e:
            print(f"Failed to cache {model_name}: {e}")

# Cache commonly used models
setup_model_cache([
    "nvidia/stt_en_fastconformer_hybrid_large_pc",
    "nvidia/stt_multilingual_fastconformer_hybrid_large_pc"
])
```

:::

:::{tab-item} Multi-GPU Processing

```python
# Parallel execution is backend-controlled. Define a single-stage pipeline and
# configure parallelism in the executor/backend configuration.
def create_asr_pipeline(model_name: str) -> Pipeline:
    pipeline = Pipeline(name="asr")
    pipeline.add_stage(
        InferenceAsrNemoStage(model_name=model_name).with_(
            resources=Resources(gpus=1.0)
        )
    )
    return pipeline
```

:::

::::

## Model Comparison

::::{tab-set}

:::{tab-item} Accuracy Benchmarks

Typical WER performance on clean speech:

| Model Size | English WER | Multilingual WER | Inference Speed |
|------------|-------------|------------------|-----------------|
| Small | 8-12% | 12-18% | 5x real-time |
| Medium | 6-9% | 10-15% | 3x real-time |
| Large | 4-7% | 8-12% | 1.5x real-time |

:::

:::{tab-item} Resource Requirements

GPU memory usage during inference:

| Model Size | GPU Memory | Batch Size 16 | Batch Size 32 |
|------------|------------|---------------|---------------|
| Small | 2-4 GB | 4 GB | 6 GB |
| Medium | 4-8 GB | 8 GB | 12 GB |
| Large | 8-12 GB | 12 GB | 18 GB |

:::

::::

## Custom Model Integration

::::{tab-set}

:::{tab-item} Fine-tuned Models

```python
# Use custom fine-tuned NeMo models
custom_asr = InferenceAsrNemoStage(
    model_name="/path/to/custom_model.nemo",  # Local .nemo file
    filepath_key="audio_filepath",
    pred_text_key="custom_pred_text"
)

# Or use Hugging Face models converted to NeMo format
hf_converted_asr = InferenceAsrNemoStage(
    model_name="path/to/converted_hf_model.nemo"
)
```

:::

:::{tab-item} Model Ensemble

```python
def create_ensemble_asr_pipeline(model_names: list[str]) -> Pipeline:
    """Create ensemble ASR pipeline using multiple models."""
    
    pipeline = Pipeline(name="ensemble_asr")
    
    # Load data once
    pipeline.add_stage(data_loading_stage)
    
    # Run inference with multiple models
    for i, model_name in enumerate(model_names):
        asr_stage = InferenceAsrNemoStage(
            model_name=model_name,
            pred_text_key=f"pred_text_{i}"
        )
        pipeline.add_stage(asr_stage)
    
    # Ensemble combination stage
    @processing_stage(name="ensemble_combination")
    def combine_predictions(audio_batch: AudioBatch) -> AudioBatch:
        for item in audio_batch.data:
            predictions = [
                item[f"pred_text_{i}"] 
                for i in range(len(model_names))
            ]
            
            # Simple voting or confidence-based combination
            item["ensemble_pred_text"] = combine_predictions_logic(predictions)
        
        return audio_batch
    
    pipeline.add_stage(combine_predictions)
    
    return pipeline
```

:::

::::

## Troubleshooting

::::{tab-set}

:::{tab-item} Network Connectivity

```bash
# Test model download manually
python -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained('nvidia/stt_en_fastconformer_hybrid_large_pc')
print('Model loaded successfully')
"
```

:::

:::{tab-item} GPU Memory Issues

```python
# Monitor GPU memory during model loading
import torch

def monitor_gpu_memory(stage_name: str):
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"{stage_name} - GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

# Use before and after model loading
monitor_gpu_memory("Before model loading")
asr_stage.setup()
monitor_gpu_memory("After model loading")
```

:::

:::{tab-item}  Model Compatibility

```python
# Check NeMo version compatibility
import nemo
print(f"NeMo version: {nemo.__version__}")

# Verify model exists
try:
    model_info = nemo_asr.models.ASRModel.list_available_models()
    if model_name in [model.pretrained_model_name for model in model_info]:
        print(f"Model {model_name} is available")
    else:
        print(f"Model {model_name} not found")
except Exception as e:
    print(f"Error checking model availability: {e}")
```

:::

::::


## Performance Tuning

::::{tab-set}

:::{tab-item} Model-Specific Optimization

```python
# Optimization settings by model type
optimization_configs = {
    "nvidia/stt_en_fastconformer_hybrid_large_pc": {
        "optimal_batch_size": 16,
        "min_gpu_memory_gb": 12.0,
        "recommended_precision": "fp16"
    },
    "nvidia/stt_multilingual_fastconformer_hybrid_large_pc": {
        "optimal_batch_size": 12,
        "min_gpu_memory_gb": 16.0, 
        "recommended_precision": "fp16"
    },
    "nvidia/stt_en_fastconformer_hybrid_small_pc": {
        "optimal_batch_size": 64,
        "min_gpu_memory_gb": 4.0,
        "recommended_precision": "fp32"
    }
}

def optimize_asr_stage(model_name: str) -> InferenceAsrNemoStage:
    """Create optimized ASR stage based on model characteristics."""
    
    config = optimization_configs.get(model_name, {
        "optimal_batch_size": 16,
        "min_gpu_memory_gb": 8.0
    })
    
    return InferenceAsrNemoStage(
        model_name=model_name
    ).with_(
        batch_size=config["optimal_batch_size"],
        resources=Resources(gpu_memory_gb=config["min_gpu_memory_gb"]) 
    )
```

:::

:::{tab-item} Dynamic Model Selection

```python
def select_model_by_dataset(dataset_characteristics: dict) -> str:
    """Select optimal model based on dataset characteristics."""
    
    # Factors to consider
    total_hours = dataset_characteristics.get("total_duration_hours", 0)
    languages = dataset_characteristics.get("languages", ["en"])
    domain = dataset_characteristics.get("domain", "general")
    quality_requirement = dataset_characteristics.get("quality", "balanced")
    
    # Decision logic
    if len(languages) > 1:
        # Multilingual dataset
        return "nvidia/stt_multilingual_fastconformer_hybrid_large_pc"
    
    elif domain == "telephony":
        return "nvidia/stt_en_fastconformer_telephony_large"
    
    elif quality_requirement == "highest":
        # Use largest model for best quality
        return "nvidia/stt_en_fastconformer_hybrid_large_pc"
    
    elif total_hours > 1000:
        # Large dataset - prioritize speed
        return "nvidia/stt_en_fastconformer_hybrid_medium_pc"
    
    else:
        # Default balanced choice
        return "nvidia/stt_en_fastconformer_hybrid_large_pc"
```

:::

::::
