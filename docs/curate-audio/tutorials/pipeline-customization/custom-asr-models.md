---
description: "Integrate custom or fine-tuned ASR models into NeMo Curator audio curation pipelines"
categories: ["tutorials"]
tags: ["custom-models", "fine-tuning", "model-integration", "nemo-toolkit", "domain-adaptation"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "tutorial"
modality: "audio-only"
---

# Custom ASR Model Integration

Learn how to integrate custom or fine-tuned ASR models into your NeMo Curator audio curation pipelines. This tutorial covers model preparation, integration strategies, and performance optimization for specialized speech recognition needs.

## When to Use Custom Models

### Use Cases for Custom ASR Models

- **Domain Adaptation**: Medical, legal, technical terminology
- **Accent Adaptation**: Regional accents, non-native speakers
- **Language Variants**: Dialects, code-switching, multilingual
- **Quality Enhancement**: Improved accuracy for specific audio conditions
- **Specialized Vocabulary**: Industry-specific jargon, proper nouns

## Model Preparation

### Fine-Tuning NeMo ASR Models

```python
# Example fine-tuning configuration for domain adaptation (conceptual)
from nemo.collections.asr.models import ASRModel

def fine_tune_nemo_model(base_model: str, training_manifest: str,
                         output_dir: str, domain_name: str) -> str:
    """Fine-tune NeMo ASR model for specific domain (conceptual example)."""

    # Load pretrained model
    asr_model = ASRModel.from_pretrained(base_model)

    # Configure fine-tuning (placeholder - training setup omitted)
    fine_tune_config = {
        "model": asr_model,
        "train_manifest": training_manifest,
        "validation_manifest": training_manifest.replace("train", "dev"),
        "batch_size": 16,
        "num_epochs": 10,
        "learning_rate": 1e-5,
        "output_dir": output_dir,
        "experiment_name": f"domain_adaptation_{domain_name}",
    }

    # Pseudo-code for training (refer to NeMo ASR training guide)
    # trainer = ASRTrainer(config=fine_tune_config)
    # trainer.fit()

    # Return path to a hypothetical fine-tuned model
    return f"{output_dir}/domain_adaptation_{domain_name}/checkpoints/final.nemo"

# Usage
custom_model_path = fine_tune_nemo_model(
    base_model="nvidia/stt_en_fastconformer_hybrid_large_pc",
    training_manifest="/data/medical_speech/train_manifest.jsonl",
    output_dir="/models/custom_asr",
    domain_name="medical",
)
```

### Converting External Models

```python
# Conceptual example: external model conversion requires nontrivial mapping.
# Refer to official NeMo documentation for supported conversion workflows.
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # pip install transformers
import torch

def convert_huggingface_to_nemo(hf_model_name: str, output_path: str) -> str:
    """Conceptual placeholder for converting a Hugging Face model to a NeMo-compatible artifact."""

    processor = WhisperProcessor.from_pretrained(hf_model_name)
    model = WhisperForConditionalGeneration.from_pretrained(hf_model_name)

    # This does NOT produce a true .nemo file; shown for illustration only.
    torch.save({
        "model_name": hf_model_name.replace("/", "_"),
        "processor": processor,
        "model": model,
    }, f"{output_path}/converted_model.pt")

    return f"{output_path}/converted_model.pt"

# Usage (illustrative)
converted_model = convert_huggingface_to_nemo(
    hf_model_name="openai/whisper-large-v3",
    output_path="/models/converted_whisper",
)
```

## Integration Strategies

### Custom Model Stage

```python
from dataclasses import dataclass, field
from loguru import logger
import nemo.collections.asr as nemo_asr
from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage

@dataclass
class CustomAsrStage(InferenceAsrNemoStage):
    """Custom ASR stage with domain-specific preprocessing."""

    domain_name: str = "general"
    preprocessing_config: dict = field(default_factory=dict)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup custom ASR model with domain-specific configuration."""
        try:
            if self.model_name.endswith(".nemo"):
                self.asr_model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=self.model_name,
                    map_location=self.check_cuda(),
                )
            else:
                super().setup(worker_metadata)

            if self.domain_name == "medical":
                self._setup_medical_preprocessing()
            elif self.domain_name == "legal":
                self._setup_legal_preprocessing()
        except Exception as e:
            logger.error(f"Failed to setup custom ASR model: {e}")
            raise

    def _setup_medical_preprocessing(self) -> None:
        medical_vocab = [
            "diagnosis", "treatment", "medication", "symptoms",
            "patient", "doctor", "prescription", "therapy",
        ]
        if hasattr(self.asr_model, "add_vocabulary"):
            self.asr_model.add_vocabulary(medical_vocab)

    def _setup_legal_preprocessing(self) -> None:
        legal_vocab = [
            "plaintiff", "defendant", "objection", "sustained",
            "contract", "liability", "evidence", "testimony",
        ]
        if hasattr(self.asr_model, "add_vocabulary"):
            self.asr_model.add_vocabulary(legal_vocab)

    def transcribe(self, files: list[str]) -> list[str]:
        """Transcribe with optional domain-specific preprocessing and post-processing."""
        selected_files = self._preprocess_audio_files(files) if self.domain_name in {"medical", "legal"} else files
        texts = super().transcribe(selected_files)
        return [self._postprocess_transcription(t) for t in texts]

    def _preprocess_audio_files(self, files: list[str]) -> list[str]:
        return files

    def _postprocess_transcription(self, text: str) -> str:
        if self.domain_name == "medical":
            medical_abbreviations = {
                "bp": "blood pressure",
                "hr": "heart rate",
                "temp": "temperature",
                "dx": "diagnosis",
            }
            for abbrev, expansion in medical_abbreviations.items():
                text = text.replace(f" {abbrev} ", f" {expansion} ")
        return text
```

### Model Ensemble Integration

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

def create_ensemble_asr_pipeline(model_configs: list[dict]) -> Pipeline:
    """Create ASR pipeline using multiple custom models."""

    pipeline = Pipeline(name="ensemble_custom_asr")

    # Process with each model (assumes prior reader stage in user pipeline)
    for config in model_configs:
        custom_asr = CustomAsrStage(
            model_name=config["model_path"],
            domain_name=config.get("domain", "general"),
            pred_text_key=f"pred_text_{config['name']}",
            preprocessing_config=config.get("preprocessing", {}),
        ).with_(resources=Resources(gpus=config.get("gpu_allocation", 0.5)))

        pipeline.add_stage(custom_asr)

    # Ensemble combination
    @processing_stage(name="ensemble_combination")
    def combine_model_outputs(audio_batch: AudioBatch) -> AudioBatch:
        for item in audio_batch.data:
            predictions: list[str] = []
            confidences: list[float] = []
            for config in model_configs:
                pred_key = f"pred_text_{config['name']}"
                if pred_key in item:
                    predictions.append(item[pred_key])
                    confidences.append(float(config.get("weight", 1.0)))
            if predictions:
                item["ensemble_pred_text"] = combine_predictions(predictions, confidences)
                item["ensemble_model_count"] = len(predictions)
        return audio_batch

    pipeline.add_stage(combine_model_outputs)
    return pipeline

def combine_predictions(predictions: list[str], weights: list[float]) -> str:
    """Combine multiple ASR predictions (simple weighted pick)."""
    if len(predictions) == 1:
        return predictions[0]
    max_weight_idx = weights.index(max(weights))
    return predictions[max_weight_idx]
```

## Model Performance Optimization

### Custom Model Benchmarking

```python
import time
import numpy as np
import torch
from nemo_curator.tasks import AudioBatch
from nemo_curator.stages.audio.metrics.get_wer import get_wer

def benchmark_custom_model(model_path: str, test_audio_data: list[dict]) -> dict:
    """Benchmark custom ASR model performance."""

    custom_asr = CustomAsrStage(model_name=model_path)
    custom_asr.setup()

    benchmark_results = {
        "model_info": {"model_path": model_path, "test_samples": len(test_audio_data)},
        "performance": {},
        "quality": {},
        "resource_usage": {},
    }

    test_batch = AudioBatch(data=test_audio_data, filepath_key="audio_filepath")

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = custom_asr.process(test_batch)

    inference_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

    total_audio_duration = sum(item.get("duration", 0.0) for item in test_audio_data)
    benchmark_results["performance"] = {
        "inference_time": inference_time,
        "real_time_factor": (total_audio_duration / inference_time) if inference_time > 0 else 0.0,
        "samples_per_second": (len(test_audio_data) / inference_time) if inference_time > 0 else 0.0,
        "peak_gpu_memory_gb": peak_memory,
    }

    if all("text" in item for item in test_audio_data):
        wer_values: list[float] = []
        for item in result.data:
            if "pred_text" in item and "text" in item:
                wer_values.append(get_wer(item["text"], item["pred_text"]))
        if wer_values:
            benchmark_results["quality"] = {
                "mean_wer": float(np.mean(wer_values)),
                "median_wer": float(np.median(wer_values)),
                "wer_std": float(np.std(wer_values)),
                # WER is a percentage; thresholds below use percentage values
                "excellent_samples": sum(1 for wer in wer_values if wer <= 10),
                "good_samples": sum(1 for wer in wer_values if wer <= 25),
            }

    return benchmark_results
```

### Model Comparison Framework

```python
import pandas as pd
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage

def compare_asr_models(model_configs: list[dict], test_data: list[dict]) -> pd.DataFrame:
    """Compare performance of multiple ASR models."""

    comparison_results: list[dict] = []

    for config in model_configs:
        print(f"Benchmarking {config['name']}...")

        # Explicitly setup stages when benchmarking outside a pipeline
        if config["type"] == "custom":
            asr_stage = CustomAsrStage(model_name=config["model_path"], domain_name=config.get("domain", "general"))
        else:
            asr_stage = InferenceAsrNemoStage(model_name=config["model_path"])
        asr_stage.setup()

        results = benchmark_custom_model(config["model_path"], test_data)

        comparison_entry = {
            "model_name": config["name"],
            "model_type": config["type"],
            "domain": config.get("domain", "general"),
            "real_time_factor": results["performance"].get("real_time_factor", 0.0),
            "samples_per_second": results["performance"].get("samples_per_second", 0.0),
            "gpu_memory_gb": results["performance"].get("peak_gpu_memory_gb", 0.0),
            "mean_wer": results.get("quality", {}).get("mean_wer", None),
            "excellent_ratio": (results.get("quality", {}).get("excellent_samples", 0) / max(1, len(test_data))),
            "memory_per_sample": results["performance"].get("peak_gpu_memory_gb", 0.0) / max(1, len(test_data)),
        }

        comparison_results.append(comparison_entry)

    return pd.DataFrame(comparison_results)

# Usage
model_configs = [
    {"name": "Base English Large", "type": "pretrained", "model_path": "nvidia/stt_en_fastconformer_hybrid_large_pc"},
    {"name": "Medical Custom", "type": "custom", "model_path": "/models/medical_asr.nemo", "domain": "medical"},
    {"name": "Legal Custom", "type": "custom", "model_path": "/models/legal_asr.nemo", "domain": "legal"},
]

comparison_df = compare_asr_models(model_configs, test_audio_samples)
print(comparison_df)
```

## Advanced Integration Patterns

### Conditional Model Selection

```python
from dataclasses import dataclass, field
from loguru import logger
import nemo.collections.asr as nemo_asr
from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioBatch

@dataclass
class ConditionalAsrStage(ProcessingStage[AudioBatch, AudioBatch]):
    """Select ASR model based on audio characteristics."""

    filepath_key: str = "audio_filepath"
    model_configs: dict[str, str] = field(
        default_factory=lambda: {
            "general": "nvidia/stt_en_fastconformer_hybrid_large_pc",
            "telephony": "nvidia/stt_en_fastconformer_telephony_large",
            "noisy": "nvidia/stt_en_fastconformer_noisy_large",
            "medical": "/models/medical_asr.nemo",
            "legal": "/models/legal_asr.nemo",
        }
    )

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Load all configured models."""
        self.loaded_models: dict[str, any] = {}
        for domain, model_path in self.model_configs.items():
            try:
                if model_path.endswith(".nemo"):
                    model = nemo_asr.models.ASRModel.restore_from(
                        restore_path=model_path, map_location=self._device()
                    )
                else:
                    model = nemo_asr.models.ASRModel.from_pretrained(
                        model_name=model_path, map_location=self._device()
                    )
                self.loaded_models[domain] = model
                logger.info(f"Loaded {domain} model: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load {domain} model: {e}")

    def _device(self):  # simple helper mirroring InferenceAsrNemoStage.check_cuda
        import torch
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def process(self, task: AudioBatch) -> AudioBatch:
        processed_data: list[dict] = []
        for item in task.data:
            domain = self._select_model_domain(item)
            if domain in self.loaded_models:
                model = self.loaded_models[domain]
                audio_file = item[self.filepath_key]
                output = model.transcribe([audio_file])[0]
                item["pred_text"] = output.text
                item["selected_model"] = domain
                item["model_path"] = self.model_configs[domain]
            else:
                item["pred_text"] = "ERROR: No suitable model"
                item["selected_model"] = "none"
            processed_data.append(item)

        return AudioBatch(
            data=processed_data,
            filepath_key=task.filepath_key,
            task_id=task.task_id,
            dataset_name=task.dataset_name,
        )

    def _select_model_domain(self, audio_item: dict) -> str:
        domain_hints = audio_item.get("domain", "")
        file_path = audio_item.get("audio_filepath", "")
        duration = audio_item.get("duration", 0.0)
        if "medical" in domain_hints.lower() or "medical" in file_path.lower():
            return "medical"
        if "legal" in domain_hints.lower() or "legal" in file_path.lower():
            return "legal"
        if "phone" in file_path.lower() or "call" in file_path.lower():
            return "telephony"
        if duration < 2.0:
            return "noisy"
        return "general"
```

### Hierarchical Model Processing

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.tasks import AudioBatch

def create_hierarchical_asr_pipeline() -> Pipeline:
    """Create pipeline with hierarchical model processing."""

    pipeline = Pipeline(name="hierarchical_asr")

    pipeline.add_stage(
        InferenceAsrNemoStage(
            model_name="nvidia/stt_en_fastconformer_hybrid_small_pc",
            pred_text_key="screening_pred",
        ).with_(batch_size=64)
    )

    pipeline.add_stage(
        GetPairwiseWerStage(text_key="text", pred_text_key="screening_pred", wer_key="screening_wer")
    )

    @processing_stage(name="conditional_high_quality")
    def conditional_processing(audio_batch: AudioBatch) -> AudioBatch:
        high_priority: list[dict] = []
        standard_priority: list[dict] = []
        for item in audio_batch.data:
            screening_wer = item.get("screening_wer", 100.0)
            if screening_wer <= 40.0:
                high_priority.append(item)
            else:
                standard_priority.append(item)

        if high_priority:
            custom_asr = CustomAsrStage(model_name="/models/domain_specific.nemo", pred_text_key="final_pred")
            custom_asr.setup()
            high_priority_batch = AudioBatch(data=high_priority, filepath_key=audio_batch.filepath_key)
            processed_high = custom_asr.process(high_priority_batch)
            for item in processed_high.data:
                item["processing_tier"] = "high_quality"

        if standard_priority:
            standard_asr = InferenceAsrNemoStage(
                model_name="nvidia/stt_en_fastconformer_hybrid_large_pc", pred_text_key="final_pred"
            )
            standard_asr.setup()
            standard_batch = AudioBatch(data=standard_priority, filepath_key=audio_batch.filepath_key)
            processed_standard = standard_asr.process(standard_batch)
            for item in processed_standard.data:
                item["processing_tier"] = "standard"

        all_processed: list[dict] = []
        if high_priority:
            all_processed.extend(processed_high.data)
        if standard_priority:
            all_processed.extend(processed_standard.data)

        return AudioBatch(
            data=all_processed,
            filepath_key=audio_batch.filepath_key,
            task_id=audio_batch.task_id,
            dataset_name=audio_batch.dataset_name,
        )

    pipeline.add_stage(conditional_processing)
    return pipeline
```

## Model Validation and Testing

### Custom Model Validation

```python
def validate_custom_model(model_path: str, validation_data: list[dict]) -> dict:
    """Comprehensive validation of custom ASR model."""
    
    validation_results = {
        "model_info": {
            "model_path": model_path,
            "validation_samples": len(validation_data)
        },
        "accuracy_metrics": {},
        "robustness_tests": {},
        "performance_tests": {},
        "recommendations": []
    }
    
    # Setup custom model
    custom_asr = CustomAsrStage(model_name=model_path)
    custom_asr.setup()
    
    # Accuracy testing
    test_batch = AudioBatch(data=validation_data, filepath_key="audio_filepath")
    results = custom_asr.process(test_batch)
    
    # Calculate accuracy metrics
    wer_values = []
    for item in results.data:
        if "text" in item and "pred_text" in item:
            wer = get_wer(item["text"], item["pred_text"])
            wer_values.append(wer)
    
    if wer_values:
        validation_results["accuracy_metrics"] = {
            "mean_wer": np.mean(wer_values),
            "median_wer": np.median(wer_values),
            "wer_std": np.std(wer_values),
            "excellent_rate": sum(1 for wer in wer_values if wer <= 10) / len(wer_values),
            "good_rate": sum(1 for wer in wer_values if wer <= 25) / len(wer_values)
        }
    
    # Robustness testing
    robustness_tests = {
        "short_audio": [item for item in validation_data if item["duration"] < 2.0],
        "long_audio": [item for item in validation_data if item["duration"] > 15.0],
        "noisy_audio": [item for item in validation_data if "noisy" in item.get("conditions", "")]
    }
    
    for test_name, test_samples in robustness_tests.items():
        if test_samples:
            test_batch = AudioBatch(data=test_samples, filepath_key="audio_filepath")
            test_results = custom_asr.process(test_batch)
            
            test_wers = [
                get_wer(item["text"], item["pred_text"])
                for item in test_results.data
                if "text" in item and "pred_text" in item
            ]
            
            if test_wers:
                validation_results["robustness_tests"][test_name] = {
                    "sample_count": len(test_samples),
                    "mean_wer": np.mean(test_wers),
                    "degradation": np.mean(test_wers) - validation_results["accuracy_metrics"]["mean_wer"]
                }
    
    # Generate recommendations
    recommendations = []
    
    if validation_results["accuracy_metrics"]["mean_wer"] > 20:
        recommendations.append("Consider additional fine-tuning or larger base model")
    
    if validation_results["robustness_tests"].get("short_audio", {}).get("degradation", 0) > 10:
        recommendations.append("Model struggles with short audio - consider duration filtering")
    
    validation_results["recommendations"] = recommendations
    
    return validation_results
```

## Deployment Strategies

### Production Model Deployment

```python
import time
import numpy as np
from loguru import logger

class ProductionCustomAsrStage(CustomAsrStage):
    """Production-ready custom ASR stage with monitoring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_monitor = {
            "total_processed": 0,
            "total_time": 0.0,
            "error_count": 0,
            "quality_scores": [],
        }

    def process(self, task: AudioBatch) -> AudioBatch:
        start_time = time.time()
        try:
            result = super().process(task)
            processing_time = time.time() - start_time
            self.performance_monitor["total_processed"] += len(task.data)
            self.performance_monitor["total_time"] += processing_time
            if result.data and "wer" in result.data[0]:
                self.performance_monitor["quality_scores"].extend([item.get("wer", 0.0) for item in result.data])
            return result
        except Exception as e:
            self.performance_monitor["error_count"] += 1
            logger.error(f"Custom ASR processing failed: {e}")
            for item in task.data:
                item["processing_error"] = str(e)
                item["pred_text"] = ""
            return task

    def get_performance_summary(self) -> dict:
        monitor = self.performance_monitor
        if monitor["total_processed"] > 0 and monitor["total_time"] > 0:
            summary = {
                "total_samples": monitor["total_processed"],
                "average_processing_time": monitor["total_time"] / monitor["total_processed"],
                "error_rate": monitor["error_count"] / monitor["total_processed"],
                "samples_per_second": monitor["total_processed"] / monitor["total_time"],
            }
            if monitor["quality_scores"]:
                summary["quality_metrics"] = {
                    "mean_wer": float(np.mean(monitor["quality_scores"])),
                    "quality_trend": "stable",
                }
            return summary
        return {"status": "no_data"}
```

### Model Versioning and Updates

```python
import os
from pathlib import Path

class VersionedCustomAsrStage(CustomAsrStage):
    """Custom ASR stage with model versioning support."""

    def __init__(self, model_name: str, model_version: str = "latest", **kwargs):
        self.model_version = model_version
        self.versioned_model_path = self._resolve_model_path(model_name, model_version)
        super().__init__(model_name=self.versioned_model_path, **kwargs)

    def _resolve_model_path(self, model_name: str, version: str) -> str:
        if version == "latest":
            model_dir = Path(f"/models/{model_name}")
            if model_dir.exists():
                version_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                if version_dirs:
                    latest_version = max(version_dirs, key=lambda x: x.stat().st_mtime)
                    return str(latest_version / "model.nemo")
        versioned_path = f"/models/{model_name}/{version}/model.nemo"
        if os.path.exists(versioned_path):
            return versioned_path
        raise FileNotFoundError(f"Model version not found: {versioned_path}")

    def validate_model_version(self) -> dict:
        info = {
            "model_path": self.versioned_model_path,
            "version": self.model_version,
            "exists": os.path.exists(self.versioned_model_path),
            "size_mb": 0.0,
            "last_modified": None,
        }
        if info["exists"]:
            stat_info = os.stat(self.versioned_model_path)
            info["size_mb"] = stat_info.st_size / (1024 * 1024)
            info["last_modified"] = stat_info.st_mtime
        return info
```

## Best Practices

### Custom Model Development

1. **Incremental Development**: Start with pretrained models and fine-tune gradually
2. **Domain-Specific Data**: Use high-quality, domain-relevant training data
3. **Validation Strategy**: Maintain separate validation sets for each domain
4. **Performance Baseline**: Compare against relevant pretrained models

### Integration Guidelines

1. **Error Handling**: Implement robust error handling for model loading failures
2. **Resource Management**: Monitor GPU memory usage with custom models
3. **Fallback Strategies**: Provide fallback to standard models when custom models fail
4. **Performance Monitoring**: Track inference speed and quality metrics

### Production Considerations

1. **Model Versioning**: Implement proper model version control
2. **A/B Testing**: Gradually roll out custom models with comparison testing
3. **Monitoring**: Continuously monitor performance and quality metrics
4. **Update Strategy**: Plan for model updates and backward compatibility

## Troubleshooting

### Common Issues

**Model Loading Failures**:
```python
# Debug model loading
try:
    model = nemo_asr.models.ASRModel.restore_from(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
    # Check file permissions, NeMo version compatibility
```

**Performance Degradation**:
```python
# Compare with baseline
baseline_results = benchmark_custom_model(baseline_model, test_data)
custom_results = benchmark_custom_model(custom_model, test_data)

performance_ratio = (
    custom_results["performance"]["real_time_factor"] /
    baseline_results["performance"]["real_time_factor"]
)

if performance_ratio < 0.5:
    print("Warning: Custom model significantly slower than baseline")
```

**Quality Issues**:
```python
# Analyze quality degradation
if custom_results["quality"]["mean_wer"] > baseline_results["quality"]["mean_wer"] + 5:
    print("Custom model quality below baseline - review training data")
```

## Related Topics

- **[ASR Inference Overview](../../process-data/asr-inference/index.md)** - Standard ASR processing workflow
- **[NeMo ASR Models](../../process-data/asr-inference/nemo-models.md)** - Pretrained model catalog
- **[Custom Quality Filters](custom-filters.md)** - Custom filtering for specialized models
- **[NeMo Framework ASR Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)** - Complete ASR training guide
