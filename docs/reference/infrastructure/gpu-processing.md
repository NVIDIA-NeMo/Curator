---
description: "Guide to leveraging NVIDIA GPU acceleration in NeMo Curator for faster data processing and memory optimization"
categories: ["reference"]
tags: ["gpu-accelerated", "cuda", "rmm", "performance", "memory-management", "optimization"]
personas: ["mle-focused", "admin-focused"]
difficulty: "intermediate"
content_type: "reference"
modality: "universal"
---

(reference-infra-gpu-processing)=

# GPU Processing Guide

This guide explains how to use GPU acceleration in NVIDIA NeMo Curator for faster text data processing.

## Setting Up GPU Support

To use GPU acceleration, you'll need:

1. NVIDIA GPU with CUDA support
2. RAPIDS libraries installed (cuDF, RMM)
3. PyTorch with CUDA support for model inference

### Example: GPU-Accelerated Text Classification

```python
from nemo_curator.stages.text.classifiers import QualityClassifier
from nemo_curator.pipeline import Pipeline
from nemo_curator.tasks import DocumentBatch
import pandas as pd

# Create sample data
data = pd.DataFrame({
    "text": ["This is high quality text.", "Poor quality text here."]
})
batch = DocumentBatch(data=data, task_id="test_task", dataset_name="test_dataset")

# Set up GPU-accelerated classifier
classifier = QualityClassifier(
    model_inference_batch_size=256,
    autocast=True  # Enable mixed precision for faster inference
)

# Create and run pipeline
pipeline = Pipeline(name="test_pipeline")
pipeline.add_stage(classifier)
result = pipeline.run(initial_tasks=[batch])

print(result)
```

### Example: GPU-Accelerated Fuzzy Deduplication

```python
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

# Set up GPU-accelerated fuzzy deduplication
workflow = FuzzyDeduplicationWorkflow(
    input_path="/path/to/input/data",
    cache_path="/path/to/cache",
    output_path="/path/to/output",
    text_field="text",
    # GPU-accelerated MinHash parameters
    char_ngrams=24,
    num_bands=20,
    minhashes_per_band=13,
    use_64_bit_hash=False
)

# Run deduplication workflow
workflow.run()
```

## GPU-Accelerated Modules

NVIDIA NeMo Curator provides these GPU-accelerated modules:

### Data Processing

- **Exact deduplication**: GPU-optimized processing for duplicate detection
- **Fuzzy deduplication**: GPU-accelerated MinHash computation for approximate duplicates
- **Semantic deduplication**: GPU embeddings and similarity calculations for content-based deduplication

### Text Classification

- **Domain classification**: English and multilingual content categorization
- **Quality classification**: Content quality assessment using GPU-accelerated models
- **Safety models**: AEGIS and Instruction Data Guard for content safety evaluation
- **Educational content**: FineWeb models for educational value scoring
- **Content type classification**: Automatic content type detection
- **Task and complexity classification**: Instruction complexity assessment

