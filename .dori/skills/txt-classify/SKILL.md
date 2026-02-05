---
name: txt-classify
description: Classify text documents using ML models for quality scoring, domain detection, educational content, and safety filtering. GPU required.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: text
  gpu-required: "true"
  parent-skill: text
---

# Text Classification Skill

Classify text documents using NeMo Curator's ML classifiers. GPU required.

## When This Skill Applies

- User wants to classify/score text quality
- User mentions: "classify", "quality score", "domain", "educational", "safety"
- User wants ML-based filtering (not heuristics)

## Important: GPU Required

All classifiers need GPU. Confirm user has:
1. GPU access (local, cloud, or cluster)
2. Docker with `--gpus all` if containerized

## Skill Workflow

### Step 1: Understand the Goal

Ask the user:
1. What do you want to classify? (quality, domain, safety, educational)
2. What will you do with results? (filter, sort, analyze)
3. Do you have GPU access?

### Step 2: Recommend Classifiers

| Goal | Classifier | Output Fields | Use Case |
|------|------------|---------------|----------|
| General quality | `QualityClassifier` | `quality_pred` (High/Medium/Low) | Filter low-quality web content |
| Educational | `FineWebEduClassifier` | `fineweb-edu-score-int` (0-5), `fineweb-edu-score-float`, `fineweb-edu-score-label` | Build educational datasets |
| Topic/domain | `DomainClassifier` | `domain_pred` | Route by topic |
| Content safety | `AegisClassifier` | `aegis_pred` (safe/unsafe/category) | Filter harmful content |
| Content type | `ContentTypeClassifier` | `content_pred` | Analyze content mix |

### Step 3: Explain Classifiers

**QualityClassifier**
- Output: `quality_pred` (High/Medium/Low)
- Parameters: `text_field`, `model_inference_batch_size` (default: 256), `filter_by` (optional list)
- GPU memory: ~4GB
- Fast, good for general web filtering

**FineWebEduClassifier**
- Output fields:
  - `fineweb-edu-score-label` (high_quality/low_quality) - default label field
  - `fineweb-edu-score-int` (0-5) - integer score
  - `fineweb-edu-score-float` (0.0-5.0) - raw float score
- Keep `fineweb-edu-score-int >= 3` for educational datasets
- GPU memory: ~4GB

**DomainClassifier**
- Output: `domain_pred` (News, Legal, Finance, etc.)
- GPU memory: ~4GB

**AegisClassifier** (requires HF token)
- Output: `aegis_pred` (safe, unsafe categories like violence, sexual, etc.)
- Requires access to `meta-llama/LlamaGuard-7b` on HuggingFace
- Pass token via `hf_token` parameter or `HF_TOKEN` environment variable
- GPU memory: ~16GB (loads LlamaGuard-7b base model)

### Step 4: Validate Before Generating

**Always validate the pipeline before generating code:**

```bash
python skills/shared/scripts/validate_pipeline.py \
  --stages "QualityClassifier,AegisClassifier" \
  --available-gpu 24 --json
```

This confirms:
- ✅ Type flow is valid
- ⚠️ GPU: 20GB required (4 + 16) - fits in 24GB
- ⚠️ Credential: HF_TOKEN required for AegisClassifier

### Step 5: Generate Pipeline Code

```python
# Text Classification Pipeline
# GPU Required: Yes

import json
import torch

if not torch.cuda.is_available():
    raise RuntimeError("GPU required for classification")

print(f"GPU: {torch.cuda.get_device_name(0)}")

from nemo_curator.stages.text.classifiers.quality import QualityClassifier
from nemo_curator.pipeline import Pipeline

def run_classification(input_path: str, output_path: str, text_field: str = "text"):
    with open(input_path) as f:
        docs = [json.loads(line) for line in f]
    
    print(f"Loaded {len(docs)} documents")
    
    classifier = QualityClassifier(
        text_field=text_field,
        model_inference_batch_size=256,
    )
    
    # Process
    import pandas as pd
    from nemo_curator.tasks import DocumentBatch
    
    df = pd.DataFrame(docs)
    task = DocumentBatch(task_id="classify", dataset_name="user_data", data=df)
    
    stages = classifier.decompose()
    for stage in stages:
        stage.setup({})
        task = stage.process(task)
    
    result_df = task.data
    
    with open(output_path, "w") as f:
        for record in result_df.to_dict('records'):
            f.write(json.dumps(record) + "\n")
    
    print("\nQuality distribution:")
    print(result_df["quality_pred"].value_counts())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()
    run_classification(args.input, args.output, args.text_field)
```

### Step 5: Filtering by Classification

```python
# Keep high-quality only
high_quality = [doc for doc in docs if doc.get("quality_pred") == "High"]

# Keep educational content (score >= 3)
educational = [doc for doc in docs if doc.get("fineweb-edu-score-int", 0) >= 3]

# Keep safe content only
safe = [doc for doc in docs if doc.get("aegis_pred") == "safe"]
```

**Alternative: Use `filter_by` parameter** (recommended for large datasets):

```python
# QualityClassifier with built-in filtering
classifier = QualityClassifier(
    text_field="text",
    filter_by=["High", "Medium"],  # Only keep High and Medium quality
)

# FineWebEduClassifier with built-in filtering
classifier = FineWebEduClassifier(
    text_field="text",
    filter_by=["high_quality"],  # Based on label field, not int score
)
```

## Available Classifiers

| Classifier | Import | Output Fields | GPU Memory |
|------------|--------|---------------|------------|
| `QualityClassifier` | `nemo_curator.stages.text.classifiers` | `quality_pred` | ~4GB |
| `FineWebEduClassifier` | `nemo_curator.stages.text.classifiers` | `fineweb-edu-score-label`, `fineweb-edu-score-int`, `fineweb-edu-score-float` | ~4GB |
| `DomainClassifier` | `nemo_curator.stages.text.classifiers` | `domain_pred` | ~4GB |
| `AegisClassifier` | `nemo_curator.stages.text.classifiers` | `aegis_pred` | ~16GB |
| `ContentTypeClassifier` | `nemo_curator.stages.text.classifiers` | `content_pred` | ~4GB |

> **Note**: All classifiers can be imported from `nemo_curator.stages.text.classifiers` directly.

## Test Before Full Run

**Always suggest testing on a small sample first:**

```bash
python skills/shared/scripts/test_pipeline.py \
  --stages "QualityClassifier" \
  --input /path/to/data.jsonl \
  --sample 20 --json
```

This verifies:
- Stage executes without errors
- New columns are created (`quality_pred`)
- Reasonable retention rate

## Execution

```bash
# Docker with GPU
docker run --gpus all --rm -v $(pwd):/data nvcr.io/nvidia/nemo-curator:latest \
    python /data/classify_pipeline.py --input /data/data.jsonl --output /data/classified.jsonl

# For AegisClassifier (needs HF token)
docker run --gpus all --rm -e HF_TOKEN=$HF_TOKEN -v $(pwd):/data nvcr.io/nvidia/nemo-curator:latest \
    python /data/classify_pipeline.py --input /data/data.jsonl --output /data/classified.jsonl
```

## Error Diagnosis

If user encounters errors, use the diagnosis tool:

```bash
python skills/shared/scripts/diagnose_error.py \
  --error "CUDA out of memory" \
  --context '{"pipeline": "QualityClassifier"}'
```
