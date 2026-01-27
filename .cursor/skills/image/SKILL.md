---
name: image
description: |
  Process image datasets for training data curation. Use when the user wants
  to filter images by quality, detect NSFW content, generate embeddings, or
  deduplicate image datasets. Supports WebDataset format.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: image
  gpu-required: true
  nemo-curator-version: ">=0.5.0"
compatibility: Requires Python 3.10+, Linux, GPU with 4-8GB memory
disable-model-invocation: true
---

# Image Processing

Process image datasets for training data curation with quality filtering, NSFW detection, and deduplication.

## Platform Requirements

> **IMPORTANT**: NeMo Curator only runs on **Linux**. macOS and Windows are not supported.

## When to Use

- Filtering images by aesthetic quality
- Detecting and removing NSFW content
- Generating CLIP embeddings for images
- Deduplicating image datasets
- Preparing image datasets for multimodal models

## Quick Start

### 1. List Available Image Stages

```bash
python scripts/list_image_stages.py

# With descriptions
python scripts/list_image_stages.py --verbose
```

### 2. Estimate Resources

```bash
python scripts/estimate_image_resources.py --input-path /data/images
```

### 3. Generate Pipeline Config

```bash
python scripts/generate_image_config.py \
  --input-path /data/images \
  --output-path /data/curated \
  --aesthetic-threshold 0.5 \
  --nsfw-threshold 0.5 \
  --output-file image_pipeline.yaml
```

### 4. Execute Pipeline

```bash
python -m nemo_curator.config.run \
  --config-path=. \
  --config-name=image_pipeline
```

## Image Processing Workflow

```
+------------------------------------------------------------------+
|                    Image Curation Pipeline                        |
+------------------------------------------------------------------+
|                                                                   |
|  1. READ IMAGES                                                  |
|     - ImageReaderStage (from WebDataset tar files)               |
|                                                                   |
|  2. EMBEDDING                                                    |
|     - ImageEmbeddingStage (CLIP, GPU 4GB)                        |
|                                                                   |
|  3. FILTERING                                                    |
|     - ImageAestheticFilterStage (quality scoring, GPU 4GB)       |
|     - ImageNSFWFilterStage (safety filtering, GPU 4GB)           |
|                                                                   |
|  4. DEDUPLICATION (optional)                                     |
|     - ImageDuplicatesRemovalStage (embedding-based)              |
|                                                                   |
|  5. WRITE IMAGES                                                 |
|     - ImageWriterStage (WebDataset tar files)                    |
|                                                                   |
+------------------------------------------------------------------+
```

## Available Stages

| Stage | GPU Memory | Description |
|-------|------------|-------------|
| ImageReaderStage | 0 | Read images from WebDataset |
| ImageEmbeddingStage | 4GB | Generate CLIP embeddings |
| ImageAestheticFilterStage | 4GB | Filter by visual quality |
| ImageNSFWFilterStage | 4GB | Filter unsafe content |
| ImageDuplicatesRemovalStage | 0 | Remove duplicate images |
| ImageWriterStage | 0 | Write to WebDataset |

## Input Format

NeMo Curator expects images in WebDataset format (tar files):

```
dataset/
  shard_000000.tar
  shard_000001.tar
  ...
```

Each tar file contains image files and optional metadata:
```
image_001.jpg
image_001.json  # Optional metadata
image_002.jpg
image_002.json
...
```

## Pipeline Options

| Option | Description | Default |
|--------|-------------|---------|
| `--aesthetic-threshold` | Minimum aesthetic score (0-1) | 0.5 |
| `--nsfw-threshold` | Maximum NSFW probability (0-1) | 0.5 |
| `--dedup` | Enable deduplication | false |
| `--batch-size` | Images per batch | 100 |
| `--images-per-tar` | Output tar file size | 100 |

## Filtering Thresholds

### Aesthetic Score

Score from 0-1 indicating visual quality:

| Threshold | Result |
|-----------|--------|
| 0.3 | Keep most images |
| 0.5 | Balanced (recommended) |
| 0.7 | High quality only |
| 0.9 | Exceptional quality |

### NSFW Score

Probability of unsafe content:

| Threshold | Result |
|-----------|--------|
| 0.3 | Strict filtering |
| 0.5 | Moderate (recommended) |
| 0.7 | Permissive |

## Common Workflows

### Basic Quality Filtering

```bash
python scripts/generate_image_config.py \
  --input-path /data/images \
  --output-path /data/filtered \
  --aesthetic-threshold 0.5 \
  --no-dedup \
  --output-file quality_filter.yaml
```

### Full Curation Pipeline

```bash
python scripts/generate_image_config.py \
  --input-path /data/images \
  --output-path /data/curated \
  --aesthetic-threshold 0.5 \
  --nsfw-threshold 0.5 \
  --dedup \
  --output-file full_pipeline.yaml
```

### Embedding-Only Pipeline

```bash
python scripts/generate_image_config.py \
  --input-path /data/images \
  --output-path /data/embedded \
  --no-aesthetic \
  --no-nsfw \
  --output-file embed_only.yaml
```

## GPU Requirements

| Configuration | GPU Memory |
|---------------|------------|
| Single filter | 4GB |
| All filters | 4GB (shared models) |
| With dedup | 4GB |

Multiple filters can share GPU memory as they use the same CLIP model.

## References

- [FILTERING_THRESHOLDS.md](references/FILTERING_THRESHOLDS.md) - Detailed threshold guidance

## Related Skills

- `/curate` - Full curation workflow (includes image)
- `/stages` - All available stages
- `/setup` - Environment setup
