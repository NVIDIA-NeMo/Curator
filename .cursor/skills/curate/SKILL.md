---
name: curate
description: |
  Full data curation workflow for text, video, image, or audio datasets.
  Use when the user wants to build a complete curation pipeline, process
  data end-to-end, or mentions "curate", "clean", "preprocess", or
  "prepare training data". Supports all modalities with appropriate stages.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  nemo-curator-version: ">=0.5.0"
compatibility: Requires Python 3.10+, Ray cluster for distributed execution
disable-model-invocation: true
---

# Full Curation Workflow

## Platform Requirements

> **IMPORTANT**: NeMo Curator only runs on **Linux**. macOS and Windows are not supported.
>
> **On macOS/Windows, use Docker:**
> ```bash
> docker run --rm -v $(pwd):/workspace -w /workspace python:3.11-slim \
>   bash -c "pip install nemo-curator && python your_script.py"
> ```

| Platform | Support | Solution |
|----------|---------|----------|
| Linux | ✅ Native | Install directly with pip/uv |
| macOS | ❌ Not supported | Use Docker |
| Windows | ❌ Not supported | Use Docker or WSL2 |

## When to Use

- Building end-to-end curation pipelines
- User says "curate my data" or "prepare training data"
- Need filtering + classification + deduplication combined
- Processing large datasets for LLM training

## Quick Start

### 0. Check Platform

Before running, verify you're on Linux or have Docker available:

```python
import sys
if sys.platform != "linux":
    print("WARNING: NeMo Curator requires Linux. Use Docker on macOS/Windows.")
```

## Quick Start

### 1. Detect Modality

First, determine what kind of data you're working with:

```bash
# Use the curator-os script
python ../curator-os/scripts/detect_modality.py /path/to/data
```

### 2. Generate Pipeline Config

```bash
# For text data
python scripts/generate_yaml.py \
  --modality text \
  --input-path /data/raw \
  --output-path /data/curated \
  --output-file text_curation.yaml

# For video data
python scripts/generate_yaml.py \
  --modality video \
  --input-path /data/videos \
  --output-path /data/processed \
  --output-file video_curation.yaml
```

### 3. Execute Pipeline

```bash
python -m nemo_curator.config.run \
  --config-path=. \
  --config-name=text_curation
```

## Text Curation Workflow

Standard text curation includes:

```
┌──────────────────────────────────────────────────────────────────┐
│                    Text Curation Pipeline                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. READ DATA                                                    │
│     └── ParquetReader / JsonlReader                              │
│                                                                   │
│  2. HEURISTIC FILTERING (30+ filters available)                  │
│     ├── WordCountFilter (50-100000 words)                        │
│     ├── NonAlphaNumericFilter (< 25%)                            │
│     ├── SymbolsToWordsFilter (< 10%)                             │
│     ├── RepeatedLinesFilter (< 30%)                              │
│     └── ... (see /filter skill for full list)                    │
│                                                                   │
│  3. QUALITY CLASSIFICATION                                       │
│     ├── QualityClassifier (general quality)                      │
│     └── FineWebEduClassifier (educational content)               │
│                                                                   │
│  4. DEDUPLICATION                                                │
│     └── FuzzyDeduplicationWorkflow (MinHash + LSH)               │
│                                                                   │
│  5. WRITE DATA                                                   │
│     └── ParquetWriter                                            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Text Pipeline Options

| Option | Description | Default |
|--------|-------------|---------|
| `--filters` | Filter preset | `standard` |
| `--classify` | Run classifier | `quality` |
| `--dedup` | Deduplication method | `fuzzy` |
| `--edu-filter` | Filter by edu score | `none` |

### Filter Presets

| Preset | Filters | Use Case |
|--------|---------|----------|
| `minimal` | WordCount, NonAlphaNumeric | Quick pass |
| `standard` | 15 common filters | General use |
| `full` | All 25 filters | Thorough cleaning |
| `code` | Code-specific filters | Source code |

## Video Curation Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                    Video Curation Pipeline                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. READ VIDEOS                                                  │
│     └── VideoReader (stages.video.io.video_reader)               │
│                                                                   │
│  2. SCENE DETECTION / CLIPPING                                   │
│     ├── TransNetV2ClipExtractionStage (ML-based, GPU)            │
│     └── OR FixedStrideExtractorStage (fixed duration, CPU)       │
│                                                                   │
│  3. MOTION FILTERING (optional)                                  │
│     ├── MotionVectorDecodeStage (decode motion vectors)          │
│     └── MotionFilterStage (filter by motion threshold)           │
│                                                                   │
│  4. AESTHETIC FILTERING (optional)                               │
│     └── ClipAestheticFilterStage (CLIP aesthetic score)          │
│                                                                   │
│  5. CAPTIONING (optional)                                        │
│     ├── CaptionPreparationStage                                  │
│     ├── CaptionGenerationStage (Qwen VL, GPU)                    │
│     └── CaptionEnhancementStage (optional refinement)            │
│                                                                   │
│  6. EMBEDDING (optional)                                         │
│     ├── CosmosEmbed1FrameCreationStage + EmbeddingStage          │
│     └── OR InternVideo2FrameCreationStage + EmbeddingStage       │
│                                                                   │
│  7. WRITE CLIPS                                                  │
│     └── ClipWriterStage (stages.video.io.clip_writer)            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Video Pipeline Options

| Option | Description | Default |
|--------|-------------|---------|
| `--clip-method` | Scene detection method | `transnetv2` |
| `--caption` | Generate captions | `false` |
| `--embed` | Generate embeddings | `false` |
| `--filter-motion` | Filter static clips | `true` |

## Image Curation Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                    Image Curation Pipeline                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. READ IMAGES                                                  │
│     └── ImageReader                                              │
│                                                                   │
│  2. EMBEDDING                                                    │
│     └── ImageEmbeddingStage (CLIP)                               │
│                                                                   │
│  3. FILTERING                                                    │
│     ├── AestheticFilterStage                                     │
│     └── NSFWFilterStage                                          │
│                                                                   │
│  4. DEDUPLICATION (optional)                                     │
│     └── SemanticDeduplicationWorkflow (embedding-based)          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Audio Curation Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                    Audio Curation Pipeline                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. CREATE MANIFEST                                              │
│     └── CreateInitialManifestFleursStage (or custom manifest)    │
│                                                                   │
│  2. TRANSCRIPTION                                                │
│     └── InferenceAsrNemoStage (NeMo ASR, GPU)                    │
│                                                                   │
│  3. QUALITY METRICS                                              │
│     └── GetPairwiseWerStage (Word Error Rate)                    │
│                                                                   │
│  4. FILTERING                                                    │
│     └── PreserveByValueStage (filter by WER threshold)           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Asset Templates

Pre-configured YAML templates are available in `assets/`:

| Template | Description |
|----------|-------------|
| `text_curation_template.yaml` | Standard text curation with 10 heuristic filters |
| `video_curation_template.yaml` | Video curation with scene detection and motion filtering |

Use `generate_yaml.py` with `--filters full` to generate a text config with all 21 filters.

## Customization

### Adding Stages

To add a stage to a generated config:

```yaml
stages:
  # Existing stages...
  
  # Add new stage
  - _target_: nemo_curator.stages.text.classifiers.DomainClassifier
    model_inference_batch_size: 256
```

### Removing Stages

Comment out or delete unwanted stages from the generated YAML.

### Changing Parameters

Edit the stage configuration directly:

```yaml
# Change filter threshold (note: filters are wrapped in ScoreFilter)
- _target_: nemo_curator.stages.text.modules.ScoreFilter
  filter_obj:
    _target_: nemo_curator.stages.text.filters.WordCountFilter
    min_words: 100  # Changed from 50
    max_words: 50000  # Changed from 100000
  text_field: ${text_field}
```

## Common Workflows

### High-Quality LLM Training Data

```bash
python scripts/generate_yaml.py \
  --modality text \
  --input-path /data/raw \
  --output-path /data/curated \
  --filters full \
  --classify quality,edu \
  --edu-filter 3 \
  --dedup fuzzy \
  --output-file llm_training.yaml
```

### Quick Quality Pass

```bash
python scripts/generate_yaml.py \
  --modality text \
  --input-path /data/raw \
  --output-path /data/filtered \
  --filters minimal \
  --no-dedup \
  --output-file quick_filter.yaml
```

### Video Dataset Preparation

```bash
python scripts/generate_yaml.py \
  --modality video \
  --input-path /data/videos \
  --output-path /data/clips \
  --caption \
  --embed \
  --output-file video_dataset.yaml
```

## Related Skills

- `/dedup-fuzzy` - Fuzzy deduplication details
- `/filter` - Detailed filter catalog
- `/classify` - Classifier options
- `/stages` - All available stages
