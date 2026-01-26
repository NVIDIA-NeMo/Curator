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

Build complete data curation pipelines for any modality.

## When to Use

- Building end-to-end curation pipelines
- User says "curate my data" or "prepare training data"
- Need filtering + classification + deduplication combined
- Processing large datasets for LLM training

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
│  2. HEURISTIC FILTERING (25 filters)                             │
│     ├── WordCountFilter (50-100000 words)                        │
│     ├── NonAlphaNumericFilter (< 25%)                            │
│     ├── SymbolsToWordsFilter (< 10%)                             │
│     ├── RepeatedLinesFilter (< 30%)                              │
│     └── ... (21 more filters)                                    │
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
│     └── VideoReader                                              │
│                                                                   │
│  2. SCENE DETECTION / CLIPPING                                   │
│     ├── TransNetV2ClipExtractionStage (ML-based, GPU)            │
│     └── OR FixedStrideExtractorStage (fixed duration, CPU)       │
│                                                                   │
│  3. FILTERING (optional)                                         │
│     ├── MotionFilterStage (remove static clips)                  │
│     └── ClipAestheticFilterStage (quality filter)                │
│                                                                   │
│  4. CAPTIONING (optional)                                        │
│     ├── CaptionPreparationStage                                  │
│     └── CaptionGenerationStage (Qwen VL, GPU)                    │
│                                                                   │
│  5. EMBEDDING (optional)                                         │
│     └── CosmosEmbed1EmbeddingStage / InternVideo2EmbeddingStage  │
│                                                                   │
│  6. WRITE CLIPS                                                  │
│     └── ClipWriterStage                                          │
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
│  1. READ AUDIO                                                   │
│     └── AudioReader                                              │
│                                                                   │
│  2. TRANSCRIPTION                                                │
│     └── InferenceAsrNemoStage (NeMo ASR, GPU)                    │
│                                                                   │
│  3. QUALITY METRICS                                              │
│     └── WERCalculationStage (Word Error Rate)                    │
│                                                                   │
│  4. FILTERING                                                    │
│     └── Filter by WER threshold                                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Asset Templates

Pre-configured YAML templates are available in `assets/`:

| Template | Description |
|----------|-------------|
| `text_curation_template.yaml` | Standard text curation |
| `text_curation_full.yaml` | Full text curation with all filters |
| `video_curation_template.yaml` | Video with captioning |
| `common_crawl_template.yaml` | Common Crawl processing |

## Customization

### Adding Stages

To add a stage to a generated config:

```yaml
stages:
  # Existing stages...
  
  # Add new stage
  - _target_: nemo_curator.stages.text.classifiers.DomainClassifier
    batch_size: 64
```

### Removing Stages

Comment out or delete unwanted stages from the generated YAML.

### Changing Parameters

Edit the stage configuration directly:

```yaml
# Change filter threshold
- _target_: nemo_curator.stages.text.filters.WordCountFilter
  min_words: 100  # Changed from 50
  max_words: 50000  # Changed from 100000
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
