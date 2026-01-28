---
name: help
description: |
  Context-aware help for CURATOR-OS and NeMo Curator.
  Use when the user asks for help, needs guidance, wants to know
  available commands, or is unsure how to proceed with data curation.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
disable-model-invocation: true
---

# CURATOR-OS Help

Context-aware help and command reference for NeMo Curator.

## Available Skills

| Skill | Description | Invoke |
|-------|-------------|--------|
| `/curate` | Full curation workflow for any modality | Multi-stage pipeline |
| `/dedup-fuzzy` | MinHash + LSH fuzzy deduplication | Near-duplicate removal |
| `/dedup-exact` | Hash-based exact deduplication | Exact duplicate removal |
| `/dedup-semantic` | Embedding-based semantic deduplication | Semantic similarity |
| `/filter` | Apply heuristic text filters | Rule-based filtering |
| `/classify` | Run ML classifiers on text | Quality/domain/safety |
| `/stages` | Look up available processing stages | Stage reference |
| `/help` | This help screen | Get guidance |

## Quick Start

### 1. What modality is your data?

- **Text** (JSONL, Parquet, TXT): Use `/curate` or specific skills
- **Video** (MP4, AVI, MOV): Use `/curate --modality video`
- **Image** (PNG, JPEG, WebP): Use `/curate --modality image`
- **Audio** (WAV, MP3, FLAC): Use `/curate --modality audio`

### 2. What do you want to do?

| Goal | Skill |
|------|-------|
| Clean web-scraped text | `/curate` with filters |
| Remove duplicates | `/dedup-fuzzy` or `/dedup-exact` |
| Score quality | `/classify` |
| Process videos | `/curate --modality video` |
| Find a specific stage | `/stages` |

### 3. Generate configuration

Each skill can generate YAML configurations for NeMo Curator pipelines.

## Common Workflows

### Text Curation for LLM Training

```
1. /curate --modality text --filters standard
2. /classify (quality + educational scoring)
3. /dedup-fuzzy (remove near-duplicates)
```

### Video Dataset Preparation

```
1. /curate --modality video --caption --embed
```

### Quick Quality Pass

```
1. /filter (apply minimal preset)
```

## Execution Hierarchy

Understanding NeMo Curator's execution model:

```
WorkflowBase
├── Orchestrates multiple Pipelines
├── Examples: FuzzyDeduplicationWorkflow
└── NOT a stage!

Pipeline
├── Container for stages
└── Executed by an Executor

CompositeStage
├── Decomposes into ProcessingStages
└── Example: VideoReader

ProcessingStage
├── Single transformation
├── inputs() → outputs()
└── process() method
```

## Task Types by Modality

| Modality | Task Type | Data Format |
|----------|-----------|-------------|
| Text | `DocumentBatch` | DataFrame/Table |
| Video | `VideoTask` | Video object |
| Image | `ImageBatch` | List of ImageObject |
| Audio | `AudioBatch` | Dict or list of dict |

## Resource Configuration

```python
from nemo_curator.stages.resources import Resources

# CPU-only
Resources(cpus=4.0)

# GPU with memory fraction (shares GPU)
Resources(gpu_memory_gb=16.0)

# Full GPU(s)
Resources(gpus=1.0)
```

## Getting More Help

### Stage Reference

```
/stages --modality text --category filters
/stages --search "dedup"
```

### Filter Catalog

```
/filter
```

### Classifier Options

```
/classify
```

## FAQ

### Q: How do I run the generated YAML?

```bash
python -m nemo_curator.config.run \
  --config-path=. \
  --config-name=<your-config>
```

### Q: What's the difference between fuzzy and exact dedup?

- **Exact**: Hash-based, catches identical documents only
- **Fuzzy**: MinHash + LSH, catches ~80% similar documents
- **Semantic**: Embedding-based, catches paraphrases

### Q: How much data reduction should I expect?

| Dataset Type | Typical Reduction |
|--------------|-------------------|
| Raw web crawl | 70-90% |
| Filtered crawl | 20-40% |
| Curated corpus | 5-15% |

### Q: What order should I run stages?

```
Cheap filters → Expensive classifiers → Deduplication
```

This minimizes compute by reducing data volume before expensive operations.

### Q: How do I customize a generated config?

Edit the YAML directly:
- Add stages by copying stage blocks
- Remove stages by deleting or commenting
- Change parameters by editing values

## Setup & Installation

If you need to install or verify NeMo Curator:

```
/setup
```

The setup skill:
- Detects your environment (CUDA, GPU, FFmpeg)
- Recommends appropriate packages
- Runs installation
- Verifies everything works

## Need More Help?

- Check `/stages` for available processing stages
- See individual skill help: `/curate`, `/dedup-fuzzy`, etc.
- NeMo Curator documentation: https://docs.nvidia.com/nemo-curator
