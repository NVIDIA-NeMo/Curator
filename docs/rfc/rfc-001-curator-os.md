# RFC-001: CURATOR-OS - Intelligent Command Interface for NeMo Curator

**Status:** Draft (Validated)  
**Authors:** NeMo Curator Team  
**Created:** 2026-01-26  
**Last Updated:** 2026-01-26  
**Validation:** Stage inventory and patterns verified against `nemo_curator/stages/` codebase

---

## Table of Contents

1. [Summary](#summary)
2. [Motivation](#motivation)
3. [Design Principles](#design-principles)
4. [Architecture Overview](#architecture-overview)
5. [Rule Hierarchy](#rule-hierarchy)
6. [Command System](#command-system)
7. [Workflow Templates](#workflow-templates)
8. [Intelligent Routing](#intelligent-routing)
9. [Module System](#module-system)
10. [Implementation Plan](#implementation-plan)
11. [Open Questions](#open-questions)
12. [References](#references)

---

## Summary

This RFC proposes **CURATOR-OS**, an intelligent command interface framework for NeMo Curator that enables users to interact with data curation pipelines through natural language and slash commands. Built on the [Agent Skills](https://agentskills.io/specification) open standard, CURATOR-OS provides:

- **Task-oriented skills** that translate user goals into pipeline configurations
- **Modality-aware routing** that automatically selects appropriate stages
- **Executable scripts** for config generation, validation, and resource estimation
- **Progressive disclosure** with tiered context loading (metadata → instructions → resources)
- **Cross-agent compatibility** with Cursor, Claude, and Codex agents
- **Built-in validation** for pipeline configuration and resource requirements

The framework operates as an Agent Skills system (`.cursor/skills/`) with:
- **SKILL.md files** defining skill behavior and instructions
- **scripts/** containing executable Python utilities
- **references/** providing on-demand documentation
- **assets/** storing templates and static resources

---

## Motivation

### Current User Experience Challenges

1. **High Learning Curve**: Users must understand 70+ processing stages across 4 modalities, their parameters, input/output requirements, and resource needs.

2. **Manual Pipeline Assembly**: Building pipelines requires:
   - Knowing which stages exist
   - Understanding stage ordering dependencies
   - Configuring resource allocation (GPU memory, CPUs)
   - Handling input/output compatibility between stages

3. **Configuration Complexity**: YAML configurations can span 200+ lines with dozens of parameters (see `tutorials/video/getting-started/video_split_clip_example.py` with 100+ CLI arguments).

4. **Lack of Guided Workflows**: No system guides users through common curation patterns like "download → filter → deduplicate → export."

### Evidence from Codebase

**Stage Count by Modality** (from `nemo_curator/stages/`):

| Modality | Stages | Key Categories |
|----------|--------|----------------|
| Text | 45+ | Classifiers, Filters, Modifiers, Deduplication, Download, IO |
| Video | 20+ | Clipping, Filtering, Embedding, Captioning, IO |
| Image | 10+ | Embedding, Filtering, Deduplication, IO |
| Audio | 6+ | ASR, Metrics, Dataset Prep, IO |

**Example Complexity** - Video curation requires understanding:

```python
# From tutorials/video/getting-started/video_split_clip_example.py
# Users must configure 50+ arguments including:
# --splitting-algorithm, --transnetv2-threshold, --transnetv2-min-length-s
# --transcode-encoder, --motion-filter, --aesthetic-threshold
# --embedding-algorithm, --captioning-algorithm, --captioning-batch-size
```

### Goals

1. **Reduce time-to-first-pipeline** from hours to minutes
2. **Enable natural language pipeline specification**
3. **Provide guardrails** for resource allocation and stage compatibility
4. **Accelerate common workflows** with pre-built templates
5. **Maintain full power** for advanced users who need fine-grained control

---

## Architecture Concepts

CURATOR-OS must understand these core NeMo Curator concepts to provide accurate guidance:

### Execution Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        WorkflowBase                              │
│  - Has run() method that creates/executes multiple Pipelines    │
│  - Examples: FuzzyDeduplicationWorkflow, SemanticDeduplication   │
│  - NOT a stage - orchestrates entire workflows programmatically  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Pipeline                                 │
│  - Container for stages, executed by an Executor                │
│  - Stages are run sequentially, tasks flow between them         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CompositeStage                              │
│  - Decomposes into multiple ProcessingStages at build time      │
│  - Examples: VideoReader, CommonCrawlDownloadExtractStage       │
│  - User-facing simplification over multiple execution stages    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ProcessingStage                              │
│  - Single transformation: Task → Task | list[Task] | None       │
│  - Has inputs(), outputs(), process() methods                   │
│  - Declares resource requirements via Resources dataclass       │
└─────────────────────────────────────────────────────────────────┘
```

### Task Types by Modality

| Modality | Task Type | Data Type |
|----------|-----------|-----------|
| Text | `DocumentBatch` | `pd.DataFrame` or `pa.Table` |
| Video | `VideoTask` | `Video` object with clips, metadata |
| Image | `ImageBatch` | `list[ImageObject]` |
| Audio | `AudioBatch` | `dict` or `list[dict]` |
| Files | `FileGroupTask` | `list[str]` (file paths) |

### Resource Configuration

```python
from nemo_curator.stages.resources import Resources

# CPU-only stage
Resources(cpus=4.0)

# GPU stage with memory fraction (shares GPU)
Resources(gpu_memory_gb=16.0)

# GPU stage with full GPU(s)
Resources(gpus=1.0)  # or gpus=2.0 for multi-GPU

# Mixed CPU + GPU
Resources(cpus=4.0, gpu_memory_gb=16.0)

# Important: Cannot specify both gpus and gpu_memory_gb
```

---

## Design Principles

### 1. Task-Oriented, Not Stage-Oriented

Users should express **what they want to achieve**, not which stages to use.

```
❌ "Add MinHashStage, then LSHStage, then ConnectedComponentsStage"
✅ "Deduplicate this text dataset using fuzzy matching"
```

### 2. Modality-Aware

The system should automatically detect data modality and route to appropriate stages.

```
User: "Curate the data in /path/to/videos"
System: Detects .mp4 files → Routes to video curation workflow
```

### 3. Progressive Disclosure

- **Tier 0 (Fast)**: Direct execution for experts who know what they want
- **Tier 1 (Light)**: Brief confirmation, then proceed
- **Tier 2 (Full)**: Complete analysis, recommendations, confirmation required

### 4. Composable and Chainable

Workflows can be:
- Built from pre-defined templates
- Customized by adding/removing steps
- Combined into larger pipelines

### 5. Validated by Default

Every pipeline configuration should be validated for:
- Stage input/output compatibility
- Resource requirements (GPU memory, CPU count)
- Parameter validity

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CURATOR-OS (Agent Skills)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌──────────────────────────────────────────┐   │
│  │   Orchestrator  │    │           Command Skills                  │   │
│  │   curator-os/   │───▶│  /curate  /dedup-*  /filter  /classify   │   │
│  │   SKILL.md      │    │  /download-*  /caption  /embed           │   │
│  │   scripts/      │    └──────────────────────────────────────────┘   │
│  │   references/   │                      │                            │
│  └─────────────────┘                      ▼                            │
│          │               ┌──────────────────────────────────────────┐  │
│          │               │              Scripts                      │  │
│          │               │  generate_yaml.py   estimate_resources.py │  │
│          │               │  detect_modality.py validate_config.py    │  │
│          │               │  search_stages.py   list_filters.py       │  │
│          │               └──────────────────────────────────────────┘  │
│          │                                │                            │
│          ▼                                ▼                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                       References (on-demand)                     │  │
│  │  STAGE_REFERENCE.md │ RESOURCE_GUIDE.md │ FILTER_CATALOG.md     │  │
│  │  MODALITY_PATTERNS.md │ CLASSIFIER_CATALOG.md │ *_PARAMS.md     │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│          │                                │                            │
│          ▼                                ▼                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                       Assets (templates)                         │  │
│  │  text_curation_template.yaml │ video_curation_template.yaml     │  │
│  │  processing_stage_template.py │ classifier_template.py          │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
          ┌─────────────────────┐   ┌─────────────────────┐
          │   Generated YAML    │   │     NeMo Curator    │
          │   Configurations    │──▶│  Pipeline │ Stages  │
          └─────────────────────┘   └─────────────────────┘
```

### Progressive Context Loading

```
User types: /dedup-fuzzy

1. METADATA (~100 tokens) - Always loaded at startup
   name: dedup-fuzzy
   description: "Perform fuzzy deduplication..."

2. INSTRUCTIONS (~2000 tokens) - Loaded when skill activated
   SKILL.md body with step-by-step guidance

3. SCRIPTS (executed on demand)
   scripts/generate_fuzzy_config.py --input /data...

4. REFERENCES (~500 tokens each) - Loaded when needed
   references/FUZZY_DEDUP_PARAMS.md for parameter tuning
```

---

## Skills Architecture

CURATOR-OS uses the [Agent Skills](https://agentskills.io/specification) open standard for cross-agent 
compatibility. This enables portable, version-controlled skills with executable scripts.

### Directory Structure

```
.cursor/skills/
├── curator-os/                          # Central orchestration skill
│   ├── SKILL.md                         # Main skill instructions
│   ├── scripts/
│   │   ├── detect_modality.py           # Data modality detection
│   │   └── validate_pipeline.py         # Pipeline validation
│   └── references/
│       ├── STAGE_REFERENCE.md           # Complete stage catalog
│       ├── RESOURCE_GUIDE.md            # Resource optimization
│       └── MODALITY_PATTERNS.md         # Modality-specific patterns
│
├── curate/                              # Full curation command
│   ├── SKILL.md
│   ├── scripts/
│   │   └── generate_yaml.py             # YAML config generator
│   └── assets/
│       ├── text_curation_template.yaml
│       ├── video_curation_template.yaml
│       └── common_crawl_template.yaml
│
├── dedup-fuzzy/                         # Fuzzy deduplication
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── estimate_resources.py        # Memory/GPU estimator
│   │   └── generate_fuzzy_config.py     # Config generator
│   └── references/
│       └── FUZZY_DEDUP_PARAMS.md        # Parameter tuning guide
│
├── dedup-semantic/                      # Semantic deduplication
│   ├── SKILL.md
│   └── scripts/
│       └── generate_semantic_config.py
│
├── dedup-exact/                         # Exact deduplication
│   └── SKILL.md
│
├── filter/                              # Heuristic filtering
│   ├── SKILL.md
│   ├── scripts/
│   │   └── list_filters.py              # List available filters
│   └── references/
│       └── FILTER_CATALOG.md            # All 33+ filters documented
│
├── classify/                            # Classification commands
│   ├── SKILL.md
│   └── references/
│       └── CLASSIFIER_CATALOG.md        # All classifiers documented
│
├── download-cc/                         # Common Crawl download
│   ├── SKILL.md
│   └── scripts/
│       └── list_snapshots.py            # List available CC snapshots
│
├── download-wiki/                       # Wikipedia download
│   └── SKILL.md
│
├── caption/                             # Video captioning
│   ├── SKILL.md
│   └── references/
│       └── CAPTIONING_MODELS.md         # Model comparison
│
├── embed/                               # Embedding generation
│   └── SKILL.md
│
├── stages/                              # Stage reference lookup
│   ├── SKILL.md
│   └── scripts/
│       └── search_stages.py             # Search stage by modality/category
│
├── validate-pipeline/                   # Pipeline validation
│   ├── SKILL.md
│   └── scripts/
│       └── validate_config.py           # Validate YAML config
│
├── create-stage/                        # Custom stage creation
│   ├── SKILL.md
│   └── assets/
│       ├── processing_stage_template.py
│       ├── classifier_template.py
│       └── filter_template.py
│
└── help/                                # Context-aware help
    └── SKILL.md
```

### Skill Types and Invocation

| Skill Type | Invocation | `disable-model-invocation` |
|------------|------------|---------------------------|
| **Orchestrator** (`curator-os`) | Auto-invoked based on context | `false` (agent decides) |
| **Commands** (`curate`, `dedup-*`, etc.) | `/curate`, `/dedup-fuzzy` | `true` (explicit only) |
| **Utilities** (`stages`, `help`) | `/stages`, `/help` | `true` (explicit only) |
| **Templates** (`create-stage`) | `/create-stage` | `true` (explicit only) |

### SKILL.md Frontmatter Specifications

#### Orchestrator Skill (curator-os)

```yaml
---
name: curator-os
description: |
  Intelligent command interface for NeMo Curator data curation pipelines.
  Use when the user mentions data curation, deduplication, filtering,
  classification, video/text/image/audio processing, or pipeline building.
  Routes user intent to appropriate curation skills.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  nemo-curator-version: ">=0.5.0"
compatibility: Requires Python 3.10+, Ray cluster for distributed execution
---
```

#### Command Skills (e.g., dedup-fuzzy)

```yaml
---
name: dedup-fuzzy
description: |
  Perform fuzzy deduplication on text datasets using MinHash + LSH.
  Use when the user wants to remove near-duplicate documents, mentions
  fuzzy matching, MinHash, LSH, or similarity-based deduplication.
  Supports datasets from 1GB to multi-TB scale.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: text
  gpu-required: true
disable-model-invocation: true
---
```

#### Utility Skills (e.g., stages)

```yaml
---
name: stages
description: |
  Look up NeMo Curator processing stages by modality and category.
  Use when the user asks about available stages, what stages do,
  or needs to find a specific processing capability.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
disable-model-invocation: true
---
```

### Scripts Specification

Scripts enable executable functionality within skills. All scripts should:

- Be self-contained Python or Bash
- Print JSON output for agent consumption
- Include `--help` documentation
- Handle errors gracefully with clear messages

#### Example: `detect_modality.py`

```python
#!/usr/bin/env python3
"""Detect data modality from file extensions in a directory."""
import argparse
import json
from pathlib import Path

MODALITY_EXTENSIONS = {
    "text": {".jsonl", ".parquet", ".json", ".txt"},
    "video": {".mp4", ".avi", ".mov", ".mkv", ".webm"},
    "image": {".jpg", ".jpeg", ".png", ".webp", ".gif"},
    "audio": {".wav", ".mp3", ".flac", ".ogg"},
}

def detect_modality(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"error": f"Path does not exist: {path}"}
    
    extensions = set()
    for f in p.rglob("*"):
        if f.is_file():
            extensions.add(f.suffix.lower())
    
    for modality, exts in MODALITY_EXTENSIONS.items():
        if extensions & exts:
            return {"modality": modality, "extensions": list(extensions & exts)}
    
    return {"modality": "unknown", "extensions": list(extensions)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Directory to analyze")
    args = parser.parse_args()
    print(json.dumps(detect_modality(args.path), indent=2))
```

#### Example: `generate_yaml.py`

```python
#!/usr/bin/env python3
"""Generate NeMo Curator pipeline YAML configuration."""
import argparse
import json
from pathlib import Path

def generate_fuzzy_dedup_config(
    input_path: str,
    output_path: str,
    cache_path: str,
    char_ngrams: int = 24,
    num_bands: int = 20,
    minhashes_per_band: int = 13,
    bands_per_iteration: int = 5,
) -> str:
    return f'''# Generated by CURATOR-OS
defaults:
  - _self_
  - override hydra/job_logging: none

input_path: {input_path}
output_path: {output_path}
cache_path: {cache_path}
text_field: "text"

workflow:
  - _target_: nemo_curator.stages.deduplication.fuzzy.workflow.FuzzyDeduplicationWorkflow
    input_path: ${{input_path}}
    output_path: ${{output_path}}
    cache_path: ${{cache_path}}
    char_ngrams: {char_ngrams}
    num_bands: {num_bands}
    minhashes_per_band: {minhashes_per_band}
    bands_per_iteration: {bands_per_iteration}
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--char-ngrams", type=int, default=24)
    parser.add_argument("--num-bands", type=int, default=20)
    parser.add_argument("--output-file", help="Write to file instead of stdout")
    args = parser.parse_args()
    
    config = generate_fuzzy_dedup_config(
        args.input_path, args.output_path, args.cache_path,
        char_ngrams=args.char_ngrams, num_bands=args.num_bands,
    )
    
    if args.output_file:
        Path(args.output_file).write_text(config)
        print(json.dumps({"status": "written", "path": args.output_file}))
    else:
        print(config)
```

### References Specification

Reference files provide detailed documentation that agents load on demand. Keep under 500 lines each.

| Reference File | Purpose | Loaded When |
|----------------|---------|-------------|
| `STAGE_REFERENCE.md` | Complete stage catalog with signatures | User asks about stages |
| `RESOURCE_GUIDE.md` | GPU/CPU allocation guidance | Configuring resources |
| `FILTER_CATALOG.md` | All 33+ filters with parameters | Using filtering |
| `CLASSIFIER_CATALOG.md` | All classifiers with models | Using classification |
| `FUZZY_DEDUP_PARAMS.md` | MinHash/LSH parameter tuning | Fuzzy deduplication |
| `MODALITY_PATTERNS.md` | Modality-specific pipelines | Building pipelines |

### Assets Specification

Assets provide static templates and data files.

| Asset Type | Purpose | Example |
|------------|---------|---------|
| YAML templates | Pre-configured pipelines | `text_curation_template.yaml` |
| Python templates | Custom stage scaffolding | `processing_stage_template.py` |
| JSON schemas | Configuration validation | `pipeline_schema.json` |

### Progressive Disclosure

Skills are structured for efficient context usage:

```
1. Metadata (~100 tokens)
   └── name + description loaded at startup for ALL skills

2. Instructions (< 5000 tokens)
   └── SKILL.md body loaded when skill is activated

3. Resources (as needed)
   ├── scripts/     → Executed when actions are needed
   ├── references/  → Loaded when detailed info requested
   └── assets/      → Used when generating configs/code
```

---

## Command System

### Skill Invocation

Skills can be invoked two ways:

1. **Explicit invocation**: Type `/skill-name` in chat (for skills with `disable-model-invocation: true`)
2. **Automatic invocation**: Agent decides based on context (for orchestrator skills)

### Available Skills

| Skill | Invocation | Description |
|-------|------------|-------------|
| `/curate` | Explicit | Full curation workflow for any modality |
| `/curate-video` | Explicit | Video-specific curation pipeline |
| `/curate-image` | Explicit | Image-specific curation pipeline |
| `/curate-audio` | Explicit | Audio-specific curation pipeline |
| `/dedup-fuzzy` | Explicit | MinHash + LSH fuzzy deduplication |
| `/dedup-exact` | Explicit | Hash-based exact deduplication |
| `/dedup-semantic` | Explicit | Embedding-based semantic deduplication |
| `/filter` | Explicit | Apply heuristic filters to text |
| `/classify` | Explicit | Run ML classifiers on data |
| `/download-cc` | Explicit | Download Common Crawl snapshots |
| `/download-wiki` | Explicit | Download Wikipedia dumps |
| `/embed` | Explicit | Generate embeddings with vLLM |
| `/caption` | Explicit | Generate video captions with Qwen VL |
| `/stages` | Explicit | Look up available processing stages |
| `/validate-pipeline` | Explicit | Validate pipeline configuration |
| `/create-stage` | Explicit | Generate custom stage template |
| `/help` | Explicit | Context-aware help |
| `curator-os` | Auto | Intelligent routing (agent decides) |

### Skill Parameters

Parameters are passed as natural language or structured arguments:

```
/dedup-fuzzy
Input path: /data/text
Output path: /data/deduped
Bands: 20
Hashes per band: 13
```

Or conversationally:

```
/dedup-fuzzy the parquet files in /data/text with 20 bands and output to /data/deduped
```

### Script Execution Examples

Skills with scripts can execute them for the user:

```
User: /stages text filters

Agent: Searching stages...
[Runs: scripts/search_stages.py --modality text --category filters]

Found 33 text filters:
- NonAlphaNumericFilter: Discard if >25% non-alphanumeric
- WordCountFilter: Filter by word count range (50-100000)
- SymbolsToWordsFilter: Discard if symbol ratio >0.1
...
```

```
User: /dedup-fuzzy generate config for /data/crawl

Agent: Generating fuzzy deduplication config...
[Runs: scripts/generate_fuzzy_config.py --input-path /data/crawl ...]

Generated: fuzzy_dedup_pipeline.yaml
```

### Natural Language Support

The orchestrator also understands natural language:

```
User: "I want to download Common Crawl, extract text, filter by quality, 
       and remove duplicates"

CURATOR-OS:
📍 Detected Intent: Full text curation pipeline

Steps:
1. CommonCrawlDownload → Extract text from WARC files
2. HeuristicFilters → Apply 25 quality heuristics
3. QualityClassifier → Score educational content
4. FuzzyDeduplication → Remove near-duplicates

Generate this pipeline? [y/n/configure]
```

---

## Workflow Templates

### Template: Text Curation End-to-End

```yaml
name: "Text Curation Pipeline"
trigger: ["::curate", "::curate-text"]
modality: text
estimated_time: "varies by data size"

steps:
  - name: "Analyze Input"
    purpose: "Detect data format, estimate size, recommend configuration"
    rule: system/curator-context-analyzer
    output: context_report
    
  - name: "Read Data"
    purpose: "Load text data from source"
    stage: JsonlReader | ParquetReader
    configurable:
      - file_paths
      - files_per_partition
      - blocksize
    
  - name: "Heuristic Filtering"
    purpose: "Apply rule-based quality filters"
    stage: ScoreFilter (multiple)
    condition: if_filter_requested
    configurable:
      - filter_list (default: english_heuristics)
      - text_field
    
  - name: "Quality Classification"
    purpose: "Score content quality with ML model"
    stage: DistributedDataClassifier
    condition: if_quality_filter_requested
    configurable:
      - model_identifier
      - filter_by
      - batch_size
    
  - name: "Deduplication"
    purpose: "Remove duplicate or near-duplicate content"
    workflow: fuzzy-deduplication | exact-deduplication | semantic-deduplication
    condition: if_dedup_requested
    configurable:
      - method (exact | fuzzy | semantic)
      - threshold
    
  - name: "Write Output"
    purpose: "Save curated dataset"
    stage: JsonlWriter | ParquetWriter
    configurable:
      - output_path
      - fields

validation:
  - Check input path exists
  - Validate stage compatibility
  - Estimate resource requirements
  - Warn if GPU required but not available
```

### Template: Fuzzy Deduplication

> **Note**: This workflow uses `FuzzyDeduplicationWorkflow` (a `WorkflowBase` class),
> not a `CompositeStage`. It orchestrates multiple pipelines programmatically.
> 
> Source: `nemo_curator/stages/deduplication/fuzzy/workflow.py`

```yaml
name: "Fuzzy Deduplication Workflow"
trigger: ["::dedup-fuzzy"]
modality: text
complexity: high
gpu_required: true
workflow_class: FuzzyDeduplicationWorkflow

# Parameters match FuzzyDeduplicationWorkflow.__init__()
parameters:
  # I/O Config
  input_path: required
  output_path: required
  cache_path: required  # For intermediate MinHash/LSH results
  input_filetype: default="parquet", choices=["jsonl", "parquet"]
  input_blocksize: default="1GiB"
  text_field: default="text"
  perform_removal: default=false  # Note: Not yet implemented
  
  # MinHash + LSH Config
  seed: default=42
  char_ngrams: default=24, range=[20, 50], help="Shingle size (< 20 may cause false positives)"
  num_bands: default=20, range=[5, 50], help="LSH bands"
  minhashes_per_band: default=13, range=[5, 25], help="Hashes per band"
  use_64_bit_hash: default=false
  bands_per_iteration: default=5, range=[1, num_bands], help="Memory control for LSH shuffle"
  
  # Storage Options (for remote storage like S3/GCS)
  read_kwargs: optional
  cache_kwargs: optional
  write_kwargs: optional

# Internal pipelines created by the workflow:
internal_pipelines:
  - name: "minhash_pipeline"
    stages:
      - FilePartitioningStage  # Groups input files
      - MinHashStage           # Computes minhash signatures
    output: cache_path/MinHashStage/
    
  - name: "lsh_duplicate_identification_pipeline"
    stages:
      - FilePartitioningStage  # Reads minhash results
      - LSHStage               # Locality Sensitive Hashing
    config:
      rmm_pool_size: "auto"
      spill_memory_limit: "auto"
    
  - name: "connected_components_pipeline"
    stages:
      - BucketsToEdgesStage        # Converts buckets to graph edges
      - ConnectedComponentsStage   # Finds duplicate clusters
      - IdentifyDuplicatesStage    # Generates removal IDs
    output: output_path/duplicate_ids.parquet

# Executor requirement
executor: RayActorPoolExecutor  # Required - FuzzyDeduplicationWorkflow validates this

resource_guidance:
  small_dataset: "< 100GB: Single GPU, 64GB RAM, bands_per_iteration=5"
  medium_dataset: "100GB-1TB: 4 GPUs, 256GB RAM, bands_per_iteration=3"
  large_dataset: "> 1TB: 8+ GPUs, bands_per_iteration=1-2"
  
warnings:
  - "char_ngrams < 20 may cause ~5% false positive rate"
  - "perform_removal=true is not yet implemented"
  - "Executor must be RayActorPoolExecutor (validated at runtime)"
```

### Template: Video Curation

```yaml
name: "Video Curation Pipeline"
trigger: ["::curate-video"]
modality: video
gpu_required: true

steps:
  - name: "Read Videos"
    stage: VideoReader
    configurable:
      - input_video_path
      - video_limit
      
  - name: "Scene Detection + Clipping"
    stage: TransNetV2ClipExtractionStage | FixedStrideExtractorStage
    condition: always
    configurable:
      - splitting_algorithm (transnetv2 | fixed_stride)
      - clip_length, threshold, min_length
      
  - name: "Transcode Clips"
    stage: ClipTranscodingStage
    resources: { cpus: 6.0 }
    configurable:
      - encoder (libopenh264 | h264_nvenc | libx264)
      
  - name: "Motion Filtering"
    stages: [MotionVectorDecodeStage, MotionFilterStage]
    condition: if_motion_filter_enabled
    resources: { cpus: 4.0, gpu_memory_gb: 8.0 }
    
  - name: "Aesthetic Filtering"
    stages: [ClipFrameExtractionStage, ClipAestheticFilterStage]
    condition: if_aesthetic_filter_enabled
    resources: { gpu_memory_gb: 8.0 }
    
  - name: "Generate Embeddings"
    stages: [CosmosEmbed1FrameCreationStage, CosmosEmbed1EmbeddingStage]
    condition: if_embeddings_requested
    resources: { gpu_memory_gb: 20.0 }
    configurable:
      - embedding_algorithm (cosmos-embed1-224p | internvideo2)
      
  - name: "Generate Captions"
    stages: [CaptionPreparationStage, CaptionGenerationStage]
    condition: if_captions_requested
    resources: { gpu_memory_gb: 40.0 }
    configurable:
      - captioning_algorithm (qwen)
      - prompt_variant
      
  - name: "Write Output"
    stage: ClipWriterStage
    configurable:
      - output_path
      - upload_clips
```

---

## Intelligent Routing

### Orchestrator Logic

```yaml
curator_orchestrator:
  
  # Step 1: Modality Detection
  modality_detection:
    rules:
      - if: file_extension in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        then: VIDEO
      - if: file_extension in [".jpg", ".jpeg", ".png", ".webp", ".gif"]
        then: IMAGE
      - if: file_extension in [".wav", ".mp3", ".flac", ".ogg"]
        then: AUDIO
      - if: file_extension in [".jsonl", ".parquet", ".json", ".txt"]
        then: TEXT
      - default: TEXT
  
  # Step 2: Intent Classification
  intent_classification:
    DOWNLOAD:
      triggers: [download, fetch, get, acquire, crawl]
      examples:
        - "download common crawl"
        - "get wikipedia data"
        - "fetch arxiv papers"
      routes_to: commands/download
      
    FILTER:
      triggers: [filter, clean, remove, heuristic, quality]
      examples:
        - "filter low quality text"
        - "apply heuristic filters"
        - "clean the dataset"
      routes_to: commands/filter
      
    DEDUPLICATE:
      triggers: [dedup, deduplicate, duplicates, unique, fuzzy, exact]
      examples:
        - "deduplicate the dataset"
        - "remove duplicates"
        - "fuzzy deduplication"
      routes_to: commands/deduplicate
      
    CLASSIFY:
      triggers: [classify, score, label, categorize, quality, domain]
      examples:
        - "classify by quality"
        - "score educational content"
        - "label domains"
      routes_to: commands/classify
      
    EMBED:
      triggers: [embed, embedding, vector, encode, representation]
      examples:
        - "generate embeddings"
        - "create vectors"
      routes_to: commands/embed
      
    CAPTION:
      triggers: [caption, describe, annotate, transcribe]
      examples:
        - "generate captions for videos"
        - "transcribe audio"
      routes_to: commands/caption
      
    CURATE:
      triggers: [curate, process, pipeline, end-to-end, full]
      examples:
        - "curate this dataset"
        - "full curation pipeline"
        - "process the data end-to-end"
      routes_to: workflows/<modality>-curation
  
  # Step 3: Confidence Scoring
  confidence_scoring:
    HIGH:   # 90-100% - Proceed immediately
      criteria:
        - Explicit command shortcut used
        - Clear action verb
        - Single unambiguous target
    MEDIUM: # 70-89% - Proceed with confirmation
      criteria:
        - Natural language with recognizable patterns
        - Some ambiguity in scope OR method
    LOW:    # <70% - Require clarification
      criteria:
        - Vague request
        - Multiple possible interpretations
        - Missing required information

  # Step 4: Tier Assignment
  tier_assignment:
    TIER_0_FAST:
      triggers:
        - Explicit shortcut with target: "::dedup-fuzzy @/data"
        - Flag: "--fast" or "-0"
      behavior: Execute immediately, minimal output
      
    TIER_1_LIGHT:
      triggers:
        - Flag: "--light" or "-1"
      behavior: Brief summary, auto-proceed
      
    TIER_2_FULL:
      triggers:
        - Default for natural language
        - Complex multi-step workflows
        - Flag: "--full" or "-2"
      behavior: Full analysis, recommendations, confirmation required
```

### Example Routing Flows

**Example 1: Explicit Command (Tier 0)**

```
User: ::dedup-fuzzy @/data/text --output /data/deduped

[TIER 0 - Fast Path]
📍 Fuzzy Deduplication

Input: /data/text (detected: 50GB Parquet)
Output: /data/deduped

Using defaults:
- char_ngrams: 24
- num_bands: 20
- minhashes_per_band: 13

Executing...
```

**Example 2: Natural Language (Tier 2)**

```
User: I want to clean up my text dataset and remove low quality content

[TIER 2 - Full Analysis]
🎯 Task Analysis

Intent: FILTER + CLASSIFY
Modality: TEXT (detected from context)
Scope: Quality improvement

Recommended Pipeline:
1. HeuristicFilters (25 filters for text quality)
2. QualityClassifier (FineWeb-Edu model)
3. Filter (keep High + Medium quality)

Estimated:
- Time: ~2 hours for 100GB
- Resources: 1 GPU (16GB), 4 CPUs

Options:
[a] Execute recommended pipeline
[b] Configure filters
[c] Add deduplication
[d] Generate YAML only
```

---

## Module System

### Always-Apply Modules

These modules are automatically loaded for all CURATOR-OS interactions:

```yaml
# .cursor/rules/system/curator-os/RULE.mdc
alwaysApply: true

# Loaded modules:
# - modules/pipeline-patterns (stage composition rules)
# - modules/resource-optimization (GPU/CPU guidance)
# - modules/evidence-handling (code references)
```

### Contextually-Loaded Modules

Loaded based on detected modality or intent:

```yaml
contextual_modules:
  TEXT:
    load:
      - modules/modality-text
    content: "Text-specific stages, filters, classifiers"
    
  VIDEO:
    load:
      - modules/modality-video
    content: "Video stages, codecs, frame extraction"
    
  IMAGE:
    load:
      - modules/modality-image
    content: "Image stages, CLIP, deduplication"
    
  AUDIO:
    load:
      - modules/modality-audio
    content: "Audio stages, ASR, WER metrics"
    
  DEDUPLICATION:
    load:
      - modules/deduplication-patterns
    content: "Dedup algorithms, parameter tuning"
```

### Module: Resource Optimization

> **Note**: Resource configuration patterns validated against actual stage implementations.

```yaml
# modules/resource-optimization/RULE.mdc

resource_patterns:
  # NeMo Curator uses two resource specification patterns:
  
  pattern_1_gpu_memory_fraction:
    description: "For stages that share a GPU with others"
    usage: "Resources(gpu_memory_gb=X)"
    example: "CosmosEmbed1EmbeddingStage uses gpu_memory_gb=20"
    when_to_use: "Single-GPU stages that can share GPU with others"
    
  pattern_2_full_gpus:
    description: "For stages that need exclusive GPU access"
    usage: "Resources(gpus=X)"
    example: "CaptionGenerationStage uses gpus=1"
    when_to_use: "Multi-GPU stages or stages needing full GPU"

resource_guidance:
  
  gpu_stages:
    TransNetV2ClipExtractionStage:
      config: "Resources(gpu_memory_gb=10)"
      notes: "Scene detection model, configurable via gpu_memory_gb param"
      source: "stages/video/clipping/transnetv2_extraction.py:88"
      
    CosmosEmbed1EmbeddingStage:
      config: "Resources(gpu_memory_gb=20)"  # default
      variants:
        224p: "gpu_memory_gb=20"
        336p: "gpu_memory_gb=30 (estimated)"
        448p: "gpu_memory_gb=40 (estimated)"
      source: "stages/video/embedding/cosmos_embed1.py:135"
      
    CaptionGenerationStage:
      config: "Resources(gpus=1)"  # Full GPU allocation
      notes: "Qwen VL model requires full GPU, FP8 reduces memory"
      source: "stages/video/caption/caption_generation.py:75"
      
    ImageEmbeddingStage:
      config: "Resources(gpus=0.25)"  # Fraction of GPU
      notes: "CLIP embedding, can share GPU"
      source: "stages/image/embedders/clip_embedder.py:45"
      
    MotionFilterStage:
      config: "Resources(gpus=num_gpus_per_worker)"  # Configurable
      notes: "Optional GPU acceleration"
      source: "stages/video/filtering/motion_filter.py:119"
      
    MinHashStage:
      config: "Resources(cpus=2.0)"
      notes: "CPU-only stage"
      source: "stages/deduplication/fuzzy/minhash.py"
      
    LSHStage:
      config: "Resources(cpus=4.0, gpu_memory_gb=16.0)"
      notes: "Shuffle-heavy, uses RMM pool for GPU memory management"
      source: "stages/deduplication/fuzzy/lsh/stage.py"
      
    ConnectedComponentsStage:
      config: "GPU-accelerated graph algorithm"
      notes: "Memory scales with number of edges"
  
  cpu_stages:
    HeuristicFilters:
      config: "Resources(cpus=1.0)"  # Default
      notes: "Most filters are lightweight"
      
    MinHashStage:
      config: "Resources(cpus=2.0)"
      source: "stages/deduplication/fuzzy/minhash.py"
      
    ClipTranscodingStage:
      config: "Resources(cpus=6.0)"  # Default num_cpus_per_worker
      notes: "FFmpeg encoding, more CPUs = faster"
      source: "stages/video/clipping/clip_extraction_stages.py:54"
      
    MotionVectorDecodeStage:
      config: "Resources(cpus=6.0)"  # Default num_cpus_per_worker
      source: "stages/video/filtering/motion_filter.py:41"
      
    JsonlReader:
      config: "Resources(cpus=1.0)"
      notes: "IO-bound"
    
  memory_scaling:
    fuzzy_dedup:
      rule: "Reduce bands_per_iteration if OOM"
      guidance:
        - "< 100GB data: bands_per_iteration=5"
        - "100GB-500GB: bands_per_iteration=3"
        - "> 500GB: bands_per_iteration=1"
      source: "FuzzyDeduplicationWorkflow docstring recommends this"
      
    lsh_stage:
      rule: "Uses RMM memory pool with auto-sizing"
      config_options:
        - "rmm_pool_size='auto'"
        - "spill_memory_limit='auto'"
```

### Module: Stage Reference

> **Important Architecture Note**: NeMo Curator has three composition patterns:
> 
> | Pattern | Base Class | Description |
> |---------|------------|-------------|
> | **ProcessingStage** | `ProcessingStage[In, Out]` | Single transformation step |
> | **CompositeStage** | `CompositeStage[In, Out]` | Decomposes into multiple ProcessingStages at pipeline build time |
> | **WorkflowBase** | `WorkflowBase` | Orchestrates multiple Pipelines with custom execution logic |
> 
> **Key distinction**: `CompositeStage` decomposes during `pipeline.build()`, while `WorkflowBase` 
> has its own `run()` method that creates and executes multiple pipelines programmatically.

```yaml
# modules/stage-reference/RULE.mdc

# Invoked with: ::stages --modality <mod> --category <cat>

stage_catalog:
  text:
    classifiers:
      - name: QualityClassifier
        model: "nvidia/quality-classifier-deberta"
        purpose: "Classify text quality (High/Medium/Low)"
        inputs: ["text"]
        outputs: ["quality_label", "quality_score"]
        source: "stages/text/classifiers/quality.py"
        
      - name: DomainClassifier
        model: "nvidia/domain-classifier"
        purpose: "Classify text domain (News/Wiki/Social/...)"
        inputs: ["text"]
        outputs: ["domain_label", "domain_score"]
        
      - name: MultilingualDomainClassifier
        purpose: "Domain classification for non-English text"
        source: "stages/text/classifiers/domain.py"
        
      - name: FineWebEduClassifier
        model: "HuggingFaceFW/fineweb-edu-classifier"
        purpose: "Score educational content quality"
        inputs: ["text"]
        outputs: ["edu_score"]
        
      - name: FineWebMixtralEduClassifier
        purpose: "Mixtral-based educational classifier"
        source: "stages/text/classifiers/fineweb_edu.py"
        
      - name: FineWebNemotronEduClassifier
        purpose: "Nemotron-based educational classifier"
        source: "stages/text/classifiers/fineweb_edu.py"
        
      - name: AegisClassifier
        model: "nvidia/Aegis-AI-Content-Safety-LlamaGuard-*"
        purpose: "Content safety classification"
        inputs: ["text"]
        outputs: ["safety_label"]
        
      - name: InstructionDataGuardClassifier
        purpose: "Guard for instruction-following data"
        source: "stages/text/classifiers/aegis.py"
        
    filters:
      heuristic:  # 25 filters in stages/text/filters/heuristic_filter.py
        - NonAlphaNumericFilter
        - SymbolsToWordsFilter
        - NumbersFilter
        - UrlsFilter
        - BulletsFilter
        - WhiteSpaceFilter
        - ParenthesesFilter
        - LongWordFilter
        - WordCountFilter
        - BoilerPlateStringFilter
        - MeanWordLengthFilter
        - RepeatedLinesFilter
        - RepeatedParagraphsFilter
        - RepeatedLinesByCharFilter
        - RepeatedParagraphsByCharFilter
        - RepeatingTopNGramsFilter
        - RepeatingDuplicateNGramsFilter
        - PunctuationFilter
        - EllipsisFilter
        - CommonEnglishWordsFilter
        - WordsWithoutAlphabetsFilter
        - PornographicUrlsFilter
        - TokenCountFilter
        - SubstringFilter
        - HistogramFilter
        
      code:  # 8 filters in stages/text/filters/code.py
        - AlphaFilter
        - GeneralCommentToCodeFilter
        - HTMLBoilerplateFilter
        - NumberOfLinesOfCodeFilter
        - PerExtensionFilter
        - PythonCommentToCodeFilter
        - TokenizerFertilityFilter
        - XMLHeaderFilter
        
      fasttext:
        - FastTextLangId: "Language identification"
        - FastTextQualityFilter: "FastText-based quality filtering"
        
    deduplication:
      exact:
        - stage: ExactDuplicateIdentification
          type: ProcessingStage
          purpose: "Hash-based exact matching"
          
      fuzzy:
        - workflow: FuzzyDeduplicationWorkflow
          type: WorkflowBase  # Important: not a stage!
          purpose: "MinHash + LSH + Connected Components"
          stages_used:
            - MinHashStage
            - LSHStage
            - BucketsToEdgesStage
            - ConnectedComponentsStage
            - IdentifyDuplicatesStage
            
      semantic:
        - workflow: SemanticDeduplicationWorkflow
          type: WorkflowBase  # Important: not a stage!
          purpose: "Embedding-based similarity clustering"
          stages_used:
            - EmbeddingCreatorStage
            - IdentifyDuplicatesStage
            - RankingStrategy
      
  video:
    io:
      - VideoReader: "CompositeStage - reads videos from path"
      - VideoReaderStage: "ProcessingStage - downloads and extracts metadata"
      - ClipWriterStage: "Writes clips to storage"
      
    clipping:
      - TransNetV2ClipExtractionStage: "ML-based scene detection (GPU)"
      - FixedStrideExtractorStage: "Fixed-duration clips (CPU)"
      - ClipTranscodingStage: "FFmpeg encoding (CPU/GPU)"
      
    embedding:
      - CosmosEmbed1FrameCreationStage: "Prepare frames for Cosmos"
      - CosmosEmbed1EmbeddingStage: "NVIDIA Cosmos embeddings (224p/336p/448p)"
      - InternVideo2EmbeddingStage: "InternVideo2 embeddings"
      
    captioning:
      - CaptionPreparationStage: "Prepare video windows for captioning"
      - CaptionGenerationStage: "Qwen VL captioning (GPU)"
      - CaptionEnhancementStage: "Caption refinement with Qwen LM"
      
    filtering:
      - MotionVectorDecodeStage: "Decode motion vectors (CPU)"
      - MotionFilterStage: "Filter static/low-motion clips"
      - ClipAestheticFilterStage: "Aesthetic quality filtering"
      
  image:
    embedding:
      - ImageEmbeddingStage: "CLIP embeddings for images"
      
    filtering:
      - AestheticFilterStage: "Aesthetic quality scoring"
      - NSFWFilterStage: "NSFW content detection"
      
  audio:
    inference:
      - InferenceAsrNemoStage: "NeMo ASR transcription"
      
    metrics:
      - WERCalculationStage: "Word Error Rate calculation"
```

---

## Implementation Plan

### Phase 1: Foundation (MVP)

**Goal**: Core orchestrator skill with stage reference and basic commands

**Deliverables**:

| Skill | Files | Scripts | Priority |
|-------|-------|---------|----------|
| `curator-os/` | `SKILL.md` | `detect_modality.py` | P0 |
| | `references/STAGE_REFERENCE.md` | | |
| | `references/MODALITY_PATTERNS.md` | | |
| `stages/` | `SKILL.md` | `search_stages.py` | P0 |
| `dedup-fuzzy/` | `SKILL.md` | `generate_fuzzy_config.py` | P0 |
| | `references/FUZZY_DEDUP_PARAMS.md` | `estimate_resources.py` | |
| `curate/` | `SKILL.md` | `generate_yaml.py` | P1 |
| | `assets/*.yaml` (templates) | | |
| `help/` | `SKILL.md` | | P1 |

**Scripts to Implement**:

```python
# P0 Scripts (required for MVP)
scripts/detect_modality.py      # Detect text/video/image/audio from path
scripts/search_stages.py        # Search stages by modality/category
scripts/generate_fuzzy_config.py # Generate fuzzy dedup YAML
scripts/estimate_resources.py   # Estimate GPU/memory for dataset size
scripts/generate_yaml.py        # Generic pipeline config generator
```

**Success Criteria**:
- `/curate` generates working pipeline config
- `/dedup-fuzzy` generates config with sensible defaults
- `/stages` returns accurate stage information
- Agent automatically routes curation requests to appropriate skill

### Phase 2: Modality-Specific Skills

**Goal**: Complete coverage for all four modalities

**Deliverables**:

| Skill | Files | Scripts |
|-------|-------|---------|
| `curate-video/` | `SKILL.md`, `references/VIDEO_STAGES.md` | `generate_video_config.py` |
| `curate-image/` | `SKILL.md`, `references/IMAGE_STAGES.md` | |
| `curate-audio/` | `SKILL.md`, `references/AUDIO_STAGES.md` | |
| `caption/` | `SKILL.md`, `references/CAPTIONING_MODELS.md` | |
| `embed/` | `SKILL.md` | |
| `download-cc/` | `SKILL.md` | `list_snapshots.py` |
| `download-wiki/` | `SKILL.md` | |

**Reference Files to Create**:

```markdown
# references/VIDEO_STAGES.md (~300 lines)
- VideoReader, VideoReaderStage, ClipWriterStage
- TransNetV2ClipExtractionStage, FixedStrideExtractorStage
- CosmosEmbed1*, InternVideo2* stages
- CaptionPreparation/Generation/Enhancement stages
- Motion and aesthetic filtering stages

# references/CAPTIONING_MODELS.md (~200 lines)
- Qwen VL model variants
- FP8 vs FP16 tradeoffs
- Batch size recommendations
```

**Success Criteria**:
- Video curation works with captioning and embedding
- Image curation with CLIP embedding and deduplication
- Audio curation with ASR transcription
- Common Crawl download with snapshot selection

### Phase 3: Validation & Utilities

**Goal**: Pipeline validation and helper utilities

**Deliverables**:

| Skill | Files | Scripts |
|-------|-------|---------|
| `validate-pipeline/` | `SKILL.md` | `validate_config.py` |
| `create-stage/` | `SKILL.md`, `assets/*.py` (templates) | |
| `filter/` | `SKILL.md`, `references/FILTER_CATALOG.md` | `list_filters.py` |
| `classify/` | `SKILL.md`, `references/CLASSIFIER_CATALOG.md` | |
| `curator-os/` | Add `scripts/validate_pipeline.py` | |

**Validation Scripts**:

```python
# scripts/validate_config.py
# - Check YAML syntax
# - Verify stage _target_ paths exist
# - Check stage input/output compatibility
# - Estimate resource requirements
# - Warn about common misconfigurations
```

**Asset Templates**:

```python
# assets/processing_stage_template.py
# assets/classifier_template.py
# assets/filter_template.py
# assets/composite_stage_template.py
```

**Success Criteria**:
- Invalid pipelines caught with clear error messages
- Custom stage templates generate valid code
- Filter/classifier catalogs are searchable

### Phase 4: Advanced Features

**Goal**: Semantic dedup, custom workflows, migration support

**Deliverables**:

| Skill | Files | Scripts |
|-------|-------|---------|
| `dedup-exact/` | `SKILL.md` | `generate_exact_config.py` |
| `dedup-semantic/` | `SKILL.md` | `generate_semantic_config.py` |
| `curator-os/` | Add `references/RESOURCE_GUIDE.md` | `estimate_cluster_size.py` |

**Success Criteria**:
- All three deduplication methods available
- Resource estimation for large-scale clusters
- Clear migration path for API changes

### Implementation Notes

**Skill Naming Conventions**:
- Lowercase with hyphens: `dedup-fuzzy`, `download-cc`
- Match functionality: `/dedup-fuzzy` does fuzzy deduplication
- No version numbers in names

**Script Requirements**:
- All scripts must be executable: `chmod +x scripts/*.py`
- Use `#!/usr/bin/env python3` shebang
- Output JSON for structured data
- Include `--help` with argparse
- Exit code 0 on success, non-zero on error

**Reference File Guidelines**:
- Keep under 500 lines each
- Use tables for stage catalogs
- Include source file paths for verification
- Update when NeMo Curator API changes

**Testing Strategy**:
- Unit tests for each script
- Integration tests with sample data
- Validation against actual NeMo Curator execution
- Cross-agent testing (Cursor, Claude, Codex)

---

## Open Questions

### Q1: Skills Storage Location

**Options**:
1. `.cursor/skills/` in NeMo Curator repo (proposed)
2. Separate `curator-os` GitHub repo (installable via Cursor's remote skills feature)
3. User-level global skills at `~/.cursor/skills/`

**Recommendation**: Option 1 for project-level, with Option 2 as distribution mechanism.

**Rationale**: 
- Project-level ensures skills stay synchronized with the codebase
- GitHub repo enables `Cursor Settings → Add Rule → Remote Rule (Github)` installation
- Users without NeMo Curator repo can still install skills globally

### Q2: YAML vs Python Output

**Question**: Should generated configs be YAML (Hydra) or Python code?

**Current State**: NeMo Curator already supports Hydra YAML configs via `nemo_curator/config/`.
Example configs exist for:
- `fuzzy_deduplication_pipeline.yaml`
- `heuristic_filter_english_pipeline.yaml`
- `semantic_deduplication_pipeline.yaml`
- And others in `nemo_curator/config/text/`

**Recommendation**: Leverage existing Hydra infrastructure:
- `::yaml` generates Hydra YAML config (matches existing `nemo_curator.config.run` pattern)
- `::python` generates Python Pipeline code for programmatic use
- For WorkflowBase classes (FuzzyDeduplication, SemanticDeduplication), generate the workflow instantiation
- Default to YAML since infrastructure already exists

**Example existing YAML** (from `fuzzy_deduplication_pipeline.yaml`):
```yaml
workflow:
  - _target_: nemo_curator.stages.deduplication.fuzzy.workflow.FuzzyDeduplicationWorkflow
    input_path: ${input_path}
    output_path: ${output_path}
    cache_path: ${cache_path}
    # ... parameters
```

### Q3: Integration with Existing Tutorials

**Question**: How do CURATOR-OS workflows relate to existing tutorials?

**Recommendation**: 
- Tutorials teach concepts and manual pipeline construction
- CURATOR-OS accelerates common patterns
- Cross-link between them

### Q4: Versioning

**Question**: How to handle API changes in NeMo Curator?

**Recommendation**:
- Use `metadata.nemo-curator-version` in frontmatter to specify compatibility
- Scripts can check installed version and warn on mismatch
- Major version changes trigger skill updates

**Example frontmatter**:
```yaml
metadata:
  nemo-curator-version: ">=0.5.0,<1.0.0"
```

### Q5: Script Language Choice

**Question**: Should scripts be Python-only or support multiple languages?

**Options**:
1. Python-only (matches NeMo Curator ecosystem)
2. Python + Bash (common combination)
3. Any executable (maximum flexibility)

**Recommendation**: Python-only for Phase 1-3, consider Bash for simple utilities in Phase 4.

**Rationale**:
- Python matches NeMo Curator's ecosystem
- Can import `nemo_curator` modules for validation
- Easier to maintain single language
- Bash only for shell-level operations (listing files, etc.)

### Q6: Cross-Agent Compatibility

**Question**: How to ensure skills work across Cursor, Claude, and Codex agents?

**Considerations**:
- Agent Skills is an open standard, but implementations may vary
- Script execution capabilities differ between agents
- Some agents may not support all directories

**Recommendation**:
- Keep `SKILL.md` instructions agent-agnostic
- Scripts should be optional enhancements, not requirements
- Test on multiple agents during development
- Document known limitations in `compatibility` field

### Q7: Reference File Size Limits

**Question**: What's the optimal size for reference files?

**Agent Skills recommendation**: Keep main SKILL.md under 500 lines, use references for detail.

**Proposed limits for CURATOR-OS**:
| File Type | Max Lines | Rationale |
|-----------|-----------|-----------|
| `SKILL.md` | 300 | Core instructions only |
| `references/*.md` | 500 | Single topic depth |
| `assets/*.yaml` | 200 | Template complexity |

### Q8: Skill Discoverability

**Question**: How do users discover available skills?

**Options**:
1. `/help` skill lists all skills
2. `curator-os` orchestrator suggests relevant skills
3. Cursor Settings → Rules shows all skills
4. README in `.cursor/skills/`

**Recommendation**: All four - defense in depth for discoverability

---

## References

### Internal Sources (Validated)

| Reference | Path | Description |
|-----------|------|-------------|
| ProcessingStage Base | `nemo_curator/stages/base.py` | Base class definitions for all stages |
| CompositeStage | `nemo_curator/stages/base.py` | Decomposable stage pattern |
| WorkflowBase | `nemo_curator/pipeline/workflow.py` | Multi-pipeline orchestration base |
| Resources | `nemo_curator/stages/resources.py` | Resource configuration dataclass |
| Pipeline | `nemo_curator/pipeline/pipeline.py` | Pipeline implementation |
| FuzzyDeduplicationWorkflow | `nemo_curator/stages/deduplication/fuzzy/workflow.py` | Reference workflow implementation |
| Text Classifiers | `nemo_curator/stages/text/classifiers/` | All classifier implementations |
| Heuristic Filters | `nemo_curator/stages/text/filters/heuristic_filter.py` | 25 heuristic filter classes |
| Video Stages | `nemo_curator/stages/video/` | Video processing stages |
| Hydra Configs | `nemo_curator/config/` | Existing YAML configuration examples |
| Task Types | `nemo_curator/tasks/` | Task definitions by modality |

### Existing Cursor Rules (Applied)

The following rules already exist in `.cursor/rules/` and inform this RFC:

| Rule | Description |
|------|-------------|
| `processing-stage-patterns.mdc` | ProcessingStage implementation patterns |
| `composite-stage-patterns.mdc` | CompositeStage decomposition patterns |
| `task-patterns.mdc` | Task types and usage |
| `resources-configuration.mdc` | Resource allocation guidance |
| `pipeline-structure.mdc` | Pipeline composition |
| `executors.mdc` | Executor backends |
| `modality-structure.mdc` | Modality-specific stage organization |
| `coding-standards.mdc` | Code style requirements |

### External

- [Agent Skills Specification](https://agentskills.io/specification) - Open standard for skill format
- [Cursor Agent Skills Documentation](https://cursor.com/docs/context/skills) - Cursor-specific implementation
- [DORI Documentation Framework](https://gitlab.com/tech-docs/prompt-library) - Inspiration for modular architecture
- [Hydra Configuration](https://hydra.cc/) - NeMo Curator uses Hydra for YAML configs

### Agent Skills Resources

- [Agent Skills Website](https://agentskills.io) - Official specification and examples
- [skills-ref Validator](https://github.com/agentskills/agentskills/tree/main/skills-ref) - Validate SKILL.md files
- [Cursor Skills Migration Guide](https://cursor.com/docs/context/skills#migrating-rules-and-commands-to-skills) - Converting existing rules

---

## Appendix A: Complete Stage Inventory

> **Note**: This inventory was validated against `nemo_curator/stages/` on 2026-01-26.
> Stage names are exact class names from the codebase.

### Text Stages (75+)

| Category | Stages | Source |
|----------|--------|--------|
| **IO** | `JsonlReader`, `ParquetReader`, `JsonlWriter`, `ParquetWriter`, `MegatronTokenizerWriter` | `stages/text/io/` |
| **Classifiers** | `QualityClassifier`, `DomainClassifier`, `MultilingualDomainClassifier`, `ContentTypeClassifier`, `FineWebEduClassifier`, `FineWebMixtralEduClassifier`, `FineWebNemotronEduClassifier`, `AegisClassifier`, `InstructionDataGuardClassifier`, `PromptTaskComplexityClassifier` | `stages/text/classifiers/` |
| **Heuristic Filters** | `NonAlphaNumericFilter`, `SymbolsToWordsFilter`, `NumbersFilter`, `UrlsFilter`, `BulletsFilter`, `WhiteSpaceFilter`, `ParenthesesFilter`, `LongWordFilter`, `WordCountFilter`, `BoilerPlateStringFilter`, `MeanWordLengthFilter`, `RepeatedLinesFilter`, `RepeatedParagraphsFilter`, `RepeatedLinesByCharFilter`, `RepeatedParagraphsByCharFilter`, `RepeatingTopNGramsFilter`, `RepeatingDuplicateNGramsFilter`, `PunctuationFilter`, `EllipsisFilter`, `CommonEnglishWordsFilter`, `WordsWithoutAlphabetsFilter`, `PornographicUrlsFilter`, `TokenCountFilter`, `SubstringFilter`, `HistogramFilter` | `stages/text/filters/heuristic_filter.py` |
| **Code Filters** | `AlphaFilter`, `GeneralCommentToCodeFilter`, `HTMLBoilerplateFilter`, `NumberOfLinesOfCodeFilter`, `PerExtensionFilter`, `PythonCommentToCodeFilter`, `TokenizerFertilityFilter`, `XMLHeaderFilter` | `stages/text/filters/code.py` |
| **FastText Filters** | `FastTextLangId`, `FastTextQualityFilter` | `stages/text/filters/fasttext_filter.py` |
| **Modifiers** | `BoilerPlateStringModifier`, `DocumentModifier`, `FastTextLabelModifier`, `LineRemover`, `MarkdownRemover`, `NewlineNormalizer`, `QuotationRemover`, `Slicer`, `UnicodeReformatter`, `UrlRemover` | `stages/text/modifiers/` |
| **Modules** | `AddId`, `DocumentSplitter`, `DocumentJoiner`, `Modify`, `Score`, `Filter`, `ScoreFilter` | `stages/text/modules/` |
| **Embedders** | `EmbeddingCreatorStage` | `stages/text/embedders/` |
| **Download** | `CommonCrawlDownloadExtractStage`, `WikipediaDownloadExtractStage`, `ArxivDownloadExtractStage` | `stages/text/download/` |

### Deduplication Stages (Shared)

| Category | Stages | Type | Source |
|----------|--------|------|--------|
| **Fuzzy** | `MinHashStage`, `LSHStage`, `BucketsToEdgesStage`, `ConnectedComponentsStage`, `IdentifyDuplicatesStage` | ProcessingStage | `stages/deduplication/fuzzy/` |
| **Fuzzy Workflow** | `FuzzyDeduplicationWorkflow` | **WorkflowBase** | `stages/deduplication/fuzzy/workflow.py` |
| **Exact** | `ExactDuplicateIdentification` | ProcessingStage | `stages/deduplication/exact/` |
| **Semantic** | `IdentifyDuplicatesStage`, `RankingStrategy` | ProcessingStage | `stages/deduplication/semantic/` |
| **Semantic Workflow** | `SemanticDeduplicationWorkflow` | **WorkflowBase** | `stages/deduplication/semantic/workflow.py` |
| **Utilities** | `FilePartitioningStage`, `ClientPartitioningStage` | ProcessingStage | `stages/` |

### Video Stages (20+)

| Category | Stages | Source |
|----------|--------|--------|
| **IO** | `VideoReader` (CompositeStage), `VideoReaderStage` (ProcessingStage), `ClipWriterStage` | `stages/video/io/` |
| **Clipping** | `TransNetV2ClipExtractionStage`, `FixedStrideExtractorStage`, `ClipTranscodingStage`, `VideoFrameExtractionStage`, `ClipFrameExtractionStage` | `stages/video/clipping/` |
| **Filtering** | `MotionVectorDecodeStage`, `MotionFilterStage`, `ClipAestheticFilterStage` | `stages/video/filtering/` |
| **Embedding** | `CosmosEmbed1FrameCreationStage`, `CosmosEmbed1EmbeddingStage`, `InternVideo2FrameCreationStage`, `InternVideo2EmbeddingStage` | `stages/video/embedding/` |
| **Captioning** | `CaptionPreparationStage`, `CaptionGenerationStage`, `CaptionEnhancementStage` | `stages/video/caption/` |
| **Preview** | `PreviewStage` | `stages/video/preview/` |

### Image Stages (8+)

| Category | Stages | Source |
|----------|--------|--------|
| **IO** | `ImageReaderStage`, `ImageWriterStage` | `stages/image/io/` |
| **Embedding** | `ImageEmbeddingStage` (uses CLIP model) | `stages/image/embedders/` |
| **Filtering** | `AestheticFilterStage`, `NSFWFilterStage` | `stages/image/filters/` |
| **Deduplication** | `ImageDuplicateRemovalStage` | `stages/image/deduplication/` |

### Audio Stages (4+)

| Category | Stages | Source |
|----------|--------|--------|
| **Inference** | `InferenceAsrNemoStage` | `stages/audio/inference/` |
| **Metrics** | `WERCalculationStage` | `stages/audio/metrics/` |
| **Datasets** | `FLEURSManifestStage` | `stages/audio/datasets/` |
| **IO** | `AudioConverter` | `stages/audio/io/`

---

## Appendix B: Example CURATOR-OS Session

```
User: I have 500GB of Common Crawl data that I want to prepare for LLM training

CURATOR-OS:
🎯 Task Analysis

Detected Intent: Full text curation pipeline
Data Size: 500GB (large dataset)
Target: LLM training data

Recommended Pipeline:

┌──────────────────────────────────────────────────────────────┐
│ Step 1: Heuristic Filtering                                  │
│ ├── 25 quality filters (English-optimized)                   │
│ ├── Estimated reduction: 40-60%                              │
│ └── Resources: 4 CPUs per worker                             │
├──────────────────────────────────────────────────────────────┤
│ Step 2: Quality Classification                               │
│ ├── FineWeb-Edu classifier                                   │
│ ├── Keep: High + Medium quality                              │
│ └── Resources: 1 GPU (16GB), 2 CPUs                          │
├──────────────────────────────────────────────────────────────┤
│ Step 3: Fuzzy Deduplication                                  │
│ ├── MinHash (char_ngrams=24)                                 │
│ ├── LSH (bands=20, hashes=13)                                │
│ ├── Connected Components                                      │
│ └── Resources: 2 GPUs (32GB each), 8 CPUs                    │
├──────────────────────────────────────────────────────────────┤
│ Step 4: Export                                               │
│ └── Parquet format, optimized for training                   │
└──────────────────────────────────────────────────────────────┘

Estimated:
- Final size: ~100-150GB (70-80% reduction)
- Time: 8-12 hours on 8-GPU cluster
- GPU memory: 32GB minimum per GPU

⚠️ Recommendations for 500GB dataset:
- Use bands_per_iteration=3 to manage memory
- Consider running dedup in streaming mode
- Monitor LSH shuffle stage for OOM

Options:
[a] Execute with recommended settings
[b] Generate YAML configuration
[c] Configure parameters
[d] Show stage details

User: b

CURATOR-OS:
📄 Generated: text_curation_pipeline.yaml

```yaml
# text_curation_pipeline.yaml
# Generated by CURATOR-OS
# Based on existing NeMo Curator Hydra config patterns

defaults:
  - _self_
  - override hydra/job_logging: none
  - override hydra/hydra_logging: none

hydra:
  run:
    dir: .
  output_subdir: null

input_path: ???  # Set your input path
output_path: ???  # Set your output path
cache_path: ${output_path}/.cache
text_field: "text"

ray_client:
  _target_: nemo_curator.core.client.RayClient
  num_cpus: null  # Use all available
  num_gpus: 4

# For filtering pipeline, use stages directly
# For deduplication, use the workflow class
workflow:
  # Option A: FuzzyDeduplicationWorkflow (handles entire dedup pipeline)
  - _target_: nemo_curator.stages.deduplication.fuzzy.workflow.FuzzyDeduplicationWorkflow
    input_path: ${input_path}
    output_path: ${output_path}
    cache_path: ${cache_path}
    input_filetype: parquet
    input_blocksize: "1GiB"
    text_field: ${text_field}
    perform_removal: false
    # MinHash + LSH Config
    seed: 42
    char_ngrams: 24
    num_bands: 20
    minhashes_per_band: 13
    use_64_bit_hash: false
    bands_per_iteration: 3  # Reduced for 500GB dataset
    env_vars: null
```

For filtering (separate pipeline):
```yaml
# heuristic_filter_pipeline.yaml
stages:
  - _target_: nemo_curator.stages.text.modules.ScoreFilter
    filter_obj:
      _target_: nemo_curator.stages.text.filters.WordCountFilter
      min_words: 50
      max_words: 100000
    text_field: ${text_field}
    
  # ... additional filters
  
  - _target_: nemo_curator.stages.text.classifiers.FineWebEduClassifier
    # Quality classification
```

Saved to: ./text_curation_pipeline.yaml

Run with:
  python -m nemo_curator.config.run --config-path=. --config-name=text_curation_pipeline \
    input_path=/data/crawl output_path=/data/curated cache_path=/data/cache
```

---

## Appendix C: Complete SKILL.md Example

This appendix provides a complete, implementable example of the `dedup-fuzzy` skill.

### File: `.cursor/skills/dedup-fuzzy/SKILL.md`

```markdown
---
name: dedup-fuzzy
description: |
  Perform fuzzy deduplication on text datasets using MinHash + LSH algorithm.
  Use when the user wants to remove near-duplicate documents, mentions fuzzy
  matching, MinHash, LSH, similarity-based deduplication, or has large text
  datasets with potential duplicates. Supports datasets from 1GB to multi-TB scale.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: text
  gpu-required: true
  nemo-curator-version: ">=0.5.0"
compatibility: Requires Python 3.10+, Ray cluster, GPU with 16GB+ memory
disable-model-invocation: true
---

# Fuzzy Deduplication

Remove near-duplicate documents from text datasets using MinHash signatures and
Locality Sensitive Hashing (LSH).

## When to Use

- Large text datasets (>1GB) with potential duplicates
- Need similarity-based matching (not exact duplicates)
- Common Crawl, web scrape, or aggregated data sources
- Preparing training data for LLMs

## Quick Start

1. Ensure you have input data in Parquet or JSONL format
2. Run the config generator script
3. Execute with NeMo Curator

## Usage

### Generate Configuration

Run the config generator to create a YAML file:

\`\`\`bash
python scripts/generate_fuzzy_config.py \
  --input-path /data/text \
  --output-path /data/deduped \
  --cache-path /data/cache \
  --output-file fuzzy_dedup.yaml
\`\`\`

### Estimate Resources

Before running on large datasets, estimate resource requirements:

\`\`\`bash
python scripts/estimate_resources.py \
  --input-path /data/text \
  --input-format parquet
\`\`\`

### Execute Pipeline

\`\`\`bash
python -m nemo_curator.config.run \
  --config-path=. \
  --config-name=fuzzy_dedup \
  input_path=/data/text \
  output_path=/data/deduped \
  cache_path=/data/cache
\`\`\`

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `char_ngrams` | 24 | 20-50 | Shingle size for hashing. Lower = more false positives |
| `num_bands` | 20 | 5-50 | Number of LSH bands. More = finer similarity threshold |
| `minhashes_per_band` | 13 | 5-25 | Hashes per band. More = stricter matching |
| `bands_per_iteration` | 5 | 1-num_bands | Memory control. Lower for large datasets |
| `use_64_bit_hash` | false | true/false | 64-bit for very large datasets (>1B docs) |

### Parameter Tuning Guide

See [references/FUZZY_DEDUP_PARAMS.md](references/FUZZY_DEDUP_PARAMS.md) for detailed tuning guidance.

**Quick rules:**
- **< 100GB data**: Use defaults (`bands_per_iteration=5`)
- **100GB-500GB**: Reduce `bands_per_iteration=3`
- **> 500GB**: Use `bands_per_iteration=1-2`, consider 64-bit hash
- **High precision needed**: Increase `char_ngrams` to 30-40
- **More aggressive dedup**: Reduce `num_bands` to 10-15

## Outputs

The workflow produces:

1. **Duplicate IDs**: `{output_path}/duplicate_ids.parquet`
   - Contains document IDs to remove
   - Can be used for downstream removal

2. **ID Generator Mapping**: `{output_path}/fuzzy_id_generator.json`
   - Maps internal IDs to original document IDs
   - Required for removal workflows

## Common Issues

### Out of Memory (OOM)

Reduce `bands_per_iteration`:
\`\`\`bash
python scripts/generate_fuzzy_config.py \
  --bands-per-iteration 1 \
  ...
\`\`\`

### High False Positive Rate

Increase `char_ngrams`:
\`\`\`bash
python scripts/generate_fuzzy_config.py \
  --char-ngrams 30 \
  ...
\`\`\`

### Slow Performance

Ensure GPU is being used. Check Ray cluster has GPUs allocated.

## Related Skills

- `/dedup-exact` - For exact duplicate removal (faster, less memory)
- `/dedup-semantic` - For semantic similarity deduplication
- `/curate` - Full curation workflow including deduplication
\`\`\`

### File: `.cursor/skills/dedup-fuzzy/scripts/generate_fuzzy_config.py`

```python
#!/usr/bin/env python3
"""Generate NeMo Curator fuzzy deduplication YAML configuration.

Examples:
    # Basic usage
    python generate_fuzzy_config.py --input-path /data/text --output-path /data/deduped --cache-path /data/cache

    # With custom parameters
    python generate_fuzzy_config.py --input-path /data/text --output-path /data/deduped --cache-path /data/cache \
        --char-ngrams 30 --num-bands 15 --bands-per-iteration 2

    # Save to file
    python generate_fuzzy_config.py --input-path /data/text --output-path /data/deduped --cache-path /data/cache \
        --output-file fuzzy_dedup.yaml
"""
import argparse
import json
import sys
from pathlib import Path


def generate_config(
    input_path: str,
    output_path: str,
    cache_path: str,
    input_filetype: str = "parquet",
    text_field: str = "text",
    char_ngrams: int = 24,
    num_bands: int = 20,
    minhashes_per_band: int = 13,
    bands_per_iteration: int = 5,
    use_64_bit_hash: bool = False,
    seed: int = 42,
) -> str:
    """Generate Hydra YAML configuration for fuzzy deduplication."""
    return f'''# Fuzzy Deduplication Pipeline
# Generated by CURATOR-OS dedup-fuzzy skill
# Run with: python -m nemo_curator.config.run --config-path=. --config-name=<this-file>

defaults:
  - _self_
  - override hydra/job_logging: none
  - override hydra/hydra_logging: none

hydra:
  run:
    dir: .
  output_subdir: null

# I/O Configuration
input_path: {input_path}
output_path: {output_path}
cache_path: {cache_path}
input_filetype: {input_filetype}
text_field: "{text_field}"

# Ray Client
ray_client:
  _target_: nemo_curator.core.client.RayClient
  num_cpus: null  # Use all available
  num_gpus: 4     # Adjust based on cluster

# Workflow
workflow:
  - _target_: nemo_curator.stages.deduplication.fuzzy.workflow.FuzzyDeduplicationWorkflow
    input_path: ${{input_path}}
    output_path: ${{output_path}}
    cache_path: ${{cache_path}}
    input_filetype: ${{input_filetype}}
    input_blocksize: "1GiB"
    text_field: ${{text_field}}
    perform_removal: false
    # MinHash + LSH Configuration
    seed: {seed}
    char_ngrams: {char_ngrams}
    num_bands: {num_bands}
    minhashes_per_band: {minhashes_per_band}
    use_64_bit_hash: {str(use_64_bit_hash).lower()}
    bands_per_iteration: {bands_per_iteration}
    env_vars: null
'''


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-path", required=True, help="Path to input data")
    parser.add_argument("--output-path", required=True, help="Path for output (duplicate IDs)")
    parser.add_argument("--cache-path", required=True, help="Path for intermediate cache files")
    parser.add_argument("--input-filetype", default="parquet", choices=["parquet", "jsonl"])
    parser.add_argument("--text-field", default="text", help="Field containing text to deduplicate")
    parser.add_argument("--char-ngrams", type=int, default=24, help="Shingle size (20-50)")
    parser.add_argument("--num-bands", type=int, default=20, help="Number of LSH bands (5-50)")
    parser.add_argument("--minhashes-per-band", type=int, default=13, help="Hashes per band (5-25)")
    parser.add_argument("--bands-per-iteration", type=int, default=5, help="Memory control (1-num_bands)")
    parser.add_argument("--use-64-bit-hash", action="store_true", help="Use 64-bit hash for large datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-file", help="Write config to file (default: stdout)")

    args = parser.parse_args()

    # Validate parameters
    if args.char_ngrams < 20:
        print(json.dumps({"warning": "char_ngrams < 20 may cause high false positive rate (~5%)"}), file=sys.stderr)

    if args.bands_per_iteration > args.num_bands:
        print(json.dumps({"error": "bands_per_iteration must be <= num_bands"}), file=sys.stderr)
        sys.exit(1)

    config = generate_config(
        input_path=args.input_path,
        output_path=args.output_path,
        cache_path=args.cache_path,
        input_filetype=args.input_filetype,
        text_field=args.text_field,
        char_ngrams=args.char_ngrams,
        num_bands=args.num_bands,
        minhashes_per_band=args.minhashes_per_band,
        bands_per_iteration=args.bands_per_iteration,
        use_64_bit_hash=args.use_64_bit_hash,
        seed=args.seed,
    )

    if args.output_file:
        Path(args.output_file).write_text(config)
        print(json.dumps({
            "status": "success",
            "file": args.output_file,
            "run_command": f"python -m nemo_curator.config.run --config-path=. --config-name={Path(args.output_file).stem}"
        }))
    else:
        print(config)


if __name__ == "__main__":
    main()
```

### File: `.cursor/skills/dedup-fuzzy/references/FUZZY_DEDUP_PARAMS.md`

```markdown
# Fuzzy Deduplication Parameter Tuning Guide

## Parameter Relationships

The fuzzy deduplication algorithm uses MinHash + LSH with these key parameters:

### Total Hashes

`num_hashes = num_bands × minhashes_per_band`

Default: 20 × 13 = 260 hashes per document

### Similarity Threshold

The similarity threshold is approximately:

`threshold ≈ (1/num_bands)^(1/minhashes_per_band)`

With defaults: `(1/20)^(1/13) ≈ 0.80` (80% similarity)

### Memory vs Speed Tradeoff

`bands_per_iteration` controls how many bands are processed in a single shuffle:

| Value | Memory | Speed | Use Case |
|-------|--------|-------|----------|
| num_bands | Highest | Fastest | Small datasets (<50GB) |
| 5 | Medium | Medium | Medium datasets (50-200GB) |
| 1 | Lowest | Slowest | Large datasets (>500GB) |

## Dataset Size Recommendations

| Dataset Size | char_ngrams | num_bands | bands_per_iteration | 64-bit hash |
|--------------|-------------|-----------|---------------------|-------------|
| < 10GB | 24 | 20 | 20 | No |
| 10-100GB | 24 | 20 | 5 | No |
| 100-500GB | 24 | 20 | 3 | No |
| 500GB-1TB | 24-30 | 20 | 1-2 | Consider |
| > 1TB | 30 | 20 | 1 | Yes |
| > 10B docs | 30-40 | 15-20 | 1 | Yes |

## Common Adjustments

### High False Positive Rate

Symptoms: Too many documents marked as duplicates

Fix:
- Increase `char_ngrams` from 24 to 30-40
- Decrease `num_bands` to raise similarity threshold

### Missing Duplicates

Symptoms: Known duplicates not detected

Fix:
- Decrease `char_ngrams` to 20
- Increase `num_bands` to lower similarity threshold

### Out of Memory

Symptoms: OOM errors during LSH or connected components

Fix:
- Decrease `bands_per_iteration` to 1-3
- Ensure sufficient GPU memory (16GB minimum)
- Check Ray cluster memory allocation
```

---

*End of RFC-001*
