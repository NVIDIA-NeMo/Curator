---
name: stages
description: |
  Look up NeMo Curator processing stages by modality and category.
  Use when the user asks about available stages, what stages do,
  stage parameters, or needs to find a specific processing capability.
  Provides searchable catalog of all text, video, image, and audio stages.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
disable-model-invocation: true
---

# Stage Reference Lookup

Search and browse NeMo Curator processing stages.

## When to Use

- User asks "what stages are available for X?"
- User needs to find a specific processing capability
- User wants to know stage parameters or requirements
- Building custom pipelines and need stage reference

## Quick Start

### Search by Modality

```bash
# List all text stages
python scripts/search_stages.py --modality text

# List all video stages
python scripts/search_stages.py --modality video

# List all image stages
python scripts/search_stages.py --modality image

# List all audio stages
python scripts/search_stages.py --modality audio
```

### Search by Category

```bash
# Text filters
python scripts/search_stages.py --modality text --category filters

# Text classifiers
python scripts/search_stages.py --modality text --category classifiers

# Video clipping stages
python scripts/search_stages.py --modality video --category clipping
```

### Search by Name

```bash
# Find stages matching a pattern
python scripts/search_stages.py --search "dedup"
python scripts/search_stages.py --search "quality"
python scripts/search_stages.py --search "caption"
```

## Stage Types

Understanding the execution hierarchy is crucial:

```
┌─────────────────────────────────────────────────────────────────┐
│                        WorkflowBase                              │
│  - Has run() method that creates/executes multiple Pipelines    │
│  - Examples: FuzzyDeduplicationWorkflow, SemanticDeduplication  │
│  - NOT a stage - orchestrates entire workflows programmatically │
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

## Categories by Modality

### Text

| Category | Description | Example Stages |
|----------|-------------|----------------|
| `io` | Read/write data | JsonlReader, ParquetWriter |
| `classifiers` | ML classification | QualityClassifier, DomainClassifier |
| `filters` | Quality filtering | WordCountFilter, UrlsFilter |
| `modifiers` | Text transformation | UnicodeReformatter, UrlRemover |
| `modules` | Utility wrappers | ScoreFilter, AddId |
| `deduplication` | Duplicate removal | FuzzyDeduplicationWorkflow |
| `download` | Data sources | CommonCrawlDownloadExtractStage |
| `embedders` | Embedding generation | EmbeddingCreatorStage |

### Video

| Category | Description | Example Stages |
|----------|-------------|----------------|
| `io` | Read/write video | VideoReader, ClipWriterStage |
| `clipping` | Scene/clip extraction | TransNetV2ClipExtractionStage |
| `captioning` | Video captions | CaptionGenerationStage |
| `embedding` | Video embeddings | CosmosEmbed1EmbeddingStage |
| `filtering` | Quality filtering | MotionFilterStage |

### Image

| Category | Description | Example Stages |
|----------|-------------|----------------|
| `io` | Read/write images | ImageReaderStage, ImageWriterStage |
| `embedding` | Image embeddings | ImageEmbeddingStage (CLIP) |
| `filtering` | Quality/safety | ImageAestheticFilterStage, ImageNSFWFilterStage |
| `deduplication` | Duplicate removal | ImageDuplicatesRemovalStage |

### Audio

| Category | Description | Example Stages |
|----------|-------------|----------------|
| `io` | Convert audio data | AudioToDocumentStage |
| `inference` | ASR models | InferenceAsrNemoStage |
| `metrics` | Quality metrics | GetPairwiseWerStage |

## Common Lookups

### "I need to filter text by quality"

```
Heuristic filters: WordCountFilter, NonAlphaNumericFilter, etc. (25 filters)
ML classifiers: QualityClassifier, FineWebEduClassifier
```

### "I need to remove duplicates"

```
Exact: ExactDuplicateIdentification
Fuzzy: FuzzyDeduplicationWorkflow (WorkflowBase)
Semantic: SemanticDeduplicationWorkflow (WorkflowBase)
```

### "I need to process video"

```
Read: VideoReader (CompositeStage)
Clip: TransNetV2ClipExtractionStage (GPU) or FixedStrideExtractorStage (CPU)
Caption: CaptionGenerationStage (GPU, Qwen VL)
Embed: CosmosEmbed1EmbeddingStage or InternVideo2EmbeddingStage
```

### "I need to generate embeddings"

```
Text: EmbeddingCreatorStage (vLLM)
Video: CosmosEmbed1EmbeddingStage, InternVideo2EmbeddingStage
Image: ImageEmbeddingStage (CLIP)
```

## Resource Requirements

### GPU Stages

| Stage | GPU Memory | Notes |
|-------|------------|-------|
| TransNetV2ClipExtractionStage | 16 GB | Scene detection |
| CaptionGenerationStage | 24 GB | Full GPU |
| CosmosEmbed1EmbeddingStage | 20 GB | NVIDIA Cosmos |
| InternVideo2EmbeddingStage | 16 GB | Video embeddings |
| ImageEmbeddingStage | ~4 GB | CLIP embeddings |
| ImageAestheticFilterStage | ~4 GB | Aesthetic scoring |
| ImageNSFWFilterStage | ~4 GB | Safety filtering |
| QualityClassifier | ~8 GB | Varies by model |
| InferenceAsrNemoStage | 16 GB | NeMo ASR |

### CPU-Only Stages

- All heuristic filters
- FixedStrideExtractorStage
- MotionVectorDecodeStage
- IO stages (readers/writers)

## Script Reference

### search_stages.py

```bash
python scripts/search_stages.py [OPTIONS]

Options:
  --modality TEXT     Filter by modality (text, video, image, audio)
  --category TEXT     Filter by category (io, filters, classifiers, etc.)
  --search TEXT       Search by stage name
  --gpu-only          Show only GPU stages
  --cpu-only          Show only CPU stages
  --json              Output as JSON
  --help              Show help
```

## Related Skills

- `/curate` - Full curation workflow
- `/filter` - Detailed filter catalog
- `/classify` - Detailed classifier catalog
- `/help` - General help
