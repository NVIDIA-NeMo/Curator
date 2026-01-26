---
name: curator-os
description: |
  Intelligent command interface for NeMo Curator data curation pipelines.
  Use when the user mentions data curation, deduplication, filtering,
  classification, video/text/image/audio processing, pipeline building,
  or working with NeMo Curator. Routes user intent to appropriate curation skills.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  nemo-curator-version: ">=0.5.0"
compatibility: Requires Python 3.10+, Ray cluster for distributed execution
---

# CURATOR-OS

Intelligent command interface for NeMo Curator data curation pipelines.

## When to Use

This skill activates automatically when users:
- Ask about data curation or preprocessing
- Want to deduplicate, filter, or classify data
- Work with text, video, image, or audio datasets
- Need to build NeMo Curator pipelines
- Ask about available stages or processing options

## Available Skills

| Skill | Purpose | Invoke |
|-------|---------|--------|
| `/curate` | Full curation workflow | Multi-stage pipeline |
| `/dedup-fuzzy` | MinHash + LSH deduplication | Near-duplicate removal |
| `/dedup-exact` | Hash-based deduplication | Exact duplicate removal |
| `/dedup-semantic` | Embedding-based deduplication | Semantic similarity |
| `/filter` | Heuristic text filtering | Rule-based quality filters |
| `/classify` | ML classification | Quality, domain, safety |
| `/stages` | Stage reference lookup | Find processing stages |
| `/help` | Context-aware help | Get guidance |

## Modality Detection

Before recommending a workflow, detect the data modality:

```bash
python scripts/detect_modality.py /path/to/data
```

### Modality → Task Type Mapping

| Modality | Task Type | Common Stages |
|----------|-----------|---------------|
| Text | `DocumentBatch` | Filters, Classifiers, Deduplication |
| Video | `VideoTask` | Clipping, Captioning, Embedding |
| Image | `ImageBatch` | CLIP Embedding, Aesthetic Filter |
| Audio | `AudioBatch` | ASR, WER Calculation |

## Routing Logic

### Text Data Requests

If user mentions text, documents, JSONL, Parquet, Common Crawl:
1. Check for deduplication need → `/dedup-fuzzy` or `/dedup-exact`
2. Check for filtering need → `/filter`
3. Check for classification need → `/classify`
4. Full pipeline → `/curate`

### Video Data Requests

If user mentions video, MP4, clips, scenes:
1. Scene detection → TransNetV2ClipExtractionStage
2. Captioning → CaptionGenerationStage
3. Embedding → CosmosEmbed1 or InternVideo2

### Image Data Requests

If user mentions images, photos, PNG, JPEG:
1. Embedding → ImageEmbeddingStage (CLIP)
2. Filtering → AestheticFilterStage, NSFWFilterStage

### Audio Data Requests

If user mentions audio, speech, WAV, transcription:
1. ASR → InferenceAsrNemoStage
2. Quality → WERCalculationStage

## Pipeline Validation

Before generating any pipeline configuration, validate:

```bash
python scripts/validate_pipeline.py config.yaml
```

Checks:
- [ ] All `_target_` paths exist in NeMo Curator
- [ ] Stage input/output compatibility
- [ ] Resource requirements are reasonable
- [ ] Required fields are present

## Resource Estimation

For large datasets, estimate resources before execution:

| Dataset Size | Recommended |
|--------------|-------------|
| < 10GB | Single node, 1 GPU |
| 10-100GB | Single node, 4 GPUs |
| 100GB-1TB | Multi-node, 8+ GPUs |
| > 1TB | Cluster, streaming mode |

## Common Workflows

### Text Curation (Most Common)

```
User: "I want to curate my text dataset"

1. Detect modality (text)
2. Ask about deduplication preference
3. Ask about quality filtering
4. Generate pipeline config
5. Validate configuration
```

### Video Processing

```
User: "Process my video dataset"

1. Detect modality (video)
2. Ask about scene detection vs fixed stride
3. Ask about captioning and embedding
4. Generate pipeline config
```

## Error Handling

### Unknown Modality

If modality cannot be detected:
```
I couldn't automatically detect the data type. What kind of data are you working with?
- Text (JSONL, Parquet, TXT)
- Video (MP4, AVI, MOV)
- Image (PNG, JPEG, WebP)
- Audio (WAV, MP3, FLAC)
```

### Missing Dependencies

If NeMo Curator is not installed:
```
NeMo Curator doesn't appear to be installed. 

Use `/setup` for guided installation with environment detection,
or install manually:
  uv pip install nemo-curator[all]
```

## References

For detailed stage information, see:
- [references/STAGE_REFERENCE.md](references/STAGE_REFERENCE.md) - Complete stage catalog
- [references/RESOURCE_GUIDE.md](references/RESOURCE_GUIDE.md) - Resource optimization
- [references/MODALITY_PATTERNS.md](references/MODALITY_PATTERNS.md) - Modality patterns
