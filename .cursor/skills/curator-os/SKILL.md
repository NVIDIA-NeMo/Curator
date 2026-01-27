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

## Skill Routing

Route user requests to the appropriate skill based on intent and modality:

| User Intent | Skill | When to Use |
|-------------|-------|-------------|
| "process videos" / "video dataset" | `/video` | Video clipping, captioning, embedding |
| "curate images" / "filter images" | `/image` | Image quality filtering, NSFW, dedup |
| "transcribe audio" / "ASR" | `/audio` | Audio transcription, WER filtering |
| "filter text" / "clean text" | `/filter` | Heuristic text filtering |
| "classify documents" / "quality score" | `/classify` | ML-based classification |
| "deduplicate" / "remove duplicates" | `/dedup-fuzzy` | Near-duplicate removal |
| "full curation" / "end-to-end" | `/curate` | Complete pipeline |
| "what stages are available" | `/stages` | Stage discovery |
| "help with setup" | `/setup` | Installation and config |

## Available Skills

### Modality-Specific Skills

| Skill | Modality | Purpose |
|-------|----------|---------|
| `/video` | Video | Clipping, captioning, embeddings, filtering |
| `/image` | Image | CLIP embedding, aesthetic/NSFW filtering, dedup |
| `/audio` | Audio | ASR transcription, WER filtering |

### Text Processing Skills

| Skill | Purpose |
|-------|---------|
| `/filter` | Heuristic text filtering (33+ filters) |
| `/classify` | ML classification (quality, domain, safety) |
| `/dedup-fuzzy` | MinHash + LSH deduplication |
| `/dedup-exact` | Hash-based exact deduplication |
| `/dedup-semantic` | Embedding-based semantic deduplication |

### Utility Skills

| Skill | Purpose |
|-------|---------|
| `/curate` | Full curation workflow (all modalities) |
| `/stages` | Stage reference and discovery |
| `/setup` | Environment detection and installation |
| `/help` | Context-aware help |

## Modality Detection

Before recommending a workflow, detect the data modality:

```bash
python scripts/detect_modality.py /path/to/data
```

### Modality Detection Patterns

| Pattern | Modality | Route To |
|---------|----------|----------|
| `.mp4`, `.avi`, `.mov`, `.mkv` | Video | `/video` |
| `.jpg`, `.png`, `.webp`, `.tar` (WebDataset) | Image | `/image` |
| `.wav`, `.mp3`, `.flac` | Audio | `/audio` |
| `.jsonl`, `.parquet`, `.txt` | Text | `/filter`, `/classify`, `/curate` |

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

### Video Processing

```
User: "Process my video dataset"

1. Route to /video skill
2. Ask about scene detection vs fixed stride
3. Ask about captioning and embedding
4. Generate pipeline config with generate_video_config.py
```

### Image Curation

```
User: "Filter my image dataset for quality"

1. Route to /image skill
2. Configure aesthetic and NSFW thresholds
3. Ask about deduplication
4. Generate pipeline config with generate_image_config.py
```

### Audio Transcription

```
User: "Transcribe and filter audio"

1. Route to /audio skill
2. Select ASR model based on language
3. Set WER threshold
4. Generate pipeline config with generate_audio_config.py
```

### Text Curation (Most Common)

```
User: "I want to curate my text dataset"

1. Detect modality (text)
2. Ask about deduplication preference → /dedup-fuzzy
3. Ask about quality filtering → /filter
4. Generate pipeline config
5. Validate configuration
```

## Error Handling

### Unknown Modality

If modality cannot be detected:
```
I couldn't automatically detect the data type. What kind of data are you working with?
- Text (JSONL, Parquet, TXT) → /filter, /classify, /curate
- Video (MP4, AVI, MOV) → /video
- Image (PNG, JPEG, WebDataset) → /image
- Audio (WAV, MP3, FLAC) → /audio
```

### Missing Dependencies

If NeMo Curator is not installed:
```
NeMo Curator doesn't appear to be installed. 

Use /setup for guided installation with environment detection,
or install manually:
  uv pip install nemo-curator[all]
```

## References

For detailed stage information, see:
- [references/STAGE_REFERENCE.md](references/STAGE_REFERENCE.md) - Complete stage catalog
- [references/RESOURCE_GUIDE.md](references/RESOURCE_GUIDE.md) - Resource optimization
- [references/MODALITY_PATTERNS.md](references/MODALITY_PATTERNS.md) - Modality patterns
