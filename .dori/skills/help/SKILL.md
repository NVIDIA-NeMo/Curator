---
name: help
description: Command reference and guidance for NeMo Curator agent skills. Use when the user asks for help, wants to see available commands, or needs guidance on which skill to use.
license: Apache-2.0
metadata:
  author: nvidia
  version: "2.0"
  type: reference
---

# NeMo Curator Help

Quick reference for all available commands and skills.

## Quick Start

```
/curator    → Main entry point - routes to any modality
/setup      → Install NeMo Curator, verify environment
/text       → Curate text data
/video      → Curate video data
/image      → Curate image data
/audio      → Curate audio data
```

## All Commands

### Text Curation

| Command | GPU | Description |
|---------|-----|-------------|
| `/text` | varies | Main entry point for text (includes full pipeline templates) |
| `/txt-filter` | No | Heuristic filtering (word count, URLs, repetition) |
| `/txt-classify` | Yes | ML classification (quality, domain, safety) |
| `/txt-dedup` | Yes* | Fuzzy deduplication (MinHash + LSH) |

*GPU recommended for large datasets

### Video Curation

| Command | GPU | Description |
|---------|-----|-------------|
| `/video` | Yes | Main entry point for video curation |
| `/vid-clip` | Yes | Extract clips (scene detection or fixed stride) |
| `/vid-caption` | Yes | Generate text descriptions |
| `/vid-embed` | Yes | Cosmos/InternVideo embeddings |

### Image Curation

| Command | GPU | Description |
|---------|-----|-------------|
| `/image` | Yes | Main entry point for image curation |
| `/img-embed` | Yes | CLIP embeddings for search/dedup |
| `/img-aesthetic` | Yes | Quality scoring and filtering |
| `/img-nsfw` | Yes | Content safety filtering |

### Audio Curation

| Command | GPU | Description |
|---------|-----|-------------|
| `/audio` | Yes | Main entry point for audio curation |
| `/aud-asr` | Yes | ASR transcription (Parakeet, Canary) |
| `/aud-wer` | Yes | WER-based quality filtering |

### Setup & Project

| Command | Description |
|---------|-------------|
| `/setup` | Install NeMo Curator, check GPU |
| `/project` | Set up directory structure, file formats |
| `/help` | This help (you're here!) |

### Schema & Validation

| Command | Description |
|---------|-------------|
| `/schema` | Query available operations and parameters |
| `/validate` | Validate code against the schema |
| `/schema-update` | Regenerate schema (developers) |

## YAML CLI (No Python Required)

For standard pipelines, use built-in YAML configs instead of writing Python:

```bash
# Run full English text filtering (20+ filters)
python -m nemo_curator.config.run \
  --config-path ./text \
  --config-name heuristic_filter_english_pipeline.yaml \
  input_path=/data/input output_path=/data/output
```

Available configs in `nemo_curator/config/text/`:
- `heuristic_filter_english_pipeline.yaml` - Full English filtering
- `heuristic_filter_non_english_pipeline.yaml` - Non-English filtering  
- `fuzzy_deduplication_pipeline.yaml` - MinHash dedup
- `exact_deduplication_pipeline.yaml` - Hash-based dedup
- `fasttext_filter_pipeline.yaml` - Language detection

See `/curator` for custom YAML config examples.

## Common Workflows

### "I'm new, where do I start?"
```
/setup        → Verify NeMo Curator is installed
/project      → Set up your directory structure
/text         → Start with text curation (or /video, /image, /audio)
```

### "I want to curate text for LLM training"
```
/text         → Full pipeline templates and guidance
/txt-filter   → Remove obvious junk (CPU, fast)
/txt-classify → Score quality with ML (GPU)
/txt-dedup    → Remove duplicates (GPU)
```

### "I want to process videos"
```
/vid-clip     → Extract clips from videos
/vid-caption  → Generate captions
/vid-embed    → Create embeddings for search
```

Or use `/video` for guidance on combining stages.

### "I want to filter images"
```
/img-aesthetic → Keep high-quality images
/img-nsfw      → Remove inappropriate content
/img-embed     → Generate CLIP embeddings
```

### "I want to transcribe audio"
```
/aud-asr    → Transcribe with NeMo ASR
/aud-wer    → Filter by transcription quality
```

## Active Tools (Agent Workflow)

The agent uses these tools to help plan and debug pipelines:

| Tool | Command | Purpose |
|------|---------|---------|
| **Analyze Data** | `python skills/shared/scripts/analyze_data.py --input data.jsonl --sample 100` | Get data-driven filter recommendations |
| **Validate Pipeline** | `python skills/shared/scripts/validate_pipeline.py --stages "Stage1,Stage2"` | Check type flow, GPU, credentials before generating |
| **Test Pipeline** | `python skills/shared/scripts/test_pipeline.py --stages "..." --sample 20` | Verify pipeline works on sample data |
| **Diagnose Error** | `python skills/shared/scripts/diagnose_error.py --error "error message"` | Get actionable fixes for errors |

## Getting More Help

- **For a specific skill**: Just invoke it (e.g., `/txt-filter`)
- **For NeMo Curator docs**: https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/
- **For the agent tool schema**: `/schema`
- **To validate code**: `/validate`

## GPU Requirements Summary

| Category | CPU-Only | GPU Required |
|----------|----------|--------------|
| Text | `/txt-filter` | `/txt-classify`, `/txt-dedup` |
| Video | — | All stages |
| Image | — | All stages |
| Audio | — | All stages |
