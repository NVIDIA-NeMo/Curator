# CURATOR-OS Skills

Intelligent command interface for NeMo Curator data curation pipelines, built on the 
[Agent Skills](https://agentskills.io/specification) open standard.

## Available Skills

### Setup & Installation

| Skill | Description | Invocation |
|-------|-------------|------------|
| `setup` | Install with environment detection + verification | `/setup` |
| `setup-ray` | Multi-node Ray cluster configuration | `/setup-ray` |

### Modality-Specific Skills

| Skill | Modality | Description | Invocation |
|-------|----------|-------------|------------|
| `video` | Video | Clipping, captioning, embeddings, filtering | `/video` |
| `image` | Image | CLIP embedding, aesthetic/NSFW filtering, dedup | `/image` |
| `audio` | Audio | ASR transcription, WER filtering | `/audio` |

### Text Processing Skills

| Skill | Description | Invocation |
|-------|-------------|------------|
| `filter` | Heuristic text filtering (33+ filters) | `/filter` |
| `classify` | ML classification (quality, domain, safety) | `/classify` |
| `dedup-fuzzy` | MinHash + LSH fuzzy deduplication | `/dedup-fuzzy` |

### Full Workflow Skills

| Skill | Description | Invocation |
|-------|-------------|------------|
| `curator-os` | Main orchestrator - routes requests to skills | Auto (agent decides) |
| `curate` | Full curation workflow for any modality | `/curate` |

### Reference & Help

| Skill | Description | Invocation |
|-------|-------------|------------|
| `stages` | Search and browse processing stages | `/stages` |
| `help` | Context-aware help | `/help` |

## Directory Structure

```
.cursor/skills/
├── README.md                            # This file
│
├── # Shared Utilities
├── shared/                              # Shared Python utilities
│   ├── __init__.py
│   └── introspect.py                    # Stage discovery & introspection
│
├── # Setup Skills
├── setup/                               # Installation + verification
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── detect_environment.py
│   │   └── verify_installation.py
│   └── references/
├── setup-ray/                           # Multi-node cluster configuration
│   └── SKILL.md
│
├── # Modality Skills
├── video/                               # Video processing
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── generate_video_config.py
│   │   ├── estimate_video_resources.py
│   │   └── list_video_stages.py
│   └── references/
│       ├── CLIPPING_OPTIONS.md
│       └── CAPTIONING_GUIDE.md
├── image/                               # Image processing
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── generate_image_config.py
│   │   ├── estimate_image_resources.py
│   │   └── list_image_stages.py
│   └── references/
│       └── FILTERING_THRESHOLDS.md
├── audio/                               # Audio processing
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── generate_audio_config.py
│   │   ├── list_audio_stages.py
│   │   └── list_asr_models.py
│   └── references/
│       └── ASR_MODELS.md
│
├── # Curation Skills
├── curator-os/                          # Main orchestrator
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── detect_modality.py
│   │   └── validate_pipeline.py
│   └── references/
├── curate/                              # Full curation workflow
│   ├── SKILL.md
│   ├── scripts/
│   │   └── generate_yaml.py
│   └── assets/
├── dedup-fuzzy/                         # Fuzzy deduplication
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── generate_fuzzy_config.py
│   │   └── estimate_resources.py
│   └── references/
├── filter/                              # Heuristic filtering
│   ├── SKILL.md
│   └── scripts/
│       └── list_filters.py
├── classify/                            # ML classification
│   └── SKILL.md
│
├── # Reference Skills
├── stages/                              # Stage reference lookup
│   ├── SKILL.md
│   └── scripts/
│       └── search_stages.py
└── help/                                # Context-aware help
    └── SKILL.md
```

## Quick Start

### 1. Install NeMo Curator

Use the `/setup` skill for guided installation:

```
/setup
```

Or install manually:

```bash
uv pip install nemo-curator[all]
```

### 2. Process Data by Modality

**Video:**
```bash
python .cursor/skills/video/scripts/generate_video_config.py \
  --input-path /data/videos \
  --output-path /data/clips \
  --caption --embed
```

**Image:**
```bash
python .cursor/skills/image/scripts/generate_image_config.py \
  --input-path /data/images \
  --output-path /data/curated \
  --aesthetic-threshold 0.5
```

**Audio:**
```bash
python .cursor/skills/audio/scripts/generate_audio_config.py \
  --input-path /data/audio \
  --output-path /data/transcribed \
  --model-name nvidia/stt_en_fastconformer_hybrid_large_pc
```

**Text:**
```bash
python .cursor/skills/curate/scripts/generate_yaml.py \
  --modality text \
  --input-path /data/text \
  --output-path /data/curated
```

### 3. Run Pipeline

```bash
python -m nemo_curator.config.run --config-path=. --config-name=pipeline
```

## Scripts

All scripts support `--help`:

```bash
# Video
python .cursor/skills/video/scripts/list_video_stages.py --verbose
python .cursor/skills/video/scripts/estimate_video_resources.py --input-path /data/videos
python .cursor/skills/video/scripts/generate_video_config.py --help

# Image
python .cursor/skills/image/scripts/list_image_stages.py --verbose
python .cursor/skills/image/scripts/generate_image_config.py --help

# Audio
python .cursor/skills/audio/scripts/list_asr_models.py
python .cursor/skills/audio/scripts/list_audio_stages.py --verbose
python .cursor/skills/audio/scripts/generate_audio_config.py --help

# Text/General
python .cursor/skills/filter/scripts/list_filters.py --verbose
python .cursor/skills/stages/scripts/search_stages.py --modality video
python .cursor/skills/dedup-fuzzy/scripts/estimate_resources.py --input-path /data/text
```

## Requirements

- Python 3.10+
- NeMo Curator >= 0.5.0
- Linux (required)
- GPU with 4-24GB memory depending on stages

## License

Apache-2.0 (same as NeMo Curator)
