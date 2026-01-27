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

```
/setup
```

The setup skill detects your environment, recommends installation options, and verifies everything works.

### 2. Process Data by Modality

Invoke the skill for your data type and describe what you want:

**Video:**
```
/video

I have videos in /data/videos. Generate clips with captions and embeddings.
```

**Image:**
```
/image

Filter images in /data/images for quality (aesthetic > 0.5) and safety.
```

**Audio:**
```
/audio

Transcribe audio files in /data/audio using FastConformer.
```

**Text:**
```
/curate

Deduplicate and filter text in /data/text for LLM training.
```

### 3. Get Help

```
/help
```

Context-aware help based on your current task.

## How Skills Work

Skills are **not** scripts you run manually. When you invoke a skill:

1. The agent reads the `SKILL.md` instructions
2. Uses bundled `scripts/` and `references/` as needed
3. Generates configs, estimates resources, or runs pipelines for you

The `scripts/` directories contain utilities the agent executes on your behalf.

## Requirements

- Python 3.10+
- NeMo Curator >= 0.5.0
- Linux (required)
- GPU with 4-24GB memory depending on stages

## License

Apache-2.0 (same as NeMo Curator)
