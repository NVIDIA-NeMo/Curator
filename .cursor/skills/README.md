# CURATOR-OS Skills

Intelligent command interface for NeMo Curator data curation pipelines, built on the 
[Agent Skills](https://agentskills.io/specification) open standard.

## Available Skills

### Setup & Installation

| Skill | Description | Invocation |
|-------|-------------|------------|
| `setup` | Install with environment detection + verification | `/setup` |
| `setup-ray` | Multi-node Ray cluster configuration | `/setup-ray` |

### Curation Workflows

| Skill | Description | Invocation |
|-------|-------------|------------|
| `curator-os` | Main orchestrator - routes requests to appropriate skills | Auto (agent decides) |
| `curate` | Full curation workflow for any modality | `/curate` |
| `dedup-fuzzy` | MinHash + LSH fuzzy deduplication | `/dedup-fuzzy` |
| `filter` | Heuristic text filtering | `/filter` |
| `classify` | ML classification (quality, domain, safety) | `/classify` |

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
│   │   ├── detect_environment.py        # CUDA/GPU/FFmpeg detection
│   │   └── verify_installation.py       # Comprehensive verification
│   └── references/
│       ├── TEXT_PACKAGES.md             # Text dependencies
│       ├── VIDEO_PACKAGES.md            # Video + FFmpeg deps
│       ├── AUDIO_PACKAGES.md            # Audio + NeMo ASR deps
│       ├── IMAGE_PACKAGES.md            # Image + DALI deps
│       └── TROUBLESHOOTING.md           # Common issues & fixes
├── setup-ray/                           # Multi-node cluster configuration
│   └── SKILL.md
│
├── # Curation Skills
├── curator-os/                          # Main orchestrator (auto-invoked)
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── detect_modality.py           # Detect text/video/image/audio
│   │   └── validate_pipeline.py         # Validate YAML configs
│   └── references/
│       ├── STAGE_REFERENCE.md           # Complete stage catalog
│       ├── RESOURCE_GUIDE.md            # GPU/memory optimization
│       └── MODALITY_PATTERNS.md         # Pipeline patterns by modality
├── curate/                              # Full curation workflow
│   ├── SKILL.md
│   ├── scripts/
│   │   └── generate_yaml.py             # Generate pipeline configs
│   └── assets/
│       ├── text_curation_template.yaml
│       └── video_curation_template.yaml
├── dedup-fuzzy/                         # Fuzzy deduplication
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── generate_fuzzy_config.py
│   │   └── estimate_resources.py
│   └── references/
│       └── FUZZY_DEDUP_PARAMS.md
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

Use the `/setup` skill for guided installation with automatic environment detection:

```
/setup
```

Or install manually:

```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create environment and install
uv venv && source .venv/bin/activate
uv pip install nemo-curator[text_cuda12]
```

### 2. Verify Installation

```bash
python .cursor/skills/setup/scripts/verify_installation.py
```

### 3. Start Curating

```
User: I want to curate my text dataset at /data/raw

Agent: [Activates curator-os, detects text modality]

Recommended workflow:
1. Heuristic filtering
2. Quality classification
3. Fuzzy deduplication

Generate pipeline config? [y/n]
```

Or explicitly invoke:

```
/curate --modality text --input-path /data/raw --output-path /data/curated
```

## Scripts

All scripts support `--help`:

```bash
# Environment detection
python .cursor/skills/setup/scripts/detect_environment.py --quick

# Installation verification
python .cursor/skills/setup/scripts/verify_installation.py
python .cursor/skills/setup/scripts/verify_installation.py --text
python .cursor/skills/setup/scripts/verify_installation.py --video

# Modality detection
python .cursor/skills/curator-os/scripts/detect_modality.py /path/to/data

# Pipeline validation
python .cursor/skills/curator-os/scripts/validate_pipeline.py config.yaml

# Fuzzy dedup config generation
python .cursor/skills/dedup-fuzzy/scripts/generate_fuzzy_config.py \
  --input-path /data/text \
  --output-path /data/deduped \
  --cache-path /data/cache

# Resource estimation
python .cursor/skills/dedup-fuzzy/scripts/estimate_resources.py \
  --input-path /data/text

# Pipeline generation
python .cursor/skills/curate/scripts/generate_yaml.py \
  --modality text \
  --input-path /data/raw \
  --output-path /data/curated

# Stage search
python .cursor/skills/stages/scripts/search_stages.py --modality video
python .cursor/skills/stages/scripts/search_stages.py --search "caption"

# Filter listing
python .cursor/skills/filter/scripts/list_filters.py --verbose
```

## Requirements

- Python 3.10+
- NeMo Curator >= 0.5.0
- GPU with 16GB+ memory for ML stages
- Ray cluster for distributed execution (optional, see `/setup-ray`)

## Related Documentation

- [NeMo Curator Documentation](https://docs.nvidia.com/nemo-curator)
- [Agent Skills Specification](https://agentskills.io/specification)

## License

Apache-2.0 (same as NeMo Curator)
