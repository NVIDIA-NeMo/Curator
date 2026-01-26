# CURATOR-OS Skills

Intelligent command interface for NeMo Curator data curation pipelines, built on the 
[Agent Skills](https://agentskills.io/specification) open standard.

## Available Skills

### Setup & Installation

| Skill | Description | Invocation |
|-------|-------------|------------|
| `setup` | Install with environment detection + verification | `/setup` |
| `setup-ray` | Ray cluster configuration | `/setup-ray` |

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
dori/
├── README.md                            # This file
│
├── # Setup Skills
├── setup/                               # Main installation (consolidated)
│   ├── SKILL.md                         # Action-oriented workflow
│   ├── scripts/
│   │   ├── detect_environment.py        # CUDA/GPU/FFmpeg detection
│   │   └── verify_installation.py       # Comprehensive verification
│   └── references/
│       ├── TEXT_PACKAGES.md             # Text dependencies
│       ├── VIDEO_PACKAGES.md            # Video + FFmpeg deps
│       ├── AUDIO_PACKAGES.md            # Audio + NeMo ASR deps
│       ├── IMAGE_PACKAGES.md            # Image + DALI deps
│       └── TROUBLESHOOTING.md           # Common issues & fixes
├── setup-ray/                           # Ray cluster setup
│   └── SKILL.md
├── setup-verify/                        # Redirects to /setup
│   └── SKILL.md
│
├── # Curation Skills
├── curator-os/                          # Main orchestrator (auto-invoked)
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── detect_modality.py           # Detect text/video/image/audio
│   │   └── validate_pipeline.py         # Validate YAML configs
│   └── references/
│       ├── STAGE_REFERENCE.md           # Complete stage catalog (80+ stages)
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
curl -LsSf https://astral.sh/uv/0.8.22/install.sh | sh
source $HOME/.local/bin/env

# Create environment and install
uv venv && source .venv/bin/activate
uv pip install torch wheel_stub psutil setuptools
echo "transformers==4.55.2" > override.txt
uv pip install --no-build-isolation "nemo-curator[all]" --override override.txt
```

### 2. Verify Installation

```bash
python dori/setup/scripts/verify_installation.py
```

### 3. Start Curating

```
User: I want to curate my text dataset at /data/raw

Agent: [Activates curator-os, detects text modality]

Recommended workflow:
1. Heuristic filtering (25 filters)
2. Quality classification
3. Fuzzy deduplication

Generate pipeline config? [y/n]
```

Or explicitly invoke:

```
/curate --modality text --input-path /data/raw --output-path /data/curated
```

### Fuzzy Deduplication

```
/dedup-fuzzy

Input path: /data/text
Output path: /data/deduped
Cache path: /data/cache
```

### Stage Lookup

```
/stages --modality text --category filters

📁 text/filters
--------------------------------------------------
  • WordCountFilter: Filter by word count (50-100000)
  • NonAlphaNumericFilter: Filter by non-alphanumeric ratio
  ... (33 filters total)
```

## Scripts

All scripts are executable Python with `--help` support:

```bash
# Detect environment (CUDA, GPU, FFmpeg)
python dori/setup/scripts/detect_environment.py
python dori/setup/scripts/detect_environment.py --modality video --human

# Verify installation
python dori/setup/scripts/verify_installation.py
python dori/setup/scripts/verify_installation.py --text  # Text only
python dori/setup/scripts/verify_installation.py --video # Video only

# Detect data modality
python dori/curator-os/scripts/detect_modality.py /path/to/data

# Validate pipeline config
python dori/curator-os/scripts/validate_pipeline.py config.yaml

# Generate fuzzy dedup config
python dori/dedup-fuzzy/scripts/generate_fuzzy_config.py \
  --input-path /data/text \
  --output-path /data/deduped \
  --cache-path /data/cache \
  --output-file fuzzy_dedup.yaml

# Estimate resources for dedup
python dori/dedup-fuzzy/scripts/estimate_resources.py \
  --input-path /data/text

# Generate curation pipeline
python dori/curate/scripts/generate_yaml.py \
  --modality text \
  --input-path /data/raw \
  --output-path /data/curated \
  --output-file pipeline.yaml

# Search stages
python dori/stages/scripts/search_stages.py --modality video --category embedding
python dori/stages/scripts/search_stages.py --search "caption"

# List filters
python dori/filter/scripts/list_filters.py --verbose
python dori/filter/scripts/list_filters.py --category repetition
```

## Running Generated Configs

After generating a YAML configuration:

```bash
python -m nemo_curator.config.run \
  --config-path=. \
  --config-name=<config-file-stem> \
  input_path=/data/raw \
  output_path=/data/curated
```

## Requirements

- Python 3.10+
- NeMo Curator >= 0.5.0
- Ray cluster for distributed execution
- GPU with 16GB+ memory for ML stages

## Related Documentation

- [RFC-001: CURATOR-OS](../docs/rfc/rfc-001-curator-os.md) - Design document
- [NeMo Curator Documentation](https://docs.nvidia.com/nemo-curator)
- [Agent Skills Specification](https://agentskills.io/specification)

## License

Apache-2.0 (same as NeMo Curator)
