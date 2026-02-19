# AGENTS.md - NeMo Curator

Instructions for AI coding agents working on NeMo Curator.

## Project Overview

NeMo Curator is a scalable data curation library for training large language models. It supports text, image, video, and audio data processing with distributed execution via Ray.

## Setup Commands

```bash
# Install dependencies (requires Python 3.10-3.12)
uv pip install -e ".[all]"

# Or use Docker (recommended)
# Get latest tag: python skills/setup/scripts/get_latest_tag.py
docker pull nvcr.io/nvidia/nemo-curator:$(python skills/setup/scripts/get_latest_tag.py)
docker run --rm --shm-size=8g -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/nemo-curator:$(python skills/setup/scripts/get_latest_tag.py)
```

**Finding the latest Docker tag**: Run `python skills/setup/scripts/get_latest_tag.py` to get the latest version from NGC. Tags follow YY.MM format (e.g., `25.09` = September 2025). Check [NGC Container Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator) for all available versions.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/stages/text/test_filters.py

# Run with coverage
pytest --cov=nemo_curator tests/
```

## Code Style

- **Formatter/Linter**: Ruff (line length 119)
- **Type hints**: Required for all functions (except `*args`, `**kwargs`)
- **Docstrings**: Google style (not enforced but preferred)
- **Imports**: Sorted by Ruff

```bash
# Format code
ruff format .

# Lint code
ruff check . --fix
```

## Key Conventions

### Copyright Header
All Python files must include:
```python
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
```

### Stage Pattern
New processing stages should:
1. Inherit from `ProcessingStage` or `CompositeStage`
2. Implement `inputs()`, `outputs()`, and `process()` methods
3. Use `Resources` dataclass for GPU/CPU requirements

```python
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources

class MyStage(ProcessingStage[InputType, OutputType]):
    name: str = "MyStage"
    resources: Resources = Resources(cpus=1.0)
    
    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["text"]
    
    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["processed_text"]
    
    def process(self, task: InputType) -> OutputType:
        # Transform task
        return output_task
```

### Test Requirements
- All source changes require tests
- GPU tests must be marked with `@pytest.mark.gpu`
- Test directory structure mirrors `nemo_curator/` structure
- 80% coverage enforced

## Directory Structure

```
nemo_curator/
├── stages/           # Processing stages
│   ├── text/        # Text processing (filters, classifiers, etc.)
│   ├── image/       # Image processing
│   ├── video/       # Video processing
│   ├── audio/       # Audio processing
│   └── deduplication/  # Shared dedup stages
├── pipeline/         # Pipeline orchestration
├── backends/         # Execution backends (Ray, Xenna)
├── tasks/           # Task types (DocumentBatch, VideoTask, etc.)
└── utils/           # Utilities
```

## Common Tasks

### Adding a New Filter

1. Create filter class in `nemo_curator/stages/text/filters/`
2. Inherit from `DocumentFilter`
3. Implement `score_document()` and `keep_document()`
4. Add to `__init__.py` exports
5. Add tests in `tests/stages/text/`

### Adding a New Classifier

1. Create classifier in `nemo_curator/stages/text/classifiers/`
2. Inherit from appropriate base class
3. Specify GPU requirements via `Resources`
4. Add HuggingFace model handling if needed
5. Add tests (mark GPU tests with `@pytest.mark.gpu`)

## Docker Notes

- Always use `--shm-size=8g` (or 30% of RAM) for Ray
- Use versioned tags (not `:latest`). Get latest: `python skills/setup/scripts/get_latest_tag.py`
- Mount workspace: `-v $(pwd):/workspace -w /workspace`

## Cursor Skills

This repo includes Cursor agent skills in `skills/` for data curation tasks:
- `/curator-os` - Main curation assistant
- `/text`, `/video`, `/image`, `/audio` - Modality-specific curation
- `/setup` - Environment setup help

See `skills/curator-os/SKILL.md` for the full workflow.

## Links

- [Documentation](https://docs.nvidia.com/nemo/curator/)
- [NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator)
- [GitHub](https://github.com/NVIDIA/NeMo-Curator)
