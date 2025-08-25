> [!note]
> This document is still a work in progress and may change frequently.

## Setup and Dev

### Prerequisites

- Python >=3.10, < 3.13
- OS: Ubuntu 22.04/20.04
- NVIDIA GPU (optional)
  - Volta™ or higher (compute capability 7.0+)
  - CUDA 12.x
- uv

```
# We use `uv` for package management and environment isolation.
pip3 install uv

# If you cannot install at the system level, you can install for your user with
pip3 install --user uv
```

### Installation

NeMo Curator uses [uv](https://docs.astral.sh/uv/) for package management.

You can configure uv with the following commands:

```bash
uv sync
```

You can additionally sync optional dependency groups:

```bash
uv sync --extra text

# Sync multiple dependency groups
uv sync --extra text --extra video

# Sync all (includes dali, deduplication_cuda12x, text, video, video_cuda)
uv sync --extra all
```

### Dev Pattern

- All upstream work/changes with the new API and ray backend should target the `NeMo-Curator/ray-api` branch.
- When re-using code already in `NeMo-Curator/nemo_curator`, use `git mv` to move those source files into the `ray-curator/ray_curator` namespace.
- Sign and signoff commits with `git commit -sS`. (May be relaxed in the future)
- If project dependencies are updated a new uv lock file needs to be generated. Run `uv lock` and add the changes of the new uv.lock file.

### Testing

Work in Progress...
